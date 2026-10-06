from __future__ import annotations

import argparse
import json

import torch

from .baselines import intersection_only, preference_gap
from .config import load_config
from .losses import merit_loss
from .matching import build_local_contrast
from .metrics import comprehensiveness, ranking_metrics, top1_reconstruction
from .model import MERITReranker
from .generation_pipeline import (
    build_generation_plans,
    fit_generator,
    generate_student_outputs,
    generate_teacher_targets,
)
from .pipeline import (
    build_candidate_pools,
    build_evidence_vocabulary,
    encode_vocabulary,
    materialize_merit_examples,
)
from .preprocess import prepare_chronological_splits
from .training import fit_reranker, fit_reranker_seeds, fit_retriever, seed_everything


def smoke_test(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    seed_everything(config.seed)
    batch = 2
    candidates = config.data.candidate_pool_size
    evidence = max(config.mceb.evidence_budget + 3, 12)
    support = torch.rand(batch, candidates, evidence)
    preference = torch.rand(batch, evidence)
    semantic = torch.randn(batch, evidence, config.mceb.semantic_dim)
    matching = torch.randn(batch, candidates, 3)
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    evidence_mask = torch.ones(batch, evidence, dtype=torch.bool)
    target = torch.tensor([1, min(7, candidates - 1)])

    model = MERITReranker(config)
    model.train()
    output = model(
        support,
        preference,
        semantic,
        matching,
        candidate_mask,
        evidence_mask,
    )
    losses = merit_loss(
        output,
        target,
        config.mceb.rerank_temperature,
        config.mceb.stable_weight,
        config.mceb.fake_weight,
    )
    losses["total"].backward()
    if not torch.isfinite(losses["total"]):
        raise AssertionError("non-finite MERIT loss")
    score_error = (output.scores - output.contributions.sum(dim=-1)).abs().max().item()
    if score_error > 1e-6:
        raise AssertionError(f"score decomposition failed: {score_error}")

    model.eval()
    with torch.no_grad():
        clean = model(
            support,
            preference,
            semantic,
            matching,
            candidate_mask,
            evidence_mask,
            augment_pseudo=False,
            compute_stability=False,
        )
        original_top1 = clean.scores.argmax(dim=-1)
        expressed = clean.gate.bool()
        top1 = top1_reconstruction(clean.contributions, original_top1, expressed)
        comp = comprehensiveness(clean.contributions, original_top1, expressed)
        contrast, _, _ = build_local_contrast(
            support,
            matching,
            candidate_mask,
            config.matching.neighbors,
            config.matching.contrast_sigma,
        )
        int_scores, _, _ = intersection_only(
            support,
            preference,
            candidate_mask,
            evidence_mask,
            config.mceb.evidence_budget,
        )
        pg_scores, _, _ = preference_gap(
            support,
            contrast,
            preference,
            candidate_mask,
            evidence_mask,
            config.mceb.evidence_budget,
        )
    metrics = ranking_metrics(clean.scores, target, [5])
    return {
        "loss": losses["total"].item(),
        "pool_loss": losses["pool"].item(),
        "stable_loss": losses["stable"].item(),
        "fake_loss": losses["fake"].item(),
        "decomposition_max_error": score_error,
        "top1_reconstruction": top1,
        "comprehensiveness": comp,
        "baseline_scores_finite": float(
            torch.isfinite(int_scores).all() and torch.isfinite(pg_scores).all()
        ),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MERIT experiment commands for training and generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-test", help="check model forward and backward passes")
    smoke.add_argument("--config", default="configs/merit.yaml")
    train = subparsers.add_parser("train-reranker", help="train MCEB on prepared JSONL")
    train.add_argument("--config", default="configs/merit.yaml")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", default="checkpoints/mceb.pt")
    train.add_argument("--device")
    train_seeds = subparsers.add_parser("train-reranker-seeds")
    train_seeds.add_argument("--config", default="configs/merit.yaml")
    train_seeds.add_argument("--train", required=True)
    train_seeds.add_argument("--validation", required=True)
    train_seeds.add_argument("--output-dir", default="checkpoints/seeds")
    train_seeds.add_argument("--device")
    retrieve = subparsers.add_parser("train-retriever", help="train SASRec on sequence JSONL")
    retrieve.add_argument("--config", default="configs/merit.yaml")
    retrieve.add_argument("--train", required=True)
    retrieve.add_argument("--num-items", required=True, type=int)
    retrieve.add_argument("--output", default="checkpoints/sasrec.pt")
    retrieve.add_argument("--device")
    retrieve.add_argument("--epochs", default=50, type=int)
    prepare = subparsers.add_parser(
        "prepare-splits", help="create leakage-safe leave-one-out prediction points"
    )
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-dir", required=True)
    vocabulary = subparsers.add_parser("build-vocabulary")
    vocabulary.add_argument("--config", default="configs/merit.yaml")
    vocabulary.add_argument("--interactions", required=True)
    vocabulary.add_argument("--items", required=True)
    vocabulary.add_argument("--output", required=True)
    vocabulary.add_argument("--aliases")
    encode = subparsers.add_parser("encode-vocabulary")
    encode.add_argument("--config", default="configs/merit.yaml")
    encode.add_argument("--vocabulary", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--device")
    pools = subparsers.add_parser("build-pools")
    pools.add_argument("--config", default="configs/merit.yaml")
    pools.add_argument("--points", required=True)
    pools.add_argument("--checkpoint", required=True)
    pools.add_argument("--output", required=True)
    pools.add_argument("--training", action="store_true")
    pools.add_argument("--device")
    materialize = subparsers.add_parser("materialize-evidence")
    materialize.add_argument("--config", default="configs/merit.yaml")
    materialize.add_argument("--pools", required=True)
    materialize.add_argument("--interactions", required=True)
    materialize.add_argument("--items", required=True)
    materialize.add_argument("--vocabulary", required=True)
    materialize.add_argument("--embeddings", required=True)
    materialize.add_argument("--item-map")
    materialize.add_argument("--output", required=True)
    plans = subparsers.add_parser("build-generation-plans")
    plans.add_argument("--config", default="configs/merit.yaml")
    plans.add_argument("--data", required=True)
    plans.add_argument("--checkpoint", required=True)
    plans.add_argument("--output", required=True)
    plans.add_argument("--device")
    plans.add_argument("--domain", choices=("book", "movies", "yelp"))
    teacher = subparsers.add_parser("generate-teacher")
    teacher.add_argument("--config", default="configs/merit.yaml")
    teacher.add_argument("--plans", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--batch-size", type=int, default=1)
    generator_train = subparsers.add_parser("train-generator")
    generator_train.add_argument("--config", default="configs/merit.yaml")
    generator_train.add_argument("--train", required=True)
    generator_train.add_argument("--output-dir", required=True)
    generator_train.add_argument("--device")
    generator_run = subparsers.add_parser("generate-explanations")
    generator_run.add_argument("--config", default="configs/merit.yaml")
    generator_run.add_argument("--plans", required=True)
    generator_run.add_argument("--checkpoint-dir", required=True)
    generator_run.add_argument("--output", required=True)
    generator_run.add_argument("--device")
    args = parser.parse_args()

    if args.command == "smoke-test":
        print(json.dumps(smoke_test(args.config), indent=2, sort_keys=True))
    elif args.command == "train-reranker":
        config = load_config(args.config)
        results = fit_reranker(
            config, args.train, args.validation, args.output, device=args.device
        )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif args.command == "train-reranker-seeds":
        config = load_config(args.config)
        results = fit_reranker_seeds(
            config, args.train, args.validation, args.output_dir, device=args.device
        )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif args.command == "train-retriever":
        config = load_config(args.config)
        loss = fit_retriever(
            config,
            args.train,
            args.num_items,
            args.output,
            device=args.device,
            epochs=args.epochs,
        )
        print(json.dumps({"final_training_loss": loss}, indent=2))
    elif args.command == "prepare-splits":
        print(
            json.dumps(
                prepare_chronological_splits(args.input, args.output_dir),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-vocabulary":
        config = load_config(args.config)
        print(
            json.dumps(
                build_evidence_vocabulary(
                    args.interactions,
                    args.items,
                    args.output,
                    config.data.evidence_vocab_size,
                    aliases_path=args.aliases,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "encode-vocabulary":
        config = load_config(args.config)
        print(
            json.dumps(
                encode_vocabulary(
                    args.vocabulary,
                    args.output,
                    config.generation.semantic_encoder_name,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-pools":
        config = load_config(args.config)
        print(
            json.dumps(
                build_candidate_pools(
                    config,
                    args.points,
                    args.checkpoint,
                    args.output,
                    training=args.training,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "materialize-evidence":
        config = load_config(args.config)
        print(
            json.dumps(
                materialize_merit_examples(
                    config,
                    args.pools,
                    args.interactions,
                    args.items,
                    args.vocabulary,
                    args.embeddings,
                    args.output,
                    item_map_path=args.item_map,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "build-generation-plans":
        config = load_config(args.config)
        print(
            json.dumps(
                build_generation_plans(
                    config,
                    args.data,
                    args.checkpoint,
                    args.output,
                    device=args.device,
                    domain=args.domain,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate-teacher":
        config = load_config(args.config)
        print(
            json.dumps(
                generate_teacher_targets(
                    config, args.plans, args.output, batch_size=args.batch_size
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "train-generator":
        config = load_config(args.config)
        print(
            json.dumps(
                fit_generator(config, args.train, args.output_dir, device=args.device),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate-explanations":
        config = load_config(args.config)
        print(
            json.dumps(
                generate_student_outputs(
                    config,
                    args.plans,
                    args.checkpoint_dir,
                    args.output,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
