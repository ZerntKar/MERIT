from __future__ import annotations

import random
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import MeritConfig
from .data import (
    MeritJsonlDataset,
    SequenceJsonlDataset,
    collate_merit,
    collate_sequences,
)
from .losses import merit_loss
from .metrics import ranking_metrics
from .model import MERITReranker
from .retriever import SASRec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def reranker_train_step(
    model: MERITReranker,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: MeritConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["candidate_support"],
        batch["preference_strength"],
        batch["semantic_embeddings"],
        batch["matching_features"],
        batch["candidate_mask"],
        batch["evidence_mask"],
    )
    losses = merit_loss(
        output,
        batch["target_index"],
        config.mceb.rerank_temperature,
        config.mceb.stable_weight,
        config.mceb.fake_weight,
    )
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.reranker_training.gradient_clip)
    optimizer.step()
    return {key: value.detach().item() for key, value in losses.items()}


@torch.no_grad()
def evaluate_reranker(
    model: MERITReranker,
    loader: Iterable[dict],
    device: torch.device,
    cutoffs: list[int],
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    count = 0
    for raw_batch in loader:
        batch = tensor_batch_to_device(raw_batch, device)
        output = model(
            batch["candidate_support"],
            batch["preference_strength"],
            batch["semantic_embeddings"],
            batch["matching_features"],
            batch["candidate_mask"],
            batch["evidence_mask"],
            augment_pseudo=False,
            compute_stability=False,
        )
        metrics = ranking_metrics(output.scores, batch["target_index"], cutoffs)
        batch_size = batch["candidate_support"].shape[0]
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in metric_sums.items()}


def fit_reranker(
    config: MeritConfig,
    train_path: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> dict[str, float]:
    seed_everything(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    collate = lambda rows: collate_merit(rows, config.data.candidate_pool_size)
    train_loader = DataLoader(
        MeritJsonlDataset(train_path),
        batch_size=config.reranker_training.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        MeritJsonlDataset(validation_path),
        batch_size=config.reranker_training.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = MERITReranker(config).to(target_device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.reranker_training.learning_rate,
        weight_decay=config.reranker_training.weight_decay,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")
    best_results: dict[str, float] = {}
    stale_epochs = 0
    for epoch in range(config.reranker_training.max_epochs):
        for raw_batch in train_loader:
            batch = tensor_batch_to_device(raw_batch, target_device)
            reranker_train_step(model, batch, optimizer, config)
        results = evaluate_reranker(
            model, validation_loader, target_device, config.evaluation.cutoffs
        )
        selection_metric = results.get("NDCG@5", next(iter(results.values()), 0.0))
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_results = results
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": results},
                output_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.reranker_training.patience:
                break
    return best_results


def fit_retriever(
    config: MeritConfig,
    train_path: str | Path,
    num_items: int,
    output_path: str | Path,
    device: str | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    epochs: int = 50,
) -> float:
    seed_everything(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        SequenceJsonlDataset(train_path),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_sequences(rows, config.data.max_sequence_length),
    )
    model = SASRec(num_items, config).to(target_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for raw_batch in loader:
            batch = tensor_batch_to_device(raw_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.next_item_loss(batch["item_sequence"], batch["target_sequence"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.detach().item()
            count += 1
        final_loss = total / max(count, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_items": num_items}, output_path)
    return final_loss


def fit_reranker_seeds(
    config: MeritConfig,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_seed = config.seed
    results = []
    for seed in config.seeds:
        config.seed = int(seed)
        metrics = fit_reranker(
            config,
            train_path,
            validation_path,
            output / f"mceb_seed_{seed}.pt",
            device=device,
        )
        results.append({"seed": seed, **metrics})
    config.seed = original_seed
    keys = sorted({key for row in results for key in row if key != "seed"})
    aggregate = {}
    for key in keys:
        values = [float(row[key]) for row in results]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    payload = {"runs": results, "aggregate": aggregate}
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
