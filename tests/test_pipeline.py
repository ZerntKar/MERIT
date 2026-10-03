import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from merit.config import MeritConfig
from merit.generation import generation_prompt
from merit.generation_pipeline import build_generation_plans
from merit.model import MERITReranker
from merit.pipeline import (
    build_candidate_pools,
    build_evidence_vocabulary,
    materialize_merit_examples,
)
from merit.preprocess import prepare_chronological_splits
from merit.retriever import SASRec


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_temporal_materialization_and_generation_plan():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        interactions = [
            {
                "user_id": "u1",
                "item_id": f"i{index}",
                "timestamp": index,
                "rating": 4 + index % 2,
                "positive_evidence": {"alpha": 1, "beta": 1},
            }
            for index in range(1, 5)
        ]
        items = [
            {
                "item_id": f"i{index}",
                "name": f"Item {index}",
                "category": "book",
                "attribute_evidence": {"alpha": 1, "beta": 1, "gamma": index},
                "kg_evidence": {"alpha": 1},
            }
            for index in range(1, 5)
        ]
        raw_path = root / "raw.jsonl"
        items_path = root / "items.jsonl"
        _write_jsonl(raw_path, interactions)
        _write_jsonl(items_path, items)
        prepare_chronological_splits(raw_path, root / "processed")
        mapped = root / "processed/mapped_interactions.jsonl"
        vocabulary_path = root / "vocabulary.json"
        build_evidence_vocabulary(mapped, items_path, vocabulary_path, 16)
        vocabulary = json.loads(vocabulary_path.read_text())
        embeddings = np.arange(len(vocabulary["labels"]) * 4, dtype=np.float32).reshape(
            len(vocabulary["labels"]), 4
        )
        embeddings_path = root / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        pool = {
            "user_id": "u1",
            "history_items": [1],
            "target_item": 2,
            "timestamp": 2,
            "split": "train",
            "candidate_ids": [2, 3, 4],
            "retrieval_scores": [0.9, 0.8, 0.7],
            "target_index": 0,
            "target_retrieval_score": 0.9,
        }
        pools_path = root / "pools.jsonl"
        _write_jsonl(pools_path, [pool])
        config = MeritConfig()
        config.data.candidate_pool_size = 3
        config.mceb.semantic_dim = 4
        config.mceb.hidden_dim = 16
        config.mceb.num_heads = 4
        config.mceb.ffn_dim = 32
        config.mceb.evidence_budget = 2
        merit_path = root / "merit.jsonl"
        stats = materialize_merit_examples(
            config,
            pools_path,
            mapped,
            items_path,
            vocabulary_path,
            embeddings_path,
            merit_path,
            item_map_path=root / "processed/item_map.json",
        )
        assert stats["written_examples"] == 1
        example = json.loads(merit_path.read_text())
        assert example["popularities"][0] == 0
        assert len(example["evidence_labels"]) >= 2
        reranker = MERITReranker(config)
        checkpoint = root / "reranker.pt"
        torch.save({"model": reranker.state_dict()}, checkpoint)
        plan_path = root / "plans.jsonl"
        plan_stats = build_generation_plans(
            config,
            merit_path,
            checkpoint,
            plan_path,
            device="cpu",
            batch_size=1,
            domain="book",
        )
        assert plan_stats["examples"] == 1
        plan = json.loads(plan_path.read_text())
        assert len(plan["selected_labels"]) == 2
        assert len(plan["selected_context"]) == 2
        assert plan["domain"] == "book"
        assert "selected_evidence_support=" in plan["item_information"]
        assert "READER PREFERENCE SUMMARY" in plan["generation_prompt"]


def test_domain_specific_generation_prompts():
    cases = {
        "book": ("READER PREFERENCE SUMMARY", "competing books"),
        "Movies&TV": ("VIEWER PREFERENCE SUMMARY", "competing titles"),
        "yelp": ("USER PREFERENCE SUMMARY", "competing businesses"),
    }
    for domain, expected in cases.items():
        prompt = generation_prompt(domain, "quiet", "name=x", ["nearby"], [0.25])
        assert expected[0] in prompt
        assert expected[1] in prompt
        assert "within 80 generated tokens" in prompt
        assert "nearby: +0.25" in prompt


def test_candidate_pool_preserves_true_target_score():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = MeritConfig()
        config.data.candidate_pool_size = 2
        config.data.max_sequence_length = 5
        model = SASRec(4, config)
        model.eval()
        sequence = torch.tensor([[1, 0, 0, 0, 0]])
        candidates = torch.tensor([2, 3, 4])
        scores = model.score_targets(sequence, candidates)
        target = int(candidates[scores.argmin()].item())
        expected = float(scores.min().item())
        checkpoint = root / "sasrec.pt"
        torch.save({"model": model.state_dict(), "num_items": 4}, checkpoint)
        points = root / "points.jsonl"
        _write_jsonl(
            points,
            [
                {
                    "user_id": "u",
                    "history_items": [1],
                    "target_item": target,
                    "timestamp": 2,
                    "split": "train",
                }
            ],
        )
        output = root / "pools.jsonl"
        build_candidate_pools(
            config, points, checkpoint, output, training=True, device="cpu", batch_size=1
        )
        row = json.loads(output.read_text())
        assert target in row["candidate_ids"]
        assert abs(row["target_retrieval_score"] - expected) < 1e-6
        assert abs(row["retrieval_scores"][row["target_index"]] - expected) < 1e-6
