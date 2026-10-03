from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int
    rating: float = 0.0


def chronological_leave_one_out(
    interactions: Iterable[Interaction],
) -> dict[str, list[Interaction]]:
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for event in interactions:
        by_user[event.user_id].append(event)
    result = {"train": [], "validation": [], "test": []}
    for events in by_user.values():
        events.sort(key=lambda x: (x.timestamp, x.item_id))
        if len(events) < 3:
            continue
        # Reserve the last two interactions to preserve the paper's temporal protocol.
        result["train"].extend(events[:-2])
        result["validation"].append(events[-2])
        result["test"].append(events[-1])
    return result


class EvidenceAggregator:
    def __init__(self, source_weights: Mapping[str, float], epsilon: float = 1e-12):
        total = float(sum(source_weights.values()))
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"evidence source weights must sum to one, got {total}")
        self.source_weights = dict(source_weights)
        self.epsilon = epsilon
        self.max_log_support: dict[str, dict[str, float]] = defaultdict(dict)

    def fit(self, item_source_counts: Iterable[Mapping[str, Mapping[str, float]]]):
        for item in item_source_counts:
            for source, weight in self.source_weights.items():
                for label, raw_count in item.get(source, {}).items():
                    value = math.log1p(max(float(raw_count), 0.0))
                    current = self.max_log_support[source].get(label, 0.0)
                    self.max_log_support[source][label] = max(current, value)
        return self

    def transform(self, item: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        output: dict[str, float] = defaultdict(float)
        for source, alpha in self.source_weights.items():
            for label, raw_count in item.get(source, {}).items():
                # Normalize each label within its source before mixing heterogeneous evidence.
                denominator = self.max_log_support[source].get(label, 0.0) + self.epsilon
                output[label] += alpha * math.log1p(max(float(raw_count), 0.0)) / denominator
        return dict(output)


def build_preference_strength(
    positive_history: Iterable[Mapping[str, float]], epsilon: float = 1e-12
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for event_evidence in positive_history:
        for label, value in event_evidence.items():
            totals[label] += max(float(value), 0.0)
    logged = {label: math.log1p(value) for label, value in totals.items()}
    denominator = max(logged.values(), default=0.0) + epsilon
    return {label: value / denominator for label, value in logged.items()}


def build_shared_evidence(
    user_preferences: Mapping[str, float],
    candidate_supports: Sequence[Mapping[str, float]],
    vocabulary_order: Mapping[str, int] | None = None,
) -> tuple[list[str], list[list[float]], list[float]]:
    pool_labels = {
        label
        for candidate in candidate_supports
        for label, value in candidate.items()
        if float(value) > 0
    }
    # MCEB only receives evidence shared by the user's history and the candidate pool.
    labels = [
        label
        for label, value in user_preferences.items()
        if float(value) > 0 and label in pool_labels
    ]
    if vocabulary_order is None:
        labels.sort()
    else:
        labels.sort(key=lambda label: vocabulary_order.get(label, 10**12))
    matrix = [
        [float(candidate.get(label, 0.0)) for label in labels]
        for candidate in candidate_supports
    ]
    preference = [float(user_preferences[label]) for label in labels]
    return labels, matrix, preference


class MeritJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


class SequenceJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = [row for row in iter_jsonl(path) if len(row.get("items", [])) >= 2]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collate_sequences(examples: Sequence[dict], max_length: int) -> dict[str, torch.Tensor | list]:
    inputs = torch.zeros(len(examples), max_length, dtype=torch.long)
    targets = torch.zeros(len(examples), max_length, dtype=torch.long)
    for row, example in enumerate(examples):
        items = [int(item) for item in example["items"]][-max_length - 1 :]
        source = items[:-1]
        target = items[1:]
        inputs[row, : len(source)] = torch.as_tensor(source)
        targets[row, : len(target)] = torch.as_tensor(target)
    return {
        "item_sequence": inputs,
        "target_sequence": targets,
        "user_id": [example.get("user_id") for example in examples],
    }


def collate_merit(
    examples: Sequence[dict],
    candidate_pool_size: int | None = None,
) -> dict[str, torch.Tensor | list]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_candidates = candidate_pool_size or max(len(x["candidate_ids"]) for x in examples)
    max_evidence = max(len(x["evidence_labels"]) for x in examples)
    if max_evidence == 0 or any(not x["evidence_labels"] for x in examples):
        raise ValueError(
            "every MERIT example needs at least one shared evidence dimension; "
            "filter empty user/pool intersections during preparation"
        )
    first_nonempty = next(x for x in examples if x["semantic_embeddings"])
    semantic_dim = len(first_nonempty["semantic_embeddings"][0])
    batch_size = len(examples)

    support = torch.zeros(batch_size, max_candidates, max_evidence, dtype=torch.float32)
    preference = torch.zeros(batch_size, max_evidence, dtype=torch.float32)
    semantic = torch.zeros(batch_size, max_evidence, semantic_dim, dtype=torch.float32)
    matching = torch.zeros(batch_size, max_candidates, 3, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    evidence_mask = torch.zeros(batch_size, max_evidence, dtype=torch.bool)
    candidate_ids = torch.full((batch_size, max_candidates), -1, dtype=torch.long)
    targets = torch.full((batch_size,), -1, dtype=torch.long)

    for row, example in enumerate(examples):
        kc = min(len(example["candidate_ids"]), max_candidates)
        k = len(example["evidence_labels"])
        if k == 0:
            continue
        candidate_ids[row, :kc] = torch.as_tensor(example["candidate_ids"][:kc], dtype=torch.long)
        candidate_mask[row, :kc] = True
        evidence_mask[row, :k] = True
        support[row, :kc, :k] = torch.as_tensor(
            [candidate[:k] for candidate in example["candidate_support"][:kc]],
            dtype=torch.float32,
        )
        preference[row, :k] = torch.as_tensor(example["preference_strength"][:k])
        current_semantic = torch.as_tensor(example["semantic_embeddings"][:k], dtype=torch.float32)
        if current_semantic.shape[-1] != semantic_dim:
            raise ValueError("all semantic embeddings in a batch must have equal dimension")
        semantic[row, :k] = current_semantic
        # The matching covariates follow the paper: retrieval score, rating, popularity.
        matching[row, :kc, 0] = torch.as_tensor(example["retrieval_scores"][:kc])
        matching[row, :kc, 1] = torch.as_tensor(example["average_ratings"][:kc])
        matching[row, :kc, 2] = torch.log1p(
            torch.as_tensor(example["popularities"][:kc], dtype=torch.float32).clamp_min(0)
        )
        target = int(example.get("target_index", -1))
        if 0 <= target < kc:
            targets[row] = target

    return {
        "candidate_ids": candidate_ids,
        "candidate_support": support,
        "preference_strength": preference,
        "semantic_embeddings": semantic,
        "matching_features": matching,
        "candidate_mask": candidate_mask,
        "evidence_mask": evidence_mask,
        "target_index": targets,
        "evidence_labels": [x["evidence_labels"] for x in examples],
        "raw_examples": list(examples),
    }
