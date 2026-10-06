from __future__ import annotations

import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from .config import MeritConfig
from .data import build_preference_strength, build_shared_evidence, iter_jsonl
from .retriever import SASRec, force_training_target_into_pool
from .semantic import encode_evidence_labels


def _load_json(path: str | Path | None, default):
    if path is None:
        return default
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evidence_mapping(value) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): float(score) for key, score in value.items()}
    return {str(key): 1.0 for key in value}


def _canonical_mapping(value, aliases: Mapping[str, str]) -> dict[str, float]:
    output: dict[str, float] = defaultdict(float)
    for label, score in _evidence_mapping(value).items():
        normalized = " ".join(label.lower().strip().split())
        canonical = aliases.get(normalized, normalized)
        if score > 0:
            output[canonical] += float(score)
    return dict(output)


def build_evidence_vocabulary(
    interactions_path: str | Path,
    items_path: str | Path,
    output_path: str | Path,
    max_size: int,
    aliases_path: str | Path | None = None,
) -> dict:
    raw_aliases = _load_json(aliases_path, {})
    aliases = {
        " ".join(str(key).lower().strip().split()): " ".join(
            str(value).lower().strip().split()
        )
        for key, value in raw_aliases.items()
    }
    frequency: Counter[str] = Counter()
    for event in iter_jsonl(interactions_path):
        if event.get("split") != "train":
            continue
        evidence = event.get("positive_evidence", event.get("evidence", {}))
        frequency.update(_canonical_mapping(evidence, aliases))
    for item in iter_jsonl(items_path):
        for field in ("attribute_evidence", "attributes", "kg_evidence", "kg"):
            frequency.update(_canonical_mapping(item.get(field, {}), aliases))
    labels = [
        label
        for label, _ in sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0]))[:max_size]
    ]
    payload = {"labels": labels, "aliases": aliases, "frequency": [frequency[x] for x in labels]}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"size": len(labels), "output": str(output)}


def encode_vocabulary(
    vocabulary_path: str | Path,
    output_path: str | Path,
    model_name: str,
    device: str | None = None,
) -> dict:
    vocabulary = _load_json(vocabulary_path, {})
    labels = vocabulary["labels"]
    embeddings = encode_evidence_labels(labels, model_name, device=device).cpu().numpy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    return {"labels": len(labels), "dimension": int(embeddings.shape[1]), "output": str(output)}


def build_candidate_pools(
    config: MeritConfig,
    points_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    training: bool,
    device: str | None = None,
    batch_size: int = 256,
) -> dict:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    num_items = int(checkpoint["num_items"])
    if config.data.candidate_pool_size > num_items:
        raise ValueError("candidate pool size cannot exceed the item catalog")
    model = SASRec(num_items, config).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    points = list(iter_jsonl(points_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    retrieved_targets = 0
    forced_targets = 0
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(points), batch_size):
            rows = points[start : start + batch_size]
            sequence = torch.zeros(
                len(rows), config.data.max_sequence_length, dtype=torch.long, device=target_device
            )
            for row_index, row in enumerate(rows):
                history = [int(x) for x in row["history_items"]][
                    -config.data.max_sequence_length :
                ]
                sequence[row_index, : len(history)] = torch.as_tensor(
                    history, dtype=torch.long, device=target_device
                )
            targets = torch.as_tensor(
                [int(row["target_item"]) for row in rows],
                dtype=torch.long,
                device=target_device,
            )
            candidate_ids, candidate_scores = model.retrieve(
                sequence, config.data.candidate_pool_size
            )
            target_scores = model.score_targets(sequence, targets)
            initial_matches = candidate_ids.eq(targets.unsqueeze(-1))
            retrieved_targets += int(initial_matches.any(dim=-1).sum().item())
            if training:
                # Training pools must contain the positive item; evaluation pools remain untouched.
                forced_targets += int((~initial_matches.any(dim=-1)).sum().item())
                candidate_ids, candidate_scores, target_index = force_training_target_into_pool(
                    candidate_ids, candidate_scores, targets, target_scores
                )
            else:
                matches = candidate_ids.eq(targets.unsqueeze(-1))
                present = matches.any(dim=-1)
                target_index = torch.where(
                    present,
                    matches.float().argmax(dim=-1),
                    torch.full_like(targets, -1),
                )
            for row_index, row in enumerate(rows):
                payload = {
                    **row,
                    "candidate_ids": candidate_ids[row_index].cpu().tolist(),
                    "retrieval_scores": candidate_scores[row_index].cpu().tolist(),
                    "target_index": int(target_index[row_index].item()),
                    "target_retrieval_score": float(target_scores[row_index].item()),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "examples": len(points),
        "retrieved_targets": retrieved_targets,
        "forced_targets": forced_targets,
        "output": str(output),
    }


def _mapped_items(
    items_path: str | Path, item_map_path: str | Path | None, aliases: Mapping[str, str]
) -> dict[int, dict]:
    item_map = _load_json(item_map_path, {})
    output = {}
    for row in iter_jsonl(items_path):
        raw_id = str(row["item_id"])
        item_id = int(item_map[raw_id]) if raw_id in item_map else int(row["item_id"])
        output[item_id] = {
            **row,
            "attribute_evidence": _canonical_mapping(
                row.get("attribute_evidence", row.get("attributes", {})), aliases
            ),
            "kg_evidence": _canonical_mapping(row.get("kg_evidence", row.get("kg", {})), aliases),
        }
    return output


def _metadata_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def materialize_merit_examples(
    config: MeritConfig,
    pools_path: str | Path,
    interactions_path: str | Path,
    items_path: str | Path,
    vocabulary_path: str | Path,
    embeddings_path: str | Path,
    output_path: str | Path,
    item_map_path: str | Path | None = None,
) -> dict:
    vocabulary = _load_json(vocabulary_path, {})
    labels = vocabulary["labels"]
    aliases = vocabulary.get("aliases", {})
    order = {label: index for index, label in enumerate(labels)}
    allowed = set(labels)
    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(labels):
        raise ValueError("vocabulary and semantic embedding row counts differ")
    if embeddings.shape[1] != config.mceb.semantic_dim:
        raise ValueError("semantic embedding dimension does not match the MCEB config")
    items = _mapped_items(items_path, item_map_path, aliases)
    static_max = {"attribute": defaultdict(float), "kg": defaultdict(float)}
    for item in items.values():
        for source, field in (("attribute", "attribute_evidence"), ("kg", "kg_evidence")):
            for label, value in item[field].items():
                static_max[source][label] = max(
                    static_max[source][label], math.log1p(max(float(value), 0.0))
                )
    events = list(iter_jsonl(interactions_path))
    for event in events:
        event["user_id"] = str(event["user_id"])
        event["item_id"] = int(event["item_id"])
        event["timestamp"] = int(event["timestamp"])
        event["positive_evidence"] = _canonical_mapping(
            event.get("positive_evidence", event.get("evidence", {})), aliases
        )
    events.sort(key=lambda row: (row["timestamp"], row["user_id"], row["item_id"]))
    points = list(iter_jsonl(pools_path))
    points.sort(key=lambda row: (int(row["timestamp"]), str(row["user_id"])))
    item_review: dict[int, Counter] = defaultdict(Counter)
    user_preference: dict[str, Counter] = defaultdict(Counter)
    rating_sum: Counter = Counter()
    rating_count: Counter = Counter()
    popularity: Counter = Counter()
    review_max: dict[str, float] = defaultdict(float)
    event_index = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    insufficient = 0
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch, (
        Path(scratch) / output.name
    ).open("w", encoding="utf-8") as handle:
        for point in points:
            timestamp = int(point["timestamp"])
            # Strict inequality prevents the target interaction from entering its own evidence.
            while event_index < len(events) and events[event_index]["timestamp"] < timestamp:
                event = events[event_index]
                item_id = event["item_id"]
                user_id = event["user_id"]
                popularity[item_id] += 1
                if event.get("rating") is not None:
                    rating_sum[item_id] += float(event.get("rating", 0.0))
                    rating_count[item_id] += 1
                for label, value in event["positive_evidence"].items():
                    if label not in allowed:
                        continue
                    item_review[item_id][label] += float(value)
                    user_preference[user_id][label] += float(value)
                    review_max[label] = max(
                        review_max[label], math.log1p(item_review[item_id][label])
                    )
                event_index += 1
            candidate_ids = [int(x) for x in point["candidate_ids"]]
            candidate_supports = []
            average_ratings = []
            popularities = []
            names = []
            categories = []
            metadata = []
            for item_id in candidate_ids:
                item = items.get(item_id, {})
                sources = {
                    "review": item_review[item_id],
                    "attribute": item.get("attribute_evidence", {}),
                    "kg": item.get("kg_evidence", {}),
                }
                support: dict[str, float] = defaultdict(float)
                for source, source_values in sources.items():
                    alpha = config.data.source_weights[source]
                    maxima = review_max if source == "review" else static_max[source]
                    for label, value in source_values.items():
                        if label not in allowed or value <= 0:
                            continue
                        denominator = maxima.get(label, 0.0) + 1e-12
                        support[label] += alpha * math.log1p(float(value)) / denominator
                candidate_supports.append(dict(support))
                count = rating_count[item_id]
                average_ratings.append(float(rating_sum[item_id] / count) if count else 0.0)
                popularities.append(int(popularity[item_id]))
                names.append(str(item.get("name", item.get("title", item_id))))
                categories.append(str(item.get("category", "")))
                metadata.append(_metadata_text(item.get("metadata", item.get("description", ""))))
            preference = build_preference_strength([user_preference[str(point["user_id"])]])
            full_preference = sorted(
                ((label, value) for label, value in preference.items() if label in allowed),
                key=lambda pair: (-pair[1], order[pair[0]]),
            )
            shared_labels, matrix, history = build_shared_evidence(
                preference, candidate_supports, order
            )
            if len(shared_labels) < config.mceb.evidence_budget:
                insufficient += 1
                continue
            semantic = [embeddings[order[label]].astype(float).tolist() for label in shared_labels]
            payload = {
                **point,
                "evidence_labels": shared_labels,
                "candidate_support": matrix,
                "preference_strength": history,
                "semantic_embeddings": semantic,
                "average_ratings": average_ratings,
                "popularities": popularities,
                "candidate_names": names,
                "candidate_categories": categories,
                "candidate_metadata": metadata,
                "user_preference_labels": [label for label, _ in full_preference],
                "user_preference_values": [float(value) for _, value in full_preference],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
        if insufficient:
            raise ValueError(
                f"{insufficient} of {len(points)} examples have fewer than "
                f"{config.mceb.evidence_budget} shared evidence dimensions; "
                "apply a common eligibility filter to every method before evaluation"
            )
        handle.close()
        (Path(scratch) / output.name).replace(output)
    return {
        "input_examples": len(points),
        "written_examples": written,
        "insufficient_evidence": insufficient,
        "output": str(output),
    }
