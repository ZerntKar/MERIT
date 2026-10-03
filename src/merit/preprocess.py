from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .data import iter_jsonl


def prepare_chronological_splits(input_path: str | Path, output_dir: str | Path) -> dict:
    rows = list(iter_jsonl(input_path))
    required = {"user_id", "item_id", "timestamp"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"interaction row is missing fields: {sorted(missing)}")
    item_values = sorted({str(row["item_id"]) for row in rows})
    item_map = {item_id: index + 1 for index, item_id in enumerate(item_values)}
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        mapped = dict(row)
        mapped["item_id"] = item_map[str(row["item_id"])]
        by_user[str(row["user_id"])].append(mapped)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train_sequences.jsonl"
    train_points_path = output / "train_points.jsonl"
    validation_path = output / "validation_points.jsonl"
    test_path = output / "test_points.jsonl"
    mapped_path = output / "mapped_interactions.jsonl"
    retained_users = 0
    with train_path.open("w", encoding="utf-8") as train_file, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_file, test_path.open("w", encoding="utf-8") as test_file, train_points_path.open(
        "w", encoding="utf-8"
    ) as train_points_file, mapped_path.open("w", encoding="utf-8") as mapped_file:
        for user_id in sorted(by_user):
            events = sorted(
                by_user[user_id], key=lambda row: (int(row["timestamp"]), int(row["item_id"]))
            )
            if len(events) < 3:
                for event in events:
                    mapped_file.write(json.dumps({**event, "split": "unused"}) + "\n")
                continue
            retained_users += 1
            training = events[:-2]
            for position, event in enumerate(events):
                split = "train" if position < len(events) - 2 else (
                    "validation" if position == len(events) - 2 else "test"
                )
                mapped_file.write(json.dumps({**event, "split": split}) + "\n")
            train_file.write(
                json.dumps(
                    {"user_id": user_id, "items": [row["item_id"] for row in training]}
                )
                + "\n"
            )
            for position in range(1, len(events) - 2):
                point = {
                    "user_id": user_id,
                    "history_items": [row["item_id"] for row in events[:position]],
                    "target_item": events[position]["item_id"],
                    "timestamp": events[position]["timestamp"],
                    "split": "train",
                }
                train_points_file.write(json.dumps(point) + "\n")
            validation = {
                "user_id": user_id,
                "history_items": [row["item_id"] for row in training],
                "target_item": events[-2]["item_id"],
                "timestamp": events[-2]["timestamp"],
                "split": "validation",
            }
            test = {
                "user_id": user_id,
                "history_items": [row["item_id"] for row in events[:-1]],
                "target_item": events[-1]["item_id"],
                "timestamp": events[-1]["timestamp"],
                "split": "test",
            }
            validation_file.write(json.dumps(validation) + "\n")
            test_file.write(json.dumps(test) + "\n")
    with (output / "item_map.json").open("w", encoding="utf-8") as handle:
        json.dump(item_map, handle, ensure_ascii=False, indent=2)
    manifest = {
        "num_items": len(item_map),
        "num_retained_users": retained_users,
        "input_interactions": len(rows),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
