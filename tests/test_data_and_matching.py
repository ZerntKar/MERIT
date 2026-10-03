import torch
import json
import tempfile
from pathlib import Path

from merit.data import EvidenceAggregator, build_preference_strength, build_shared_evidence
from merit.matching import build_local_contrast
from merit.metrics import feature_diversity, feature_matching_ratio
from merit.preprocess import prepare_chronological_splits


def test_evidence_construction_and_matching():
    raw = [
        {"review": {"quiet": 3}, "attribute": {"nearby": 1}, "kg": {}},
        {"review": {"quiet": 1}, "attribute": {"nearby": 2}, "kg": {}},
    ]
    aggregator = EvidenceAggregator({"review": 0.5, "attribute": 0.5, "kg": 0.0}).fit(raw)
    items = [aggregator.transform(item) for item in raw]
    preference = build_preference_strength([{"quiet": 2, "nearby": 1}])
    labels, matrix, history = build_shared_evidence(preference, items)
    assert labels == ["nearby", "quiet"]
    assert len(matrix) == 2 and len(history) == 2

    support = torch.tensor(matrix).unsqueeze(0)
    features = torch.tensor([[[0.9, 4.0, 2.0], [0.8, 4.1, 2.1]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    contrast, _, _ = build_local_contrast(support, features, mask, 1, 1.0)
    assert torch.allclose(contrast[:, 0], support[:, 0] - support[:, 1])
    assert feature_matching_ratio([{"quiet"}, {"nearby"}], ["quiet", "quiet"]) == 0.5
    assert feature_diversity([{"quiet", "nearby"}, {"quiet"}]) == 1.0


def test_chronological_preparation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "events.jsonl"
        rows = [
            {"user_id": "u", "item_id": f"i{index}", "timestamp": index}
            for index in range(1, 5)
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = prepare_chronological_splits(source, root / "out")
        assert manifest["num_items"] == 4
        validation = json.loads((root / "out/validation_points.jsonl").read_text().strip())
        test = json.loads((root / "out/test_points.jsonl").read_text().strip())
        assert len(validation["history_items"]) == 2
        assert len(test["history_items"]) == 3
