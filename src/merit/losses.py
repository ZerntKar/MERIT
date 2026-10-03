from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import RerankerOutput


def pool_listwise_loss(
    scores: torch.Tensor, target_index: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    valid = target_index >= 0
    if not valid.any():
        return scores[torch.isfinite(scores)].sum() * 0.0
    return F.cross_entropy(scores[valid] / temperature, target_index[valid])


def stability_loss(output: RerankerOutput) -> torch.Tensor:
    if output.stable_probabilities is None:
        return output.activation_probabilities.sum() * 0.0
    difference = (output.activation_probabilities - output.stable_probabilities).abs()
    denominator = output.evidence_mask.sum(dim=-1).clamp_min(1)
    return ((difference * output.evidence_mask).sum(dim=-1) / denominator).mean()


def pseudo_evidence_loss(output: RerankerOutput) -> torch.Tensor:
    denominator = output.fake_mask.sum(dim=-1)
    per_example = (output.activation_probabilities * output.fake_mask).sum(dim=-1)
    valid = denominator > 0
    if not valid.any():
        return output.activation_probabilities.sum() * 0.0
    return (per_example[valid] / denominator[valid]).mean()


def merit_loss(
    output: RerankerOutput,
    target_index: torch.Tensor,
    rerank_temperature: float,
    stable_weight: float,
    fake_weight: float,
) -> dict[str, torch.Tensor]:
    pool = pool_listwise_loss(output.scores, target_index, rerank_temperature)
    stable = stability_loss(output)
    fake = pseudo_evidence_loss(output)
    total = pool + stable_weight * stable + fake_weight * fake
    return {"total": total, "pool": pool, "stable": stable, "fake": fake}

