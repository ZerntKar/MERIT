from __future__ import annotations

import torch


def _topk_mask(score: torch.Tensor, evidence_mask: torch.Tensor, budget: int) -> torch.Tensor:
    score = score.masked_fill(~evidence_mask, -torch.inf)
    index = score.topk(min(budget, score.shape[-1]), dim=-1).indices
    return torch.zeros_like(score).scatter(-1, index, 1.0) * evidence_mask


def intersection_only(
    support: torch.Tensor,
    preference: torch.Tensor,
    candidate_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    average_support = (support * candidate_mask.unsqueeze(-1)).sum(dim=1) / count
    gate = _topk_mask(preference * average_support, evidence_mask, budget)
    contributions = support * (preference * gate).unsqueeze(1)
    scores = contributions.sum(dim=-1).masked_fill(~candidate_mask, -torch.inf)
    return scores, gate, contributions


def preference_gap(
    support: torch.Tensor,
    contrast: torch.Tensor,
    preference: torch.Tensor,
    candidate_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    average_gap = (contrast.clamp_min(0) * candidate_mask.unsqueeze(-1)).sum(dim=1) / count
    gate = _topk_mask(preference * average_gap, evidence_mask, budget)
    contributions = support * (preference * gate).unsqueeze(1)
    scores = contributions.sum(dim=-1).masked_fill(~candidate_mask, -torch.inf)
    return scores, gate, contributions

