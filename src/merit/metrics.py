from __future__ import annotations

from typing import Iterable, Sequence

import torch


def ranking_metrics(
    scores: torch.Tensor, target_index: torch.Tensor, cutoffs: Iterable[int]
) -> dict[str, float]:
    order = scores.argsort(dim=-1, descending=True)
    result: dict[str, float] = {}
    valid_target = target_index >= 0
    matches = order.eq(target_index.clamp_min(0).unsqueeze(-1)) & valid_target.unsqueeze(-1)
    rank = matches.float().argmax(dim=-1) + 1
    found = matches.any(dim=-1)
    for cutoff in cutoffs:
        hit = found & (rank <= cutoff)
        result[f"HR@{cutoff}"] = hit.float().mean().item()
        gain = torch.where(hit, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(rank.float()))
        result[f"NDCG@{cutoff}"] = gain.mean().item()
    return result


def top1_reconstruction(
    contributions: torch.Tensor,
    original_top1: torch.Tensor,
    expressed_evidence_mask: torch.Tensor,
) -> float:
    reconstructed = (
        contributions * expressed_evidence_mask.unsqueeze(1)
    ).sum(dim=-1).argmax(dim=-1)
    return reconstructed.eq(original_top1).float().mean().item()


def comprehensiveness(
    contributions: torch.Tensor,
    original_top1: torch.Tensor,
    expressed_evidence_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    original_scores = contributions.sum(dim=-1)
    remaining_scores = (
        contributions * (~expressed_evidence_mask.bool()).unsqueeze(1)
    ).sum(dim=-1)
    batch = torch.arange(contributions.shape[0], device=contributions.device)
    original_masked = original_scores.clone()
    original_masked[batch, original_top1] = -torch.inf
    original_margin = (
        original_scores[batch, original_top1] - original_masked.max(dim=-1).values
    )
    remaining_masked = remaining_scores.clone()
    remaining_masked[batch, original_top1] = -torch.inf
    remaining_margin = (
        remaining_scores[batch, original_top1] - remaining_masked.max(dim=-1).values
    )
    value = ((original_margin - remaining_margin) / (original_margin + epsilon)).clamp(0, 1)
    return value.mean().item()


def jaccard_at_k(original_gate: torch.Tensor, resampled_gates: Sequence[torch.Tensor]) -> float:
    original = original_gate.bool()
    values = []
    for gate in resampled_gates:
        current = gate.bool()
        intersection = (original & current).sum(dim=-1).float()
        union = (original | current).sum(dim=-1).clamp_min(1).float()
        values.append(intersection / union)
    return torch.stack(values).mean().item()


def pseudo_select_rate(gate: torch.Tensor, fake_mask: torch.Tensor, budget: int) -> float:
    return ((gate.bool() & fake_mask.bool()).sum(dim=-1).float() / budget).mean().item()


def feature_hallucination_rates(
    generated: Sequence[set[str]],
    item_supported: Sequence[set[str]],
    user_preferred: Sequence[set[str]],
) -> dict[str, float]:
    factual, preference = [], []
    for output, item, user in zip(generated, item_supported, user_preferred):
        denominator = max(len(output), 1)
        factual.append(len(output - item) / denominator)
        preference.append(len(output - user) / denominator)
    return {
        "F-EHR": sum(factual) / max(len(factual), 1),
        "P-EHR": sum(preference) / max(len(preference), 1),
    }


def feature_matching_ratio(
    generated: Sequence[set[str]], references: Sequence[str | set[str]]
) -> float:
    values = []
    for output, reference in zip(generated, references):
        target = {reference} if isinstance(reference, str) else reference
        values.append(float(bool(output & target)))
    return sum(values) / max(len(values), 1)


def feature_coverage_ratio(generated: Sequence[set[str]], vocabulary: set[str]) -> float:
    covered = set().union(*generated) if generated else set()
    return len(covered & vocabulary) / max(len(vocabulary), 1)


def feature_diversity(generated: Sequence[set[str]]) -> float:
    if len(generated) < 2:
        return 0.0
    overlap = 0.0
    pairs = 0
    for left in range(len(generated)):
        for right in range(left + 1, len(generated)):
            overlap += len(generated[left] & generated[right])
            pairs += 1
    return overlap / pairs
