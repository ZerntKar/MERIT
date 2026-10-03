from __future__ import annotations

import torch


def standardize_matching_features(
    features: torch.Tensor, candidate_mask: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    # Standardize within each candidate pool so distances are comparable across covariates.
    mask = candidate_mask.unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (features * mask).sum(dim=1, keepdim=True) / count
    variance = ((features - mean).square() * mask).sum(dim=1, keepdim=True) / count
    standardized = (features - mean) / (variance.sqrt() + epsilon)
    return standardized.masked_fill(~mask, 0.0)


def matched_neighbors(
    standardized_features: torch.Tensor,
    candidate_mask: torch.Tensor,
    neighbors: int,
    resample: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, candidates, _ = standardized_features.shape
    distance = torch.cdist(standardized_features, standardized_features, p=2)
    valid_pairs = candidate_mask[:, :, None] & candidate_mask[:, None, :]
    eye = torch.eye(candidates, device=distance.device, dtype=torch.bool).unsqueeze(0)
    distance = distance.masked_fill(~valid_pairs | eye, torch.inf)
    search_k = min(candidates - 1, neighbors * (2 if resample else 1))
    rank_offset = torch.arange(candidates, device=distance.device, dtype=distance.dtype)
    selection_distance = distance + rank_offset.view(1, 1, -1) * 1e-7
    _, nearest_index = torch.topk(
        selection_distance, k=max(search_k, 1), dim=-1, largest=False, sorted=True
    )
    nearest_distance = distance.gather(-1, nearest_index)
    if resample and search_k > neighbors:
        # Draw an alternative neighborhood from the wider nearest-neighbor set for stability.
        random_scores = torch.rand(
            batch, candidates, search_k, device=distance.device, generator=generator
        )
        random_scores = random_scores.masked_fill(~torch.isfinite(nearest_distance), torch.inf)
        selected = random_scores.argsort(dim=-1)[..., :neighbors]
        nearest_index = nearest_index.gather(-1, selected)
        nearest_distance = nearest_distance.gather(-1, selected)
        order = nearest_distance.argsort(dim=-1)
        nearest_index = nearest_index.gather(-1, order)
        nearest_distance = nearest_distance.gather(-1, order)
    else:
        nearest_index = nearest_index[..., :neighbors]
        nearest_distance = nearest_distance[..., :neighbors]
    return nearest_index, nearest_distance


def local_contrast(
    support: torch.Tensor,
    neighbor_index: torch.Tensor,
    neighbor_distance: torch.Tensor,
    sigma: float,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    batch, candidates, evidence = support.shape
    m = neighbor_index.shape[-1]
    expanded = support[:, None, :, :].expand(batch, candidates, candidates, evidence)
    gather_index = neighbor_index.unsqueeze(-1).expand(batch, candidates, m, evidence)
    neighbor_support = expanded.gather(2, gather_index)
    valid = torch.isfinite(neighbor_distance)
    logits = (-neighbor_distance / sigma).masked_fill(~valid, -torch.inf)
    no_valid = ~valid.any(dim=-1, keepdim=True)
    logits = torch.where(no_valid, torch.zeros_like(logits), logits)
    weights = torch.softmax(logits, dim=-1) * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    comparison = (neighbor_support * weights.unsqueeze(-1)).sum(dim=2)
    # Contrast isolates evidence that distinguishes a candidate from matched alternatives.
    contrast = support - comparison
    return contrast.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)


def build_local_contrast(
    support: torch.Tensor,
    matching_features: torch.Tensor,
    candidate_mask: torch.Tensor,
    neighbors: int,
    sigma: float,
    resample: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    standardized = standardize_matching_features(matching_features, candidate_mask)
    index, distance = matched_neighbors(
        standardized, candidate_mask, neighbors, resample=resample, generator=generator
    )
    contrast = local_contrast(support, index, distance, sigma, candidate_mask)
    return contrast, index, distance
