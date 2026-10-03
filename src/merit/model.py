from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import MeritConfig
from .hard_concrete import HardConcreteTopK
from .matching import build_local_contrast


@dataclass
class RerankerOutput:
    scores: torch.Tensor
    contributions: torch.Tensor
    gate: torch.Tensor
    activation_probabilities: torch.Tensor
    selected_indices: torch.Tensor
    contextualized_evidence: torch.Tensor
    evidence_mask: torch.Tensor
    fake_mask: torch.Tensor
    local_contrast: torch.Tensor
    stable_probabilities: torch.Tensor | None = None


class EvidenceBottleneck(nn.Module):
    def __init__(self, config: MeritConfig):
        super().__init__()
        self.config = config
        token_input = (
            2 * config.data.candidate_pool_size + 1 + config.mceb.semantic_dim
        )
        self.token_projection = nn.Linear(token_input, config.mceb.hidden_dim)
        self.token_norm = nn.LayerNorm(config.mceb.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.mceb.hidden_dim,
            nhead=config.mceb.num_heads,
            dim_feedforward=config.mceb.ffn_dim,
            dropout=config.mceb.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.mceb.num_layers)
        self.gate_predictor = nn.Linear(config.mceb.hidden_dim, 1)
        self.gate = HardConcreteTopK(
            budget=config.mceb.evidence_budget,
            temperature=config.mceb.gate_temperature,
            gamma=config.mceb.hard_concrete_gamma,
            zeta=config.mceb.hard_concrete_zeta,
        )

    def encode(
        self,
        support: torch.Tensor,
        contrast: torch.Tensor,
        preference: torch.Tensor,
        semantic: torch.Tensor,
        candidate_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_candidates = self.config.data.candidate_pool_size
        if support.shape[1] != expected_candidates:
            raise ValueError(
                f"MCEB expects Kc={expected_candidates}; got {support.shape[1]}. "
                "Pass candidate_pool_size to collate_merit."
            )
        cmask = candidate_mask.unsqueeze(-1)
        support_columns = support.masked_fill(~cmask, 0.0).transpose(1, 2)
        contrast_columns = contrast.masked_fill(~cmask, 0.0).transpose(1, 2)
        token_input = torch.cat(
            [support_columns, contrast_columns, preference.unsqueeze(-1), semantic], dim=-1
        )
        tokens = self.token_norm(self.token_projection(token_input))
        tokens = tokens.masked_fill(~evidence_mask.unsqueeze(-1), 0.0)
        contextualized = self.encoder(tokens, src_key_padding_mask=~evidence_mask)
        logits = self.gate_predictor(contextualized).squeeze(-1)
        logits = logits.masked_fill(~evidence_mask, -30.0)
        return contextualized, logits

    def forward(
        self,
        support: torch.Tensor,
        contrast: torch.Tensor,
        preference: torch.Tensor,
        semantic: torch.Tensor,
        candidate_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        contextualized, logits = self.encode(
            support,
            contrast,
            preference,
            semantic,
            candidate_mask,
            evidence_mask,
        )
        gate, probabilities, selected = self.gate(logits, evidence_mask)
        return contextualized, gate, probabilities, selected


def inject_pseudo_evidence(
    support: torch.Tensor,
    preference: torch.Tensor,
    semantic: torch.Tensor,
    evidence_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, candidates, evidence = support.shape
    fake_count = max(1, round(evidence * ratio)) if ratio > 0 else 0
    if fake_count == 0:
        return (
            support,
            preference,
            semantic,
            evidence_mask,
            torch.zeros_like(evidence_mask),
        )
    fake_support = support.new_zeros(batch, candidates, fake_count)
    fake_preference = preference.new_zeros(batch, fake_count)
    fake_semantic = semantic.new_zeros(batch, fake_count, semantic.shape[-1])
    fake_valid = torch.zeros(batch, fake_count, dtype=torch.bool, device=support.device)

    for row in range(batch):
        valid = evidence_mask[row].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() < 3:
            continue
        row_fake_count = min(fake_count, max(1, round(valid.numel() * ratio)))
        label_order = valid[torch.randperm(valid.numel(), device=valid.device, generator=generator)]
        for fake_index in range(row_fake_count):
            label_index = label_order[fake_index % valid.numel()]
            # Combine semantics, support, and preference from different real evidence dimensions.
            support_candidates = valid[valid != label_index]
            support_index = support_candidates[
                torch.randint(
                    support_candidates.numel(),
                    (1,),
                    device=valid.device,
                    generator=generator,
                )
            ].squeeze(0)
            preference_candidates = valid[(valid != label_index) & (valid != support_index)]
            preference_index = preference_candidates[
                torch.randint(
                    preference_candidates.numel(),
                    (1,),
                    device=valid.device,
                    generator=generator,
                )
            ].squeeze(0)
            fake_support[row, :, fake_index] = support[row, :, support_index]
            fake_preference[row, fake_index] = preference[row, preference_index]
            fake_semantic[row, fake_index] = semantic[row, label_index]
            fake_valid[row, fake_index] = True

    augmented_mask = torch.cat([evidence_mask, fake_valid], dim=-1)
    fake_mask = torch.cat([torch.zeros_like(evidence_mask), fake_valid], dim=-1)
    return (
        torch.cat([support, fake_support], dim=-1),
        torch.cat([preference, fake_preference], dim=-1),
        torch.cat([semantic, fake_semantic], dim=1),
        augmented_mask,
        fake_mask,
    )


class MERITReranker(nn.Module):
    def __init__(self, config: MeritConfig):
        super().__init__()
        self.config = config
        self.mceb = EvidenceBottleneck(config)

    def forward(
        self,
        candidate_support: torch.Tensor,
        preference_strength: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        matching_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        augment_pseudo: bool | None = None,
        compute_stability: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> RerankerOutput:
        augment_pseudo = self.training if augment_pseudo is None else augment_pseudo
        compute_stability = self.training if compute_stability is None else compute_stability
        support = candidate_support
        preference = preference_strength
        semantic = semantic_embeddings
        current_evidence_mask = evidence_mask
        fake_mask = torch.zeros_like(evidence_mask)
        if augment_pseudo and self.config.mceb.pseudo_evidence_ratio > 0:
            support, preference, semantic, current_evidence_mask, fake_mask = inject_pseudo_evidence(
                support,
                preference,
                semantic,
                evidence_mask,
                self.config.mceb.pseudo_evidence_ratio,
                generator=generator,
            )

        contrast, _, _ = build_local_contrast(
            support,
            matching_features,
            candidate_mask,
            self.config.matching.neighbors,
            self.config.matching.contrast_sigma,
        )
        contextualized, gate, probabilities, selected = self.mceb(
            support,
            contrast,
            preference,
            semantic,
            candidate_mask,
            current_evidence_mask,
        )
        evidence_weights = preference * gate
        # Candidate scores are additive contributions from the fixed-budget evidence set.
        contributions = support * evidence_weights.unsqueeze(1)
        scores = contributions.sum(dim=-1).masked_fill(~candidate_mask, -torch.inf)

        stable_probabilities = None
        if compute_stability:
            # Selection probabilities should remain stable under matched-neighborhood resampling.
            resampled_contrast, _, _ = build_local_contrast(
                support,
                matching_features,
                candidate_mask,
                self.config.matching.neighbors,
                self.config.matching.contrast_sigma,
                resample=True,
                generator=generator,
            )
            _, stable_logits = self.mceb.encode(
                support,
                resampled_contrast,
                preference,
                semantic,
                candidate_mask,
                current_evidence_mask,
            )
            stable_probabilities = self.mceb.gate.activation_probability(stable_logits)
            stable_probabilities = stable_probabilities * current_evidence_mask

        return RerankerOutput(
            scores=scores,
            contributions=contributions,
            gate=gate,
            activation_probabilities=probabilities,
            selected_indices=selected,
            contextualized_evidence=contextualized,
            evidence_mask=current_evidence_mask,
            fake_mask=fake_mask,
            local_contrast=contrast,
            stable_probabilities=stable_probabilities,
        )


def contribution_gaps(
    output: RerankerOutput, competitor_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recommendation = output.scores.argmax(dim=-1)
    masked = output.scores.clone()
    masked.scatter_(1, recommendation.unsqueeze(-1), -torch.inf)
    q = min(competitor_count, max(1, masked.shape[1] - 1))
    competitors = masked.topk(q, dim=-1).indices
    batch_index = torch.arange(output.scores.shape[0], device=output.scores.device)
    selected_contribution = output.contributions[batch_index, recommendation]
    expanded_batch = batch_index[:, None].expand_as(competitors)
    competitor_contribution = output.contributions[expanded_batch, competitors].mean(dim=1)
    # The signed gap assigns each selected feature its explanation role.
    gaps = (selected_contribution - competitor_contribution) * output.gate.detach()
    return recommendation, competitors, gaps


def gather_selected_context(output: RerankerOutput) -> torch.Tensor:
    index = output.selected_indices.unsqueeze(-1).expand(
        -1, -1, output.contextualized_evidence.shape[-1]
    )
    return output.contextualized_evidence.gather(1, index)
