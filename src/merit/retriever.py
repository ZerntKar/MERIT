from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MeritConfig


class SASRec(nn.Module):
    def __init__(self, num_items: int, config: MeritConfig):
        super().__init__()
        dim = config.retriever.hidden_dim
        self.num_items = num_items
        self.max_length = config.data.max_sequence_length
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_length, dim)
        self.input_dropout = nn.Dropout(config.retriever.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.retriever.num_heads,
            dim_feedforward=4 * dim,
            dropout=config.retriever.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.retriever.num_layers)
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, item_sequence: torch.Tensor) -> torch.Tensor:
        if item_sequence.shape[1] > self.max_length:
            item_sequence = item_sequence[:, -self.max_length :]
        length = item_sequence.shape[1]
        positions = torch.arange(length, device=item_sequence.device).unsqueeze(0)
        hidden = self.item_embedding(item_sequence) + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(length, length, device=item_sequence.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=item_sequence.eq(0),
        )
        return self.output_norm(hidden)

    def forward(self, item_sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(item_sequence)
        return hidden @ self.item_embedding.weight.T

    def next_item_loss(self, item_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        logits = self(item_sequence)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_sequence.reshape(-1),
            ignore_index=0,
        )

    @torch.no_grad()
    def retrieve(
        self,
        item_sequence: torch.Tensor,
        candidate_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        truncated = item_sequence[:, -self.max_length :]
        encoded = self.encode(truncated)
        lengths = truncated.ne(0).sum(dim=-1).clamp_min(1) - 1
        batch = torch.arange(item_sequence.shape[0], device=item_sequence.device)
        user = encoded[batch, lengths]
        scores = user @ self.item_embedding.weight.T
        scores[:, 0] = -torch.inf
        scores.scatter_(1, truncated, -torch.inf)
        values, indices = scores.topk(candidate_count, dim=-1)
        return indices, values

    @torch.no_grad()
    def score_targets(
        self, item_sequence: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        truncated = item_sequence[:, -self.max_length :]
        encoded = self.encode(truncated)
        lengths = truncated.ne(0).sum(dim=-1).clamp_min(1) - 1
        batch = torch.arange(truncated.shape[0], device=truncated.device)
        user = encoded[batch, lengths]
        target = self.item_embedding(target_ids)
        return (user * target).sum(dim=-1)


def force_training_target_into_pool(
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    target_ids: torch.Tensor,
    target_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = candidate_ids.clone()
    scores = candidate_scores.clone()
    matches = ids.eq(target_ids.unsqueeze(-1))
    present = matches.any(dim=-1)
    missing = ~present
    ids[missing, -1] = target_ids[missing]
    scores[missing, -1] = target_scores[missing]
    matches = ids.eq(target_ids.unsqueeze(-1))
    target_index = matches.float().argmax(dim=-1)
    return ids, scores, target_index
