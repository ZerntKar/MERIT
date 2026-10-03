from __future__ import annotations

import math

import torch
from torch import nn


class HardConcreteTopK(nn.Module):
    def __init__(
        self,
        budget: int,
        temperature: float = 0.67,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        if not gamma < 0 < zeta:
            raise ValueError("Hard-Concrete requires gamma < 0 < zeta")
        self.budget = budget
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta

    def activation_probability(self, logits: torch.Tensor) -> torch.Tensor:
        shift = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(logits - shift)

    def forward(
        self, logits: torch.Tensor, evidence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_count = evidence_mask.sum(dim=-1)
        if (valid_count < self.budget).any():
            minimum = int(valid_count.min().item())
            raise ValueError(
                f"fixed evidence budget {self.budget} requires at least {self.budget} "
                f"valid dimensions per example; minimum is {minimum}"
            )
        probabilities = self.activation_probability(logits).masked_fill(~evidence_mask, 0.0)
        if self.training:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            concrete = torch.sigmoid(
                (torch.log(uniform) - torch.log1p(-uniform) + logits) / self.temperature
            )
            relaxed = (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0, 1)
        else:
            relaxed = probabilities
        relaxed = relaxed.masked_fill(~evidence_mask, -torch.inf)
        top_index = relaxed.topk(k=self.budget, dim=-1).indices
        hard = torch.zeros_like(logits).scatter(-1, top_index, 1.0)
        hard = hard * evidence_mask
        surrogate = relaxed.masked_fill(~evidence_mask, 0.0)
        # Use a hard top-k mask in the forward pass and relaxed gradients in backpropagation.
        gate = hard + surrogate - surrogate.detach()
        return gate, probabilities, top_index
