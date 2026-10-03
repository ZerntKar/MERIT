from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def encode_evidence_labels(
    labels: Sequence[str], model_name: str, device: str | None = None
) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("install MERIT with the [semantic] extra") from exc
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.encode(
        list(labels), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    ).float()
