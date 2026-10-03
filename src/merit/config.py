from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    evidence_vocab_size: int = 1024
    candidate_pool_size: int = 100
    max_sequence_length: int = 50
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "review": 1.0 / 3,
            "attribute": 1.0 / 3,
            "kg": 1.0 / 3,
        }
    )


@dataclass
class RetrieverConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2


@dataclass
class MatchingConfig:
    neighbors: int = 4
    contrast_sigma: float = 1.0


@dataclass
class MCEBConfig:
    semantic_dim: int = 384
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    evidence_budget: int = 5
    gate_temperature: float = 0.67
    hard_concrete_gamma: float = -0.1
    hard_concrete_zeta: float = 1.1
    pseudo_evidence_ratio: float = 0.25
    stable_weight: float = 0.01
    fake_weight: float = 0.01
    rerank_temperature: float = 1.0


@dataclass
class RerankerTrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5
    gradient_clip: float = 1.0


@dataclass
class GenerationConfig:
    domain: str = "book"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    teacher_model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    semantic_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 80
    max_input_tokens: int = 1024
    competitor_count: int = 4
    summary_evidence_count: int = 10
    role_threshold: float = 0.01


@dataclass
class EvaluationConfig:
    cutoffs: list[int] = field(default_factory=lambda: [5, 10, 20])
    neighborhood_resamples: int = 10


@dataclass
class MeritConfig:
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [13, 21, 42, 87, 100])
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    mceb: MCEBConfig = field(default_factory=MCEBConfig)
    reranker_training: RerankerTrainingConfig = field(default_factory=RerankerTrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def from_dict(values: dict[str, Any]) -> MeritConfig:
    return MeritConfig(
        seed=values.get("seed", 42),
        seeds=values.get("seeds", [13, 21, 42, 87, 100]),
        data=_merge_dataclass(DataConfig, values.get("data")),
        retriever=_merge_dataclass(RetrieverConfig, values.get("retriever")),
        matching=_merge_dataclass(MatchingConfig, values.get("matching")),
        mceb=_merge_dataclass(MCEBConfig, values.get("mceb")),
        reranker_training=_merge_dataclass(
            RerankerTrainingConfig, values.get("reranker_training")
        ),
        generation=_merge_dataclass(GenerationConfig, values.get("generation")),
        evaluation=_merge_dataclass(EvaluationConfig, values.get("evaluation")),
    )


def load_config(path: str | Path) -> MeritConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return from_dict(raw)
