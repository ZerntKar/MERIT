from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .config import MeritConfig


@dataclass(frozen=True)
class EvidencePlanEntry:
    label: str
    preference: float
    item_support: float
    gap: float
    role: str


@dataclass(frozen=True)
class DomainPrompt:
    task: str
    preference_name: str
    preference_description: str
    item_name: str
    item_description: str
    item_reference: str
    item_plural: str


DOMAIN_PROMPTS = {
    "book": DomainPrompt(
        task=(
            "You are a professional book recommendation explanation assistant. "
            "Your task is to explain why the recommended book matches the reading "
            "preferences of the user and which supplied evidence supports its relative advantage."
        ),
        preference_name="READER PREFERENCE SUMMARY",
        preference_description="a summary of the historical reading preferences of the user",
        item_name="RECOMMENDED BOOK INFORMATION",
        item_description="information about the book selected by the system",
        item_reference="recommended-book information",
        item_plural="books",
    ),
    "movies": DomainPrompt(
        task=(
            "You are a professional film and television recommendation explanation assistant. "
            "Your task is to explain why the recommended title matches the viewing preferences "
            "of the user and which supplied evidence supports its relative advantage."
        ),
        preference_name="VIEWER PREFERENCE SUMMARY",
        preference_description="a summary of the historical viewing preferences of the user",
        item_name="RECOMMENDED TITLE INFORMATION",
        item_description="information about the film or television title selected by the system",
        item_reference="recommended-title information",
        item_plural="titles",
    ),
    "yelp": DomainPrompt(
        task=(
            "You are a professional local business recommendation explanation assistant. "
            "Your task is to explain why the recommended business matches the preferences "
            "of the user and which supplied evidence supports its relative advantage."
        ),
        preference_name="USER PREFERENCE SUMMARY",
        preference_description="a summary of the historical preferences of the user",
        item_name="RECOMMENDED BUSINESS INFORMATION",
        item_description="information about the business selected by the system",
        item_reference="recommended-business information",
        item_plural="businesses",
    ),
}


def normalize_domain(domain: str) -> str:
    normalized = "".join(character for character in domain.lower() if character.isalnum())
    aliases = {
        "book": "book",
        "books": "book",
        "movie": "movies",
        "movies": "movies",
        "moviestv": "movies",
        "filmtv": "movies",
        "yelp": "yelp",
        "business": "yelp",
        "businesses": "yelp",
    }
    if normalized not in aliases:
        raise ValueError("domain must be one of: book, movies, yelp")
    return aliases[normalized]


def assign_role(gap: float, threshold: float) -> str:
    if gap > threshold:
        return "Advantage"
    if gap < -threshold:
        return "TradeOff"
    return "SharedMatch"


def build_content_plan(
    labels: Sequence[str],
    preference: Sequence[float],
    item_support: Sequence[float],
    gaps: Sequence[float],
    threshold: float,
) -> list[EvidencePlanEntry]:
    entries = [
        EvidencePlanEntry(
            label=label,
            preference=float(pref),
            item_support=float(support),
            gap=float(gap),
            role=assign_role(float(gap), threshold),
        )
        for label, pref, support, gap in zip(labels, preference, item_support, gaps)
        if float(support) > 0
    ]
    role_order = {"Advantage": 0, "SharedMatch": 1, "TradeOff": 2}
    return sorted(entries, key=lambda x: (role_order[x.role], -abs(x.gap), x.label))


def serialize_content_plan(entries: Sequence[EvidencePlanEntry]) -> str:
    lines = [
        f"- {entry.label}: preference={entry.preference:.2f}, "
        f"item_support={entry.item_support:.2f}, gap={entry.gap:.2f}, role={entry.role}"
        for entry in entries
    ]
    return "\n".join(lines)


def teacher_prompt(user_summary: str, item_information: str, plan: str) -> str:
    return f"""Write a 2-3 sentence recommendation explanation of at most 80 tokens.
Use only the supplied evidence. Include at least one preference match, at least one
Advantage when available, and at most one TradeOff. Do not print numeric values or
name competing items; say 'similarly ranked alternatives' for comparisons.

User preferences: {user_summary}
Recommended item: {item_information}
Evidence plan:
{plan}

Explanation:"""


def generation_prompt(
    domain: str,
    user_summary: str,
    item_information: str,
    selected_labels: Sequence[str],
    gaps: Sequence[float],
) -> str:
    prompt = DOMAIN_PROMPTS[normalize_domain(domain)]
    evidence = ", ".join(selected_labels)
    gap_text = ", ".join(
        f"{label}: {float(gap):+.2f}" for label, gap in zip(selected_labels, gaps)
    )
    return f"""{prompt.task}

I will provide you with:

{prompt.preference_name}: {prompt.preference_description}.

{prompt.item_name}: {prompt.item_description}.

SELECTED EVIDENCE: the evidence units used to determine the final reranking score.

CONTRIBUTION GAPS: the relative contribution of each selected evidence unit compared with the strongest competing candidates.

Task requirements:

1. Ground the explanation only in the supplied preference summary, {prompt.item_reference}, selected evidence, contribution gaps, and prepended soft evidence tokens.
2. Describe evidence with positive contribution gaps as relative advantages. Describe evidence with negative contribution gaps as trade-offs when relevant.
3. Do not describe or infer specific attributes of competing {prompt.item_plural}. Generic expressions such as "similar options" may be used when necessary.
4. Do not mention numerical scores, contribution values, candidate identifiers, evidence identifiers, or internal model operations.
5. Do not introduce unsupported attributes or generic praise.
6. Return one concise paragraph within 80 generated tokens and no additional text.

{prompt.preference_name}: {user_summary}

{prompt.item_name}: {item_information}

SELECTED EVIDENCE: {evidence}

CONTRIBUTION GAPS: {gap_text}

Your reply:"""


class EvidenceProjector(nn.Module):
    def __init__(self, evidence_dim: int, llm_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(evidence_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, contextualized_evidence: torch.Tensor) -> torch.Tensor:
        return self.network(contextualized_evidence)


class DecisionAlignedGenerator(nn.Module):
    def __init__(self, config: MeritConfig, torch_dtype: torch.dtype | None = None):
        super().__init__()
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install MERIT with the [generation] extra") from exc

        generation = config.generation
        self.tokenizer = AutoTokenizer.from_pretrained(generation.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backbone = AutoModelForCausalLM.from_pretrained(
            generation.model_name, torch_dtype=torch_dtype
        )
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        lora_config = LoraConfig(
            r=generation.lora_rank,
            lora_alpha=generation.lora_alpha,
            lora_dropout=generation.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.llm = get_peft_model(backbone, lora_config)
        llm_dim = int(backbone.config.hidden_size)
        self.projector = EvidenceProjector(config.mceb.hidden_dim, llm_dim)

    def forward(
        self,
        selected_context: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        soft_tokens = self.projector(selected_context)
        embedding_layer = self.llm.get_input_embeddings()
        text_tokens = embedding_layer(input_ids)
        soft_tokens = soft_tokens.to(text_tokens.dtype)
        # Prepend selected MCEB evidence as trainable soft prompt tokens.
        inputs_embeds = torch.cat([soft_tokens, text_tokens], dim=1)
        soft_mask = torch.ones(
            soft_tokens.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_attention = torch.cat([soft_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            # Soft evidence tokens condition generation but never contribute to language loss.
            ignore = torch.full(
                soft_tokens.shape[:2], -100, dtype=labels.dtype, device=labels.device
            )
            full_labels = torch.cat([ignore, labels], dim=1)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        selected_context: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> torch.Tensor:
        soft_tokens = self.projector(selected_context)
        text_tokens = self.llm.get_input_embeddings()(input_ids)
        soft_tokens = soft_tokens.to(text_tokens.dtype)
        inputs_embeds = torch.cat([soft_tokens, text_tokens], dim=1)
        soft_mask = torch.ones(
            soft_tokens.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_attention = torch.cat([soft_mask, attention_mask], dim=1)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
