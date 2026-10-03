from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import MeritConfig
from .data import MeritJsonlDataset, collate_merit, iter_jsonl
from .generation import (
    DecisionAlignedGenerator,
    build_content_plan,
    generation_prompt,
    normalize_domain,
    serialize_content_plan,
    teacher_prompt,
)
from .model import MERITReranker, contribution_gaps, gather_selected_context
from .training import seed_everything, tensor_batch_to_device


def build_generation_plans(
    config: MeritConfig,
    data_path: str | Path,
    reranker_checkpoint: str | Path,
    output_path: str | Path,
    device: str | None = None,
    batch_size: int = 64,
    domain: str | None = None,
) -> dict:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MERITReranker(config).to(target_device)
    checkpoint = torch.load(reranker_checkpoint, map_location=target_device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(
        MeritJsonlDataset(data_path),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_merit(rows, config.data.candidate_pool_size),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for raw_batch in loader:
            batch = tensor_batch_to_device(raw_batch, target_device)
            result = model(
                batch["candidate_support"],
                batch["preference_strength"],
                batch["semantic_embeddings"],
                batch["matching_features"],
                batch["candidate_mask"],
                batch["evidence_mask"],
                augment_pseudo=False,
                compute_stability=False,
            )
            recommendation, competitors, gaps = contribution_gaps(
                result, config.generation.competitor_count
            )
            selected_context = gather_selected_context(result)
            for row_index, raw in enumerate(raw_batch["raw_examples"]):
                recommendation_index = int(recommendation[row_index].item())
                selected_indices = result.selected_indices[row_index].cpu().tolist()
                selected_labels = [raw["evidence_labels"][index] for index in selected_indices]
                selected_preference = [raw["preference_strength"][index] for index in selected_indices]
                selected_support = [
                    raw["candidate_support"][recommendation_index][index]
                    for index in selected_indices
                ]
                selected_gaps = [float(gaps[row_index, index].item()) for index in selected_indices]
                plan_entries = build_content_plan(
                    selected_labels,
                    selected_preference,
                    selected_support,
                    selected_gaps,
                    config.generation.role_threshold,
                )
                plan = serialize_content_plan(plan_entries)
                preference_labels = raw.get("user_preference_labels", raw["evidence_labels"])
                user_summary = ", ".join(
                    preference_labels[: config.generation.summary_evidence_count]
                )
                name = raw.get("candidate_names", raw["candidate_ids"])[recommendation_index]
                category = raw.get("candidate_categories", [""] * len(raw["candidate_ids"]))[
                    recommendation_index
                ]
                metadata = raw.get("candidate_metadata", [""] * len(raw["candidate_ids"]))[
                    recommendation_index
                ]
                support_text = ", ".join(
                    f"{label}={float(support):.2f}"
                    for label, support in zip(selected_labels, selected_support)
                )
                item_fields = [f"name={name}"]
                if category:
                    item_fields.append(f"category={category}")
                if metadata:
                    item_fields.append(f"metadata={metadata}")
                item_fields.append(f"selected_evidence_support={support_text}")
                item_information = " | ".join(item_fields)
                prompt_domain = normalize_domain(
                    raw.get("domain", domain or config.generation.domain)
                )
                payload = {
                    "user_id": raw["user_id"],
                    "timestamp": raw["timestamp"],
                    "domain": prompt_domain,
                    "recommended_candidate_index": recommendation_index,
                    "recommended_item_id": raw["candidate_ids"][recommendation_index],
                    "competitor_indices": competitors[row_index].cpu().tolist(),
                    "selected_indices": selected_indices,
                    "selected_labels": selected_labels,
                    "selected_preference": selected_preference,
                    "selected_support": selected_support,
                    "contribution_gaps": selected_gaps,
                    "selected_context": selected_context[row_index].cpu().float().tolist(),
                    "user_summary": user_summary,
                    "item_information": item_information,
                    "content_plan": plan,
                    "teacher_prompt": teacher_prompt(user_summary, item_information, plan),
                    "generation_prompt": generation_prompt(
                        prompt_domain,
                        user_summary,
                        item_information,
                        selected_labels,
                        selected_gaps,
                    ),
                }
                if raw.get("target_explanation"):
                    payload["target_explanation"] = raw["target_explanation"]
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
    return {"examples": written, "output": str(output)}


def generate_teacher_targets(
    config: MeritConfig,
    plans_path: str | Path,
    output_path: str | Path,
    batch_size: int = 1,
    device_map: str = "auto",
) -> dict:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install MERIT with the [generation] extra") from exc
    tokenizer = AutoTokenizer.from_pretrained(config.generation.teacher_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.generation.teacher_model_name,
        torch_dtype="auto",
        device_map=device_map,
    )
    model.eval()
    rows = list(iter_jsonl(plans_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    first_device = next(model.parameters()).device
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["teacher_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.generation.max_input_tokens,
            ).to(first_device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config.generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text in zip(batch, texts):
                payload = {**row, "target_explanation": text.strip()}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"examples": len(rows), "output": str(output)}


def _collate_generation(rows: list[dict], tokenizer, max_length: int) -> dict:
    input_rows = []
    label_rows = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["generation_prompt"], add_special_tokens=True)
        target_ids = tokenizer.encode(
            " " + row["target_explanation"], add_special_tokens=False
        )
        if tokenizer.eos_token_id is not None:
            target_ids.append(tokenizer.eos_token_id)
        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1] + [target_ids[-1]]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
        combined = prompt_ids + target_ids
        # Supervise only the teacher explanation, not the textual conditioning prompt.
        labels = [-100] * len(prompt_ids) + target_ids
        input_rows.append(combined)
        label_rows.append(labels)
    maximum = max(len(row) for row in input_rows)
    input_ids = torch.full(
        (len(rows), maximum), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros(len(rows), maximum, dtype=torch.long)
    labels = torch.full((len(rows), maximum), -100, dtype=torch.long)
    for index, (tokens, target) in enumerate(zip(input_rows, label_rows)):
        input_ids[index, : len(tokens)] = torch.as_tensor(tokens)
        attention[index, : len(tokens)] = 1
        labels[index, : len(target)] = torch.as_tensor(target)
    context = torch.as_tensor([row["selected_context"] for row in rows], dtype=torch.float32)
    return {
        "selected_context": context,
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
    }


def fit_generator(
    config: MeritConfig,
    train_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> dict:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("install MERIT with the [generation] extra") from exc
    seed_everything(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = (
        torch.bfloat16
        if target_device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = DecisionAlignedGenerator(config, torch_dtype=model_dtype).to(target_device)
    rows = list(iter_jsonl(train_path))
    rows = [row for row in rows if row.get("target_explanation")]
    loader = DataLoader(
        rows,
        batch_size=config.generation.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_generation(
            batch, model.tokenizer, config.generation.max_input_tokens
        ),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=config.generation.learning_rate)
    final_loss = 0.0
    steps = 0
    for _ in range(config.generation.epochs):
        model.train()
        for raw_batch in loader:
            batch = tensor_batch_to_device(raw_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            final_loss += float(result.loss.detach().item())
            steps += 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(output / "lora_adapter")
    model.tokenizer.save_pretrained(output / "tokenizer")
    torch.save(model.projector.state_dict(), output / "evidence_projector.pt")
    torch.save(
        get_peft_model_state_dict(model.llm), output / "lora_state.pt"
    )
    return {
        "examples": len(rows),
        "steps": steps,
        "mean_training_loss": final_loss / max(steps, 1),
        "output": str(output),
    }


def generate_student_outputs(
    config: MeritConfig,
    plans_path: str | Path,
    checkpoint_dir: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> dict:
    try:
        from peft import set_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("install MERIT with the [generation] extra") from exc
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = (
        torch.bfloat16
        if target_device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = DecisionAlignedGenerator(config, torch_dtype=model_dtype).to(target_device)
    checkpoint = Path(checkpoint_dir)
    model.projector.load_state_dict(
        torch.load(checkpoint / "evidence_projector.pt", map_location=target_device, weights_only=True)
    )
    lora_state = torch.load(
        checkpoint / "lora_state.pt", map_location=target_device, weights_only=True
    )
    set_peft_model_state_dict(model.llm, lora_state)
    model.eval()
    rows = list(iter_jsonl(plans_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for row in rows:
            encoded = model.tokenizer(
                row["generation_prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=config.generation.max_input_tokens,
            ).to(target_device)
            context = torch.as_tensor(
                row["selected_context"], dtype=torch.float32, device=target_device
            ).unsqueeze(0)
            generated = model.generate(
                context,
                encoded["input_ids"],
                encoded["attention_mask"],
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                max_new_tokens=config.generation.max_new_tokens,
            )
            text = model.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            handle.write(
                json.dumps({**row, "generated_explanation": text}, ensure_ascii=False) + "\n"
            )
    return {"examples": len(rows), "output": str(output)}
