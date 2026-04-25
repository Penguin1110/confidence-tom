from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from confidence_tom.infra.client_utils import local_prompt_text


@lru_cache(maxsize=8)
def load_representation_stack(
    model_name: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    getattr(model, "eval")()
    return torch, tokenizer, model


def extract_prompt_representation(
    *,
    model_name: str,
    trust_remote_code: bool,
    messages: list[dict[str, Any]],
    prefix_text: str,
    selected_layer: int = -1,
) -> dict[str, Any]:
    torch, tokenizer, model = load_representation_stack(model_name, trust_remote_code)
    prompt_text = local_prompt_text(messages, tokenizer)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden states")

    resolved_layer = selected_layer if selected_layer >= 0 else len(hidden_states) + selected_layer
    if resolved_layer < 0 or resolved_layer >= len(hidden_states):
        raise ValueError(f"selected_layer {selected_layer} resolved to invalid index {resolved_layer}")

    layer_hidden = hidden_states[resolved_layer][0].detach().float().cpu().numpy()
    last_hidden = layer_hidden[-1]
    mean_hidden = layer_hidden.mean(axis=0)
    prompt_tokens = int(layer_hidden.shape[0])
    hidden_dim = int(layer_hidden.shape[1])

    prefix_token_count = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
    prefix_token_count = max(0, min(prefix_token_count, prompt_tokens))

    attention_summary: dict[str, Any] = {
        "attention_entropy_mean": None,
        "attention_to_prefix_mass_mean": None,
        "attention_to_prefix_mass_last": None,
        "num_heads": 0,
    }
    if attentions:
        layer_attn = attentions[resolved_layer][0].detach().float().cpu().numpy()
        last_query = layer_attn[:, -1, :]
        entropy_values: list[float] = []
        for head_probs in last_query:
            safe = np.clip(head_probs.astype(np.float64), 1e-12, 1.0)
            entropy_values.append(float(-(safe * np.log(safe)).sum()))

        prefix_slice = slice(max(0, prompt_tokens - prefix_token_count), prompt_tokens)
        prefix_mass_per_head = (
            last_query[:, prefix_slice].sum(axis=1).astype(np.float64)
            if prefix_token_count > 0
            else np.zeros(last_query.shape[0], dtype=np.float64)
        )
        attention_summary = {
            "attention_entropy_mean": float(np.mean(entropy_values)),
            "attention_to_prefix_mass_mean": float(np.mean(prefix_mass_per_head)),
            "attention_to_prefix_mass_last": float(prefix_mass_per_head[-1])
            if len(prefix_mass_per_head)
            else None,
            "num_heads": int(layer_attn.shape[0]),
        }

    return {
        "prompt_text": prompt_text,
        "prompt_tokens": prompt_tokens,
        "prefix_tokens_estimated": prefix_token_count,
        "selected_layer": resolved_layer,
        "hidden_dim": hidden_dim,
        "last_token_hidden": last_hidden.tolist(),
        "mean_pool_hidden": mean_hidden.tolist(),
        **attention_summary,
    }
