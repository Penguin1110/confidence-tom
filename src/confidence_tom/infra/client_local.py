from __future__ import annotations

from functools import lru_cache
from typing import Any

from confidence_tom.data.task_models import ApiTrace
from confidence_tom.infra.client_utils import local_prompt_text


@lru_cache(maxsize=8)
def load_local_stack(
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


def local_generate_text(
    *,
    model_name: str,
    trust_remote_code: bool,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> tuple[str, ApiTrace]:
    torch, tokenizer, model = load_local_stack(model_name, trust_remote_code)
    prompt_text = local_prompt_text(messages, tokenizer)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[-1])
    do_sample = bool(temperature > 0)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**encoded, **gen_kwargs)
    gen_tokens = out[0][prompt_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    trace = ApiTrace(
        model_id=model_name,
        response_content=text,
        prompt_tokens=prompt_len,
        completion_tokens=int(gen_tokens.shape[-1]),
        total_tokens=prompt_len + int(gen_tokens.shape[-1]),
    )
    return text, trace
