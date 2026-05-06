from __future__ import annotations

import json as _json
from typing import Any, Optional, Type, cast

from confidence_tom.data.task_models import ApiTrace
from confidence_tom.infra.client_types import T

_LOCAL_MODEL_MAP = {
    "qwen/qwen3-14b:nitro": "Qwen/Qwen3-14B",
    "mistralai/ministral-8b-2512": "mistral-small3.2:24b",
    "meta-llama/llama-4-scout": "llama3.1:8b",
}


def extract_first_json_object(raw: str) -> str:
    """Return the first balanced JSON object substring from a raw model reply."""
    start = raw.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return ""


def coerce_json_response(raw: str, response_model: Type[T]) -> Optional[T]:
    """Fallback parser for models that prepend numbering or prose before JSON."""
    candidate = extract_first_json_object(raw.strip())
    if not candidate:
        return None
    try:
        data = _json.loads(candidate)
    except Exception:
        return None
    try:
        return response_model.model_validate(data)
    except Exception:
        return None


def normalize_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    """Ensure provider-facing chat messages always have a string content field."""
    normalized = dict(message)
    if normalized.get("role") == "assistant" and normalized.get("tool_calls"):
        normalized["content"] = normalized.get("content") or ""
    elif "content" in normalized and normalized["content"] is None:
        normalized["content"] = ""
    return normalized


def api_messages(messages: list[dict[str, Any]]) -> Any:
    """Cast normalized message lists to the OpenAI SDK's chat message union."""
    return cast(Any, messages)


def extract_trace(response: object) -> ApiTrace:
    """Pull all metadata fields out of a raw OpenAI/OpenRouter response object."""
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    choices = getattr(response, "choices", [])
    msg = choices[0].message if choices else None

    reasoning_content = getattr(msg, "reasoning", None) or ""
    raw_content = getattr(msg, "content", None) or ""
    if isinstance(raw_content, list):
        raw_content = _json.dumps(raw_content, ensure_ascii=False)
    else:
        raw_content = str(raw_content)

    cache_read = (
        getattr(prompt_details, "cached_tokens", 0) or getattr(usage, "cache_read_tokens", 0) or 0
    )
    cache_write = (
        getattr(prompt_details, "cache_write_tokens", 0)
        or getattr(usage, "cache_write_tokens", 0)
        or 0
    )

    return ApiTrace(
        model_id=getattr(response, "model", ""),
        request_id=getattr(response, "id", ""),
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        reasoning_content=reasoning_content,
        response_content=raw_content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def resolve_local_model_name(model: str, local_model_name: str | None) -> str:
    if local_model_name:
        return local_model_name
    return _LOCAL_MODEL_MAP.get(model, model)


def local_prompt_text(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool | None = None,
) -> str:
    normalized = [normalize_chat_message(message) for message in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        base_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            try:
                return cast(
                    str,
                    tokenizer.apply_chat_template(
                        normalized, **base_kwargs, enable_thinking=enable_thinking
                    ),
                )
            except Exception:
                pass  # tokenizer doesn't support enable_thinking, retry without
        try:
            return cast(str, tokenizer.apply_chat_template(normalized, **base_kwargs))
        except Exception:
            pass

    lines: list[str] = []
    for msg in normalized:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)
