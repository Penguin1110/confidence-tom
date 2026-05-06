from __future__ import annotations

import inspect
from typing import Any, Optional, cast

from omegaconf import DictConfig

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import (
    load_aime_2024,
    load_gpqa_diamond,
    load_livebench_reasoning,
    load_math500,
    load_olympiadbench,
)
from confidence_tom.infra.client import LLMClient
from confidence_tom.intervention import ModelPricing


def client_kwargs_from_cfg(worker_cfg: DictConfig) -> dict[str, Any]:
    raw_kwargs = {
        "model": str(worker_cfg.model),
        "temperature": float(worker_cfg.get("temperature", 0.0)),
        "max_tokens": int(worker_cfg.get("max_tokens", 2048)),
        "reasoning_effort": worker_cfg.get("reasoning_effort"),
        "backend": str(worker_cfg.get("backend", "local")),
        "local_model_name": worker_cfg.get("local_model_name"),
        "top_p": worker_cfg.get("top_p"),
        "top_k": worker_cfg.get("top_k"),
        "seed": worker_cfg.get("seed"),
        "num_ctx": worker_cfg.get("num_ctx"),
        "num_predict": worker_cfg.get("num_predict"),
        "enable_thinking": worker_cfg.get("enable_thinking"),
    }
    valid = set(inspect.signature(LLMClient.__init__).parameters.keys())
    valid.discard("self")
    return {k: v for k, v in raw_kwargs.items() if k in valid}


def sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def pricing_from_cfg(cfg: DictConfig, model_name: str) -> Optional[ModelPricing]:
    item = cfg.pricing.get(model_name)
    if not item:
        return None
    pricing = ModelPricing(
        input_per_1k=float(item.get("input_per_1k", 0.0)),
        output_per_1k=float(item.get("output_per_1k", 0.0)),
        reasoning_per_1k=float(item.get("reasoning_per_1k", 0.0)),
    )
    if (
        pricing.input_per_1k == 0.0
        and pricing.output_per_1k == 0.0
        and pricing.reasoning_per_1k == 0.0
    ):
        return None
    return pricing


def load_static_questions(benchmark_name: str, dataset_cfg: DictConfig) -> list[StaticTask]:
    start_index = int(dataset_cfg.get("start_index", 0) or 0)
    limit_value = dataset_cfg.get("limit")
    limit = int(limit_value) if limit_value else 0

    def _requested(default_count: int) -> int:
        if limit > 0:
            return max(default_count, start_index + limit)
        if start_index > 0:
            return default_count + start_index
        return default_count

    if benchmark_name == "olympiadbench":
        questions = load_olympiadbench(num_samples=_requested(int(dataset_cfg.olympiadbench)))
    elif benchmark_name == "livebench_reasoning":
        livebench_count = int(
            dataset_cfg.get("livebench_reasoning", dataset_cfg.get("livebench", 0))
        )
        questions = load_livebench_reasoning(num_samples=_requested(livebench_count))
    elif benchmark_name == "aime_2024":
        default_count = int(dataset_cfg.get("aime_2024", dataset_cfg.get("limit", 0)))
        questions = load_aime_2024(num_samples=_requested(default_count))
    elif benchmark_name == "math500":
        default_count = int(dataset_cfg.get("math500", dataset_cfg.get("limit", 0)))
        questions = load_math500(num_samples=_requested(default_count))
    elif benchmark_name == "gpqa_diamond":
        default_count = int(dataset_cfg.get("gpqa_diamond", dataset_cfg.get("limit", 0)))
        questions = load_gpqa_diamond(num_samples=_requested(default_count))
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    if start_index > 0:
        questions = questions[start_index:]
    if limit > 0:
        questions = questions[:limit]
    return cast(list[StaticTask], questions)
