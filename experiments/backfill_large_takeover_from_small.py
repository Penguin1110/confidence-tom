from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from confidence_tom.client import LLMClient
from confidence_tom.intervention import ModelPricing, trace_to_cost
from confidence_tom.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.static_evaluators import build_static_evaluator

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_prefix_oracle_gain_mapping import (  # noqa: E402
    _LARGE_TAKEOVER_SYSTEM_PROMPT,
    _client_kwargs_from_cfg,
    _pricing_from_cfg,
    _run_continue,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list payload in {path}, got {type(data).__name__}")
    return data


def _infer_benchmark(rows: list[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    values = {
        str(r.get("benchmark", "")).strip() for r in rows if str(r.get("benchmark", "")).strip()
    }
    if len(values) == 1:
        return next(iter(values))
    raise ValueError(f"Cannot infer single benchmark from rows: {sorted(values)}")


def _load_tasks(benchmark: str, limit: int = 0) -> dict[str, Any]:
    if benchmark == "olympiadbench":
        tasks = load_olympiadbench(num_samples=limit if limit > 0 else 10_000)
    elif benchmark == "livebench_reasoning":
        tasks = load_livebench_reasoning(num_samples=limit if limit > 0 else 10_000)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return {t.id: t for t in tasks}


async def _backfill(args: argparse.Namespace) -> None:
    input_path = Path(args.input_json)
    output_path = Path(args.output_json) if args.output_json else input_path
    rows = _load_rows(input_path)
    benchmark = _infer_benchmark(rows, args.benchmark)
    tasks = _load_tasks(benchmark, limit=args.task_limit)
    logger.info("Loaded %d rows from %s", len(rows), input_path)
    logger.info("Benchmark=%s tasks_loaded=%d", benchmark, len(tasks))

    cfg = OmegaConf.create(
        {
            "large_worker": {
                "model": args.large_model,
                "label": args.large_label,
                "max_tokens": args.large_max_tokens,
                "temperature": args.temperature,
                "backend": args.large_backend,
                "local_model_name": args.large_local_model_name,
                "reasoning_effort": args.reasoning_effort,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "enable_thinking": args.enable_thinking,
            },
            "extractor": {
                "enabled": bool(args.extractor_enabled),
                "model": args.extractor_model,
                "label": "Extractor",
                "max_tokens": args.extractor_max_tokens,
                "temperature": 0.0,
                "backend": args.extractor_backend,
            },
            "pricing": {},
        }
    )
    if args.large_input_per_1k or args.large_output_per_1k or args.large_reasoning_per_1k:
        cfg.pricing[str(args.large_model)] = {
            "input_per_1k": float(args.large_input_per_1k),
            "output_per_1k": float(args.large_output_per_1k),
            "reasoning_per_1k": float(args.large_reasoning_per_1k),
        }

    large_client = LLMClient(**_client_kwargs_from_cfg(cfg.large_worker))
    extract_client = None
    if bool(cfg.extractor.enabled):
        extract_client = LLMClient(**_client_kwargs_from_cfg(cfg.extractor))

    large_pricing: ModelPricing | None = _pricing_from_cfg(cfg, str(cfg.large_worker.model))

    touched = 0
    for i, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        task = tasks.get(task_id)
        if task is None:
            logger.warning("skip row=%d task_id=%s reason=task_not_found", i, task_id)
            continue
        evaluator = build_static_evaluator(task)
        steps: list[dict[str, Any]] = list(row.get("prefix_oracle_steps", []))
        if not steps:
            logger.warning("skip row=%d task_id=%s reason=no_prefix_oracle_steps", i, task_id)
            continue

        logger.info("[%d/%d] task=%s steps=%d", i, len(rows), task_id, len(steps))
        task_changed = False
        for step in steps:
            step_index = int(step.get("step_index", 0))
            large_text_existing = str(step.get("large_takeover_text", "") or "")
            large_ans_existing = str(step.get("large_takeover_answer", "") or "")
            already_has_large = bool(large_text_existing or large_ans_existing)
            if already_has_large and not args.force:
                continue

            prefix_text = str(step.get("prefix_text", "") or "")
            if not prefix_text:
                logger.warning("task=%s step=%d missing_prefix_text skip", task_id, step_index)
                continue

            try:
                large_text, large_answer, large_api = await _run_continue(
                    task=task,
                    client=large_client,
                    extract_client=extract_client,
                    system_prompt=_LARGE_TAKEOVER_SYSTEM_PROMPT,
                    prefix_text=prefix_text,
                    timeout_sec=int(args.large_worker_sec),
                    tag="large_takeover_prefix_backfill",
                    retry_attempts=int(args.retry_attempts),
                    retry_backoff_sec=float(args.retry_backoff_sec),
                )
            except Exception:
                logger.error(
                    "large_backfill.error task=%s step=%d\n%s",
                    task_id,
                    step_index,
                    traceback.format_exc(),
                )
                continue

            large_eval = evaluator(large_answer, task) if large_answer else evaluator("", task)
            small_correct = bool(step.get("small_continue_correct", False))
            small_score = 1.0 if small_correct else 0.0
            large_score = float(large_eval.score or 0.0)
            step["large_takeover_answer"] = large_answer
            step["large_takeover_correct"] = bool(large_eval.is_correct)
            step["large_takeover_text"] = large_text
            step["large_takeover_api_trace"] = (
                large_api.model_dump() if hasattr(large_api, "model_dump") else large_api
            )
            step["large_takeover_cost"] = trace_to_cost(large_api, large_pricing).model_dump()
            step["delta_correctness"] = float(large_score - small_score)
            task_changed = True
            touched += 1

            # Incremental save for crash safety.
            row["prefix_oracle_steps"] = steps
            _atomic_write(output_path, rows)

        if task_changed:
            row["prefix_oracle_steps"] = steps
            _atomic_write(output_path, rows)

    logger.info("Done. Updated large-side steps=%d output=%s", touched, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill large-takeover side from small-only prefix rows."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", default=None, help="Default: overwrite input json.")
    parser.add_argument(
        "--benchmark", choices=["olympiadbench", "livebench_reasoning"], default=None
    )
    parser.add_argument(
        "--task-limit", type=int, default=0, help="0 means load full benchmark task index."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing large_takeover fields."
    )

    parser.add_argument("--large-model", default="openai/gpt-5.4")
    parser.add_argument("--large-label", default="GPT-5.4")
    parser.add_argument(
        "--large-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument("--large-local-model-name", default=None)
    parser.add_argument("--large-max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--enable-thinking", type=lambda x: x.lower() == "true", default=None)

    parser.add_argument("--large-worker-sec", type=int, default=900)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)

    parser.add_argument("--extractor-enabled", action="store_true")
    parser.add_argument("--extractor-model", default="openai/gpt-5.4")
    parser.add_argument(
        "--extractor-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument("--extractor-max-tokens", type=int, default=512)

    parser.add_argument("--large-input-per-1k", type=float, default=0.0)
    parser.add_argument("--large-output-per-1k", type=float, default=0.0)
    parser.add_argument("--large-reasoning-per-1k", type=float, default=0.0)
    args = parser.parse_args()

    asyncio.run(_backfill(args))


if __name__ == "__main__":
    main()
