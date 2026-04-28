from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.intervention import (
    PrefixOracleGainStepResult,
    PrefixOracleGainTaskResult,
)
from experiments.mainline.run.core.common import (
    client_kwargs_from_cfg as _client_kwargs_from_cfg,
    load_static_questions,
    sanitize_label as _sanitize_label,
)
from experiments.mainline.run.core.run_prefix_oracle_gain_mapping import (
    ResultStore,
    PartialTaskStore,
    _run_full_trace,
    _segment_full_trace,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


async def _map_task_small_only(task: Any, cfg: DictConfig) -> PrefixOracleGainTaskResult:
    partial_store = PartialTaskStore(Path(to_absolute_path(str(cfg.output_dir))) / "partials")
    from confidence_tom.infra.client import LLMClient

    small_client = LLMClient(**_client_kwargs_from_cfg(cfg.small_worker))
    extract_client = None
    if bool(cfg.extractor.get("enabled", False)):
        extract_client = LLMClient(**_client_kwargs_from_cfg(cfg.extractor))

    evaluator = build_static_evaluator(task)
    execution_cfg = cfg.execution
    retry_attempts = int(execution_cfg.retry_attempts)
    retry_backoff_sec = float(execution_cfg.retry_backoff_sec)

    trace_id = f"{task.id}_reentry"
    resume_payload = partial_store.load(task.id) if bool(execution_cfg.get("resume_from_partials", True)) else None
    full_text = ""
    full_answer = ""
    full_trace_api = None
    parse_incomplete = False
    segments: list[Any] = []
    oracle_steps: list[PrefixOracleGainStepResult] = []

    if resume_payload and resume_payload.get("status") in {"full_trace_done", "prefix_step_done"}:
        trace_id = str(resume_payload.get("trace_id") or trace_id)
        full_text = str(resume_payload.get("full_trace_text") or "")
        full_answer = str(resume_payload.get("full_trace_answer") or "")
        parse_incomplete = bool(resume_payload.get("parse_incomplete", False))
        segments = list(resume_payload.get("segments", []))
        full_trace_api = resume_payload.get("full_trace_api_trace")
        logger.info("partial.resume task=%s segments=%d", task.id, len(segments))
    else:
        full_text, full_answer, full_trace_api = await _run_full_trace(
            task,
            small_client,
            extract_client,
            int(cfg.timeouts.full_trace_sec),
            retry_attempts,
            retry_backoff_sec,
        )
        segments, parsed_final_answer, parse_incomplete = await _segment_full_trace(
            full_text, extract_client, task.id
        )
        if parsed_final_answer:
            full_answer = parsed_final_answer
        full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
        logger.info(
            "full_trace.summary task=%s segments=%d final_answer=%r correct=%s parse_incomplete=%s",
            task.id,
            len(segments),
            full_answer,
            full_eval.is_correct,
            parse_incomplete,
        )
        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "full_trace_done",
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "trace_id": trace_id,
                "full_trace_text": full_text,
                "full_trace_answer": full_answer,
                "full_trace_correct": full_eval.is_correct,
                "full_trace_api_trace": (
                    full_trace_api.model_dump()
                    if full_trace_api is not None and hasattr(full_trace_api, "model_dump")
                    else full_trace_api
                ),
                "parse_incomplete": parse_incomplete,
                "segments": [segment.model_dump() for segment in segments],
                "prefix_oracle_steps": [],
            },
        )

    full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
    logger.info(
        "prepare.light task=%s segments=%d prepare_mode=segments_only",
        task.id,
        len(segments),
    )

    return PrefixOracleGainTaskResult(
        task_id=task.id,
        benchmark=task.source,
        small_model=str(cfg.small_worker.model),
        large_model=str(cfg.large_worker.model),
        trace_id=trace_id,
        full_trace_text=full_text,
        full_trace_answer=full_answer,
        full_trace_correct=full_eval.is_correct,
        full_trace_api_trace=full_trace_api,
        segments=segments,
        prefix_oracle_steps=oracle_steps,
        metadata={
            "reference_answer": task.reference_answer,
            "evaluator_name": task.evaluator_name,
            "category": task.category,
            "external_difficulty": task.external_difficulty,
            "takeover_mode": "reentry_small_only_prepare",
            "prepare_mode": "segments_only",
        },
    )


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="prefix_oracle_gain_mapping",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{_sanitize_label(str(cfg.small_worker.label))}_small_only.json"
    store = ResultStore(result_path)

    benchmark_name = str(cfg.dataset.benchmark)
    questions = load_static_questions(benchmark_name, cfg.dataset)
    task_concurrency = max(1, int(cfg.execution.task_concurrency))
    task_timeout = float(cfg.timeouts.task_sec)

    sem = asyncio.Semaphore(task_concurrency)
    save_lock = asyncio.Lock()

    async def _run_one(index: int, task: Any) -> None:
        if store.has(task.id):
            return
        async with sem:
            logger.info("[%d/%d] %s", index, len(questions), task.id)
            try:
                result = await asyncio.wait_for(_map_task_small_only(task, cfg), timeout=task_timeout)
            except Exception:
                logger.error("task.error task=%s\n%s", task.id, traceback.format_exc())
                raise
            async with save_lock:
                store.save(result)

    async def _amain() -> None:
        await asyncio.gather(*[_run_one(i, task) for i, task in enumerate(questions, start=1)])

    asyncio.run(_amain())
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
