from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from confidence_tom.client import LLMClient, _coerce_json_response
from confidence_tom.dataset_models import StaticTask
from confidence_tom.intervention import (
    ModelPricing,
    NextStepOutput,
    OracleGainStepResult,
    OracleGainTaskResult,
    StepRecord,
    StepwiseWorkerOutput,
    parse_with_llm_fallback,
    trace_to_cost,
)
from confidence_tom.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.static_evaluators import build_static_evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

_SMALL_SYSTEM_PROMPT = """You are a careful math reasoning worker solving the task one step at a time.
Return exactly one JSON object only. No prose before or after JSON.

Schema:
{
  "next_step": {
    "step": 1,
    "subgoal": "...",
    "reasoning": "...",
    "partial_answer": "...",
    "step_confidence": 0-100,
    "assumptions": ["..."],
    "uncertainty_note": "...",
    "is_revision": false,
    "revision_target": "...",
    "intermediate_result": "...",
    "verification_status": "none|partial|verified|failed"
  },
  "done": false,
  "final_answer": "",
  "final_confidence": 0
}

Rules:
- Produce exactly one next step.
- If the problem is solved at this step, set done=true and provide final_answer and final_confidence.
- Keep reasoning concise and specific.
- Do not generate multiple future steps.
"""

_LARGE_SYSTEM_PROMPT = """You are a stronger takeover math reasoning worker.
You receive the original question and the small worker's partial trace.
Continue from that state, correcting errors if needed, and finish the task.
Return exactly one JSON object only, using this schema:
{
  "steps": [ ... step objects ... ],
  "final_answer": "...",
  "final_confidence": 0-100
}
Do not include prose outside JSON.
"""


def _auto_finalize_from_step(step: StepRecord, parsed: NextStepOutput) -> tuple[bool, str, int]:
    if parsed.parse_incomplete:
        return False, "", 0
    if parsed.done and parsed.final_answer.strip():
        return True, parsed.final_answer.strip(), parsed.final_confidence
    if step.verification_status == "verified" and step.partial_answer.strip():
        return (
            True,
            step.partial_answer.strip(),
            max(int(step.step_confidence), int(parsed.final_confidence or 0)),
        )
    return False, "", 0


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = self._load()
        self.index = {row["task_id"]: i for i, row in enumerate(self.rows)}

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def has(self, task_id: str) -> bool:
        return task_id in self.index

    def save(self, result: OracleGainTaskResult) -> None:
        row = result.model_dump()
        existing = self.index.get(result.task_id)
        if existing is None:
            self.index[result.task_id] = len(self.rows)
            self.rows.append(row)
        else:
            self.rows[existing] = row
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2))
        tmp.replace(self.path)


class PartialTaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, task_id: str, payload: dict[str, Any]) -> None:
        tmp = self.root / f"{task_id}.tmp"
        final = self.root / f"{task_id}.json"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(final)

    def clear(self, task_id: str) -> None:
        path = self.root / f"{task_id}.json"
        if path.exists():
            path.unlink()


def _sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def _pricing_from_cfg(cfg: DictConfig, model_name: str) -> Optional[ModelPricing]:
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


def _steps_to_json(steps: list[StepRecord]) -> str:
    return json.dumps([s.model_dump() for s in steps], ensure_ascii=False, indent=2)


async def _generate_json_text(
    client: LLMClient,
    messages: list[dict[str, str]],
    timeout_sec: int,
    tag: str,
    task_id: str,
) -> tuple[str, Any]:
    raw, trace = await asyncio.wait_for(
        client.agenerate_text_with_trace(
            messages, max_tokens=client.max_tokens, temperature=client.temperature
        ),
        timeout=timeout_sec,
    )
    logger.info(
        "%s.raw task=%s raw_len=%d raw_head=%r", tag, task_id, len(raw or ""), (raw or "")[:200]
    )
    return raw, trace


async def _continue_small_worker(
    task: StaticTask,
    client: LLMClient,
    extract_client: Optional[LLMClient],
    existing_steps: list[StepRecord],
    max_steps: int,
    timeout_sec: int,
    tag_prefix: str,
) -> tuple[StepwiseWorkerOutput, Any]:
    steps = list(existing_steps)
    traces: list[Any] = []
    final_answer = ""
    final_confidence = 0

    start_step = len(steps) + 1
    for step_idx in range(start_step, max_steps + 1):
        if steps:
            user_content = (
                f"Original question:\n{task.question}\n\n"
                f"Steps so far:\n{_steps_to_json(steps)}\n\n"
                f"Produce step {step_idx} only."
            )
        else:
            user_content = f"Original question:\n{task.question}\n\nProduce step 1 only."

        messages = [
            {"role": "system", "content": _SMALL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        logger.info("%s.start task=%s step=%d", tag_prefix, task.id, step_idx)
        raw, trace = await _generate_json_text(client, messages, timeout_sec, tag_prefix, task.id)
        traces.append(trace)
        parsed = _coerce_json_response(raw, NextStepOutput)
        if parsed is None and extract_client is not None:
            parsed, extract_trace = await parse_with_llm_fallback(
                raw, NextStepOutput, extract_client
            )
            logger.info(
                "%s.extract_fallback task=%s step=%d parsed=%s",
                tag_prefix,
                task.id,
                step_idx,
                parsed is not None,
            )
            if extract_trace is not None:
                trace = extract_trace
        logger.info(
            "%s.done task=%s step=%d parsed=%s", tag_prefix, task.id, step_idx, parsed is not None
        )
        if parsed is None:
            break
        if parsed.parse_incomplete:
            logger.info(
                "%s.incomplete task=%s step=%d note=%r",
                tag_prefix,
                task.id,
                step_idx,
                parsed.parse_incomplete_note,
            )

        step = parsed.next_step
        if step.step != step_idx:
            step.step = step_idx
        steps.append(step)

        should_finish, inferred_answer, inferred_conf = _auto_finalize_from_step(step, parsed)
        if should_finish:
            final_answer = inferred_answer
            final_confidence = inferred_conf
            break

    return StepwiseWorkerOutput(
        steps=steps, final_answer=final_answer, final_confidence=final_confidence
    ), (traces[-1] if traces else None)


async def _run_large_takeover(
    task: StaticTask,
    client: LLMClient,
    extract_client: Optional[LLMClient],
    steps_so_far: list[StepRecord],
    handoff_step: int,
    timeout_sec: int,
) -> tuple[Optional[StepwiseWorkerOutput], Any]:
    prefix = StepwiseWorkerOutput(steps=steps_so_far, final_answer="", final_confidence=0)
    user_prompt = (
        f"Original question:\n{task.question}\n\n"
        f"Small worker trace up to step {handoff_step}:\n"
        f"{prefix.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
        "Continue from here. If the partial trace is wrong, explicitly correct it and finish the problem."
    )
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    logger.info("large_takeover.start task=%s handoff_step=%s", task.id, handoff_step)
    raw, trace = await _generate_json_text(client, messages, timeout_sec, "large_takeover", task.id)
    parsed = _coerce_json_response(raw, StepwiseWorkerOutput)
    if parsed is None and extract_client is not None:
        parsed, extract_trace = await parse_with_llm_fallback(
            raw, StepwiseWorkerOutput, extract_client
        )
        logger.info(
            "large_takeover.extract_fallback task=%s parsed=%s", task.id, parsed is not None
        )
        if extract_trace is not None:
            trace = extract_trace
    logger.info("large_takeover.done task=%s parsed=%s", task.id, parsed is not None)
    return parsed, trace


async def _map_task(task: StaticTask, cfg: DictConfig) -> OracleGainTaskResult:
    partial_store = PartialTaskStore(Path(to_absolute_path(str(cfg.output_dir))) / "partials")
    small_client = LLMClient(
        model=str(cfg.small_worker.model),
        temperature=float(cfg.small_worker.temperature),
        max_tokens=int(cfg.small_worker.max_tokens),
        reasoning_effort=cfg.small_worker.get("reasoning_effort"),
    )
    large_client = LLMClient(
        model=str(cfg.large_worker.model),
        temperature=float(cfg.large_worker.temperature),
        max_tokens=int(cfg.large_worker.max_tokens),
        reasoning_effort=cfg.large_worker.get("reasoning_effort"),
    )
    extract_cfg = cfg.get("extractor", {})
    extract_client = None
    if bool(extract_cfg.get("enabled", False)):
        extract_client = LLMClient(
            model=str(extract_cfg.model),
            temperature=float(extract_cfg.get("temperature", 0.0)),
            max_tokens=int(extract_cfg.get("max_tokens", 512)),
            reasoning_effort=extract_cfg.get("reasoning_effort"),
        )
    evaluator = build_static_evaluator(task)

    base_trace, base_api_trace = await _continue_small_worker(
        task,
        small_client,
        extract_client,
        existing_steps=[],
        max_steps=int(cfg.small_worker.max_steps),
        timeout_sec=int(cfg.timeouts.small_worker_sec),
        tag_prefix="base_small",
    )
    base_eval = (
        evaluator(base_trace.final_answer, task) if base_trace.final_answer else evaluator("", task)
    )
    logger.info(
        "base_small.summary task=%s steps=%d final_answer=%r correct=%s",
        task.id,
        len(base_trace.steps),
        base_trace.final_answer,
        base_eval.is_correct,
    )
    partial_store.save(
        task.id,
        {
            "task_id": task.id,
            "status": "base_small_done",
            "small_model": str(cfg.small_worker.model),
            "large_model": str(cfg.large_worker.model),
            "base_small_answer": base_trace.final_answer,
            "base_small_correct": base_eval.is_correct,
            "base_small_trace": base_trace.model_dump(),
            "oracle_gain_steps": [],
        },
    )

    small_pricing = _pricing_from_cfg(cfg, str(cfg.small_worker.model))
    large_pricing = _pricing_from_cfg(cfg, str(cfg.large_worker.model))

    oracle_steps: list[OracleGainStepResult] = []
    for step_index in range(1, len(base_trace.steps) + 1):
        prefix_steps = base_trace.steps[:step_index]
        logger.info("oracle.step task=%s step=%d/%d", task.id, step_index, len(base_trace.steps))

        try:
            small_continue_trace, small_continue_api = await _continue_small_worker(
                task,
                small_client,
                extract_client,
                existing_steps=prefix_steps,
                max_steps=int(cfg.small_worker.max_steps),
                timeout_sec=int(cfg.timeouts.small_worker_sec),
                tag_prefix="small_continue",
            )
        except Exception:
            logger.error(
                "small_continue.error task=%s step=%d\n%s",
                task.id,
                step_index,
                traceback.format_exc(),
            )
            small_continue_trace, small_continue_api = (
                StepwiseWorkerOutput(steps=list(prefix_steps)),
                None,
            )

        try:
            large_takeover_trace, large_takeover_api = await _run_large_takeover(
                task,
                large_client,
                extract_client,
                steps_so_far=prefix_steps,
                handoff_step=step_index,
                timeout_sec=int(cfg.timeouts.large_worker_sec),
            )
        except Exception:
            logger.error(
                "large_takeover.error task=%s step=%d\n%s",
                task.id,
                step_index,
                traceback.format_exc(),
            )
            large_takeover_trace, large_takeover_api = None, None

        small_continue_eval = (
            evaluator(small_continue_trace.final_answer, task)
            if small_continue_trace.final_answer
            else evaluator("", task)
        )
        large_takeover_eval = (
            evaluator(large_takeover_trace.final_answer, task)
            if large_takeover_trace is not None and large_takeover_trace.final_answer
            else evaluator("", task)
        )

        oracle_steps.append(
            OracleGainStepResult(
                step_index=step_index,
                prefix_steps=prefix_steps,
                small_continue_answer=small_continue_trace.final_answer,
                small_continue_correct=small_continue_eval.is_correct,
                large_takeover_answer=large_takeover_trace.final_answer
                if large_takeover_trace is not None
                else "",
                large_takeover_correct=large_takeover_eval.is_correct,
                delta_correctness=float(large_takeover_eval.score - small_continue_eval.score),
                small_continue_cost=trace_to_cost(small_continue_api, small_pricing),
                large_takeover_cost=trace_to_cost(large_takeover_api, large_pricing),
                small_continue_trace=small_continue_trace,
                large_takeover_trace=large_takeover_trace,
                small_continue_api_trace=small_continue_api,
                large_takeover_api_trace=large_takeover_api,
            )
        )
        logger.info(
            "oracle.step.summary task=%s step=%d small_correct=%s large_correct=%s delta=%.3f",
            task.id,
            step_index,
            small_continue_eval.is_correct,
            large_takeover_eval.is_correct,
            float(large_takeover_eval.score - small_continue_eval.score),
        )
        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "oracle_step_done",
                "completed_step_index": step_index,
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "base_small_answer": base_trace.final_answer,
                "base_small_correct": base_eval.is_correct,
                "base_small_trace": base_trace.model_dump(),
                "oracle_gain_steps": [s.model_dump() for s in oracle_steps],
            },
        )

    result = OracleGainTaskResult(
        task_id=task.id,
        benchmark=task.source,
        small_model=str(cfg.small_worker.model),
        large_model=str(cfg.large_worker.model),
        base_small_answer=base_trace.final_answer,
        base_small_correct=base_eval.is_correct,
        base_small_trace=base_trace,
        oracle_gain_steps=oracle_steps,
        metadata={
            "reference_answer": task.reference_answer,
            "evaluator_name": task.evaluator_name,
            "category": task.category,
            "external_difficulty": task.external_difficulty,
        },
    )
    partial_store.clear(task.id)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="oracle_gain_mapping")
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = str(cfg.dataset.benchmark)
    if benchmark_name == "olympiadbench":
        questions = load_olympiadbench(num_samples=int(cfg.dataset.olympiadbench))
    elif benchmark_name == "livebench_reasoning":
        questions = load_livebench_reasoning(num_samples=int(cfg.dataset.livebench))
    else:
        raise ValueError(f"Unsupported oracle-gain benchmark: {benchmark_name}")

    if cfg.dataset.limit:
        questions = questions[: int(cfg.dataset.limit)]

    out_path = (
        output_dir
        / f"{_sanitize_label(str(cfg.small_worker.label))}_to_{_sanitize_label(str(cfg.large_worker.label))}.json"
    )
    store = ResultStore(out_path)
    logger.info("Loaded %d tasks for oracle gain mapping", len(questions))

    async def _run_all() -> None:
        for i, task in enumerate(questions, start=1):
            if store.has(task.id):
                continue
            logger.info("[%d/%d] %s", i, len(questions), task.id)
            try:
                result = await asyncio.wait_for(
                    _map_task(task, cfg),
                    timeout=float(cfg.timeouts.task_sec),
                )
            except asyncio.TimeoutError:
                logger.error("task.timeout task=%s timeout_sec=%s", task.id, cfg.timeouts.task_sec)
                continue
            except Exception:
                logger.error("task.error task=%s\n%s", task.id, traceback.format_exc())
                continue
            store.save(result)

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
