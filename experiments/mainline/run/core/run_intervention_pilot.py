from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient, _coerce_json_response
from confidence_tom.intervention import (
    InterventionOutcome,
    ModelPricing,
    NextStepOutput,
    StepRecord,
    StepwiseWorkerOutput,
    ThresholdRouter,
    build_state,
    combine_costs,
    estimate_voi,
    extract_features,
    parse_with_llm_fallback,
    trace_to_cost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

_SMALL_SYSTEM_PROMPT = """You are a careful math reasoning worker solving the task
one step at a time.
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
- If the problem is solved at this step, set done=true and provide
  final_answer and final_confidence.
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


class OutcomeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = self._load()
        self.index = {row["task_id"]: i for i, row in enumerate(self.rows)}

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return cast(list[dict[str, Any]], json.loads(self.path.read_text()))
        except Exception:
            return []

    def has(self, task_id: str) -> bool:
        return task_id in self.index

    def save(self, outcome: InterventionOutcome) -> None:
        row = outcome.model_dump()
        existing = self.index.get(outcome.task_id)
        if existing is None:
            self.index[outcome.task_id] = len(self.rows)
            self.rows.append(row)
        else:
            self.rows[existing] = row
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2))
        tmp.replace(self.path)


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


async def _run_small_iterative(
    task: StaticTask,
    client: LLMClient,
    extract_client: Optional[LLMClient],
    cfg: DictConfig,
    router: ThresholdRouter,
) -> tuple[StepwiseWorkerOutput, Any, list[Any], list[Any], Optional[int], str]:
    steps: list[StepRecord] = []
    feature_history = []
    decisions = []
    embedding_window: list[list[float]] = []
    traces: list[Any] = []
    handoff_step: Optional[int] = None
    handoff_trigger = ""
    final_answer = ""
    final_confidence = 0

    for step_idx in range(1, int(cfg.small_worker.max_steps) + 1):
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
        logger.info("small_worker.start task=%s step=%d", task.id, step_idx)
        raw, trace = await _generate_json_text(
            client, messages, int(cfg.timeouts.small_worker_sec), "small_worker", task.id
        )
        traces.append(trace)
        parsed = _coerce_json_response(raw, NextStepOutput)
        if parsed is None and extract_client is not None:
            parsed, extract_trace = await parse_with_llm_fallback(
                raw, NextStepOutput, extract_client
            )
            logger.info(
                "small_worker.extract_fallback task=%s step=%d parsed=%s",
                task.id,
                step_idx,
                parsed is not None,
            )
            if extract_trace is not None:
                trace = extract_trace
        logger.info(
            "small_worker.done task=%s step=%d parsed=%s", task.id, step_idx, parsed is not None
        )
        if parsed is None:
            break
        if parsed.parse_incomplete:
            logger.info(
                "small_worker.incomplete task=%s step=%d note=%r",
                task.id,
                step_idx,
                parsed.parse_incomplete_note,
            )

        step = parsed.next_step
        if step.step != step_idx:
            step.step = step_idx
        steps.append(step)

        if bool(cfg.embedding.enabled):
            try:
                logger.info("embedding.start task=%s step=%d", task.id, step_idx)
                emb = await asyncio.wait_for(
                    client.aembed_text(
                        step.reasoning or step.partial_answer, model=str(cfg.embedding.model)
                    ),
                    timeout=int(cfg.timeouts.embedding_sec),
                )
                embedding_window.append(emb)
                logger.info("embedding.done task=%s step=%d", task.id, step_idx)
            except TimeoutError:
                logger.warning("embedding.timeout task=%s step=%d", task.id, step_idx)
            except Exception as e:
                logger.warning("embedding.error task=%s step=%d err=%s", task.id, step_idx, e)

        state = build_state(task.id, task.question, steps, len(steps))
        features = extract_features(state, embedding_window if embedding_window else None)
        decision = router.decide(features)
        feature_history.append(features)
        decisions.append(decision)
        logger.info(
            "router.decision task=%s step=%d handoff=%s score=%.3f reason=%s",
            task.id,
            step_idx,
            decision.handoff,
            decision.score,
            decision.reason,
        )

        should_finish, inferred_answer, inferred_conf = _auto_finalize_from_step(step, parsed)
        if should_finish:
            final_answer = inferred_answer
            final_confidence = inferred_conf
            break

        if decision.handoff:
            handoff_step = step_idx
            handoff_trigger = decision.reason
            break

    return (
        StepwiseWorkerOutput(
            steps=steps, final_answer=final_answer, final_confidence=final_confidence
        ),
        traces[-1] if traces else None,
        feature_history,
        decisions,
        handoff_step,
        handoff_trigger,
    )


async def _run_takeover_worker(
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
        "Continue from here. If the partial trace is wrong, explicitly correct "
        "it and finish the problem."
    )
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    logger.info("takeover.start task=%s handoff_step=%s", task.id, handoff_step)
    raw, trace = await _generate_json_text(client, messages, timeout_sec, "takeover", task.id)
    parsed = _coerce_json_response(raw, StepwiseWorkerOutput)
    if parsed is None and extract_client is not None:
        parsed, extract_trace = await parse_with_llm_fallback(
            raw, StepwiseWorkerOutput, extract_client
        )
        logger.info("takeover.extract_fallback task=%s parsed=%s", task.id, parsed is not None)
        if extract_trace is not None:
            trace = extract_trace
    logger.info("takeover.done task=%s parsed=%s", task.id, parsed is not None)
    return parsed, trace


def _project_remaining_small_cost(total_tokens: int, num_steps: int, handoff_step: int) -> float:
    if num_steps <= 0:
        return 0.0
    avg_per_step = total_tokens / num_steps
    remaining_steps = max(0, num_steps - handoff_step)
    return avg_per_step * remaining_steps


async def _evaluate_task(
    task: StaticTask, cfg: DictConfig, large_model: str
) -> InterventionOutcome:
    small_client = LLMClient(
        model=str(cfg.small_worker.model),
        temperature=float(cfg.small_worker.temperature),
        max_tokens=int(cfg.small_worker.max_tokens),
        reasoning_effort=cfg.small_worker.get("reasoning_effort"),
    )
    large_client = LLMClient(
        model=large_model,
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
    router = ThresholdRouter(
        min_step_confidence=float(cfg.router.min_step_confidence),
        min_drop_intensity=float(cfg.router.min_drop_intensity),
        min_token_density_ratio=float(cfg.router.min_token_density_ratio),
        min_hedge_density=float(cfg.router.min_hedge_density),
        min_semantic_drift=float(cfg.router.min_semantic_drift),
    )
    evaluator = build_static_evaluator(task)

    try:
        (
            small_trace_obj,
            small_trace,
            feature_history,
            decisions,
            handoff_step,
            handoff_trigger,
        ) = await _run_small_iterative(task, small_client, extract_client, cfg, router)
    except TimeoutError:
        logger.error("small_worker.timeout task=%s", task.id)
        return InterventionOutcome(
            task_id=task.id,
            benchmark=task.source,
            small_model=str(cfg.small_worker.model),
            large_model=large_model,
            router_name=router.name,
            success_small_only=False,
            success_after_handoff=False,
            metadata={"error": "small_worker_timeout"},
        )
    except Exception as e:
        logger.error("small_worker.error task=%s err=%s\n%s", task.id, e, traceback.format_exc())
        return InterventionOutcome(
            task_id=task.id,
            benchmark=task.source,
            small_model=str(cfg.small_worker.model),
            large_model=large_model,
            router_name=router.name,
            success_small_only=False,
            success_after_handoff=False,
            metadata={"error": f"small_worker_exception:{type(e).__name__}"},
        )

    small_eval = (
        evaluator(small_trace_obj.final_answer, task)
        if small_trace_obj.final_answer
        else evaluator("", task)
    )
    final_answer = small_trace_obj.final_answer
    success_after_handoff = small_eval.is_correct
    large_parsed = None
    large_trace = None

    if handoff_step is not None:
        try:
            large_parsed, large_trace = await _run_takeover_worker(
                task,
                large_client,
                extract_client,
                small_trace_obj.steps[:handoff_step],
                handoff_step,
                int(cfg.timeouts.large_worker_sec),
            )
        except TimeoutError:
            logger.error("takeover.timeout task=%s handoff_step=%s", task.id, handoff_step)
            large_parsed, large_trace = None, None
        except Exception as e:
            logger.error("takeover.error task=%s err=%s\n%s", task.id, e, traceback.format_exc())
            large_parsed, large_trace = None, None
        if large_parsed is not None:
            final_answer = large_parsed.final_answer
            success_after_handoff = evaluator(final_answer, task).is_correct

    small_pricing = _pricing_from_cfg(cfg, str(cfg.small_worker.model))
    large_pricing = _pricing_from_cfg(cfg, large_model)
    small_cost = trace_to_cost(small_trace, small_pricing)
    large_cost = trace_to_cost(large_trace, large_pricing)
    router_cost = trace_to_cost(None)
    total_cost = combine_costs(small_cost, large_cost, router_cost)

    voi_estimate = None
    voi_realized = None
    if handoff_step is not None and feature_history:
        chosen_features = feature_history[handoff_step - 1]
        projected_small_continue_tokens = _project_remaining_small_cost(
            small_cost.total_tokens, max(1, len(small_trace_obj.steps)), handoff_step
        )
        projected_small_continue_cost = projected_small_continue_tokens / 1000.0
        takeover_cost_est = (
            large_cost.estimated_cost_usd
            if large_cost.estimated_cost_usd is not None
            else large_cost.total_tokens / 1000.0
        )
        continue_cost_est = (
            projected_small_continue_cost
            if small_cost.estimated_cost_usd is None
            else (
                projected_small_continue_tokens
                / 1000.0
                * (small_pricing.output_per_1k if small_pricing is not None else 0.0)
            )
        )
        voi_estimate = estimate_voi(
            p_takeover=float(cfg.router.takeover_success_prior),
            p_continue=chosen_features.current_step_confidence,
            takeover_cost=takeover_cost_est,
            continue_cost=continue_cost_est,
            lambda_cost=float(cfg.router.lambda_cost),
        )
        voi_realized = (
            (1.0 if success_after_handoff else 0.0)
            - float(cfg.router.lambda_cost) * takeover_cost_est
        ) - (
            (1.0 if small_eval.is_correct else 0.0)
            - float(cfg.router.lambda_cost) * continue_cost_est
        )

    return InterventionOutcome(
        task_id=task.id,
        benchmark=task.source,
        small_model=str(cfg.small_worker.model),
        large_model=large_model,
        router_name=router.name,
        handoff_step=handoff_step,
        handoff_trigger=handoff_trigger,
        success_small_only=small_eval.is_correct,
        success_after_handoff=success_after_handoff,
        small_answer=small_trace_obj.final_answer,
        final_answer=final_answer,
        voi_estimate=voi_estimate,
        voi_realized=voi_realized,
        small_cost=small_cost,
        large_cost=large_cost,
        router_cost=router_cost,
        total_cost=total_cost,
        small_trace=small_trace_obj,
        takeover_trace=large_parsed,
        decisions=decisions,
        feature_history=feature_history,
        small_api_trace=small_trace,
        large_api_trace=large_trace,
        metadata={
            "reference_answer": task.reference_answer,
            "evaluator_name": task.evaluator_name,
            "category": task.category,
            "external_difficulty": task.external_difficulty,
        },
    )


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="intervention_pilot",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = str(cfg.dataset.benchmark)
    if benchmark_name == "olympiadbench":
        questions = load_olympiadbench(num_samples=int(cfg.dataset.olympiadbench))
    elif benchmark_name == "livebench_reasoning":
        questions = load_livebench_reasoning(num_samples=int(cfg.dataset.livebench))
    else:
        raise ValueError(f"Unsupported intervention benchmark: {benchmark_name}")

    if cfg.dataset.limit:
        questions = questions[: int(cfg.dataset.limit)]

    logger.info("Loaded %d tasks for intervention pilot", len(questions))

    for large_model in cfg.large_worker.models:
        out_path = output_dir / (
            f"{_sanitize_label(str(cfg.small_worker.label))}_to_"
            f"{_sanitize_label(str(large_model))}.json"
        )
        store = OutcomeStore(out_path)
        logger.info("Running takeover model %s -> %s", cfg.small_worker.model, large_model)

        async def _run_all() -> None:
            for i, task in enumerate(questions, start=1):
                if store.has(task.id):
                    continue
                logger.info("[%s] %d/%d %s", large_model, i, len(questions), task.id)
                outcome = await _evaluate_task(task, cfg, str(large_model))
                logger.info("store.save.start task=%s", task.id)
                store.save(outcome)
                logger.info("store.save.done task=%s", task.id)

        asyncio.run(_run_all())


if __name__ == "__main__":
    main()
