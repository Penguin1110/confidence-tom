from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.logic_bench import load_prontoqa_tasks
from confidence_tom.infra.client import LLMClient
from experiments.mainline.run.core.run_prontoqa_floor_test import (
    FloorExample,
    FloorResult,
    Recipe,
    _build_recipe_step,
    _extract_floor_material,
    _extract_subject_facts,
    _run_example_with_retry,
)

RunMode = Literal["explicit", "continue"]


class SelfCotSubstrate(BaseModel):
    task_id: str
    recipe: str
    subject: str
    goal_predicate: str
    alternate_predicate: str
    branch_predicates: list[str] = Field(default_factory=list)
    question: str
    self_generated_steps: list[str]
    final_answer: str
    raw_generation: str
    generation_trace: dict[str, Any] = Field(default_factory=dict)


class SelfCotPilotState(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    substrates: list[SelfCotSubstrate] = Field(default_factory=list)
    results: list[FloorResult] = Field(default_factory=list)


_SELF_COT_SYSTEM_PROMPT = """You are a concise logic proof solver.
Produce your own proof. Do not quote any hidden or dataset solution.
Use numbered steps and end with exactly one final-answer line.
"""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_final_answer(text: str) -> str:
    matches = re.findall(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else ""


def _answer_matches(task: StaticTask, text: str) -> bool:
    final = _normalize(_extract_final_answer(text))
    reference = _normalize(task.reference_answer)
    return bool(final and reference and (reference in final or final in reference))


def _goal_is_direct_fact(task: StaticTask) -> bool:
    try:
        subject, goal_predicate = _parse_goal_for_direct_fact(task.reference_answer)
    except ValueError:
        return False
    return any(goal_predicate in fact_group for fact_group in _extract_subject_facts(task, subject))


def _parse_goal_for_direct_fact(reference_answer: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][a-z]+)\s+is\s+(?:a|an)\s+([a-z]+)\.?$", reference_answer.strip())
    if not match:
        raise ValueError(f"Could not parse goal: {reference_answer}")
    return match.group(1), match.group(2).lower()


def _parse_numbered_steps(text: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?m)^\s*(?:Step\s*)?(\d+)[\).:\-]\s+(.*?)(?=^\s*(?:Step\s*)?\d+[\).:\-]\s+|\Z)",
            text,
            flags=re.DOTALL,
        )
    )
    steps: list[str] = []
    for match in matches:
        step = match.group(2).strip()
        step = re.sub(r"\n\s+", " ", step)
        step = re.sub(r"\s+", " ", step)
        if step and not step.lower().startswith("final answer"):
            steps.append(step.rstrip())
    if len(steps) >= 2:
        return steps

    fallback: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*[-*]\s+", "", line).strip()
        if not cleaned or cleaned.lower().startswith("final answer"):
            continue
        if any(marker in cleaned.lower() for marker in ["therefore", "given", "since", "from "]):
            fallback.append(cleaned)
    return fallback


def _build_self_cot_prompt(task: StaticTask) -> str:
    return (
        f"Question:\n{task.question}\n\n"
        "Write a concise proof in 3 to 6 numbered steps.\n"
        "Start directly with Step 1; do not include introspective preamble.\n"
        f"End with: Final Answer: {task.reference_answer}"
    )


async def _generate_self_cot(
    client: LLMClient,
    task: StaticTask,
    *,
    timeout_sec: float,
    retries: int,
    generation_max_tokens: int,
) -> tuple[list[str], str, str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": _SELF_COT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_self_cot_prompt(task)},
    ]
    attempts = retries + 1
    last_trace: dict[str, Any] = {}
    last_raw = ""
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            raw, trace = await asyncio.wait_for(
                client.agenerate_text_with_trace(
                    messages,
                    max_tokens=generation_max_tokens,
                    temperature=0.0,
                ),
                timeout=timeout_sec,
            )
            last_raw = raw
            last_trace = trace.model_dump() if hasattr(trace, "model_dump") else {}
            last_trace.setdefault("runtime", {})
            last_trace["runtime"].update(
                {
                    "stage": "self_cot_generation",
                    "attempt": attempt,
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "timeout_sec": timeout_sec,
                }
            )
            visible = str(last_trace.get("response_content") or "")
            # For the self-CoT substrate, we only admit visible proof text. Exposed
            # reasoning is useful metadata, but editing hidden-only reasoning would
            # change the task into a different intervention.
            if not visible.strip() and raw.strip() and not last_trace.get("reasoning_content"):
                visible = raw
            steps = _parse_numbered_steps(visible)
            final = _extract_final_answer(visible)
            return steps, final, visible, last_trace
        except asyncio.TimeoutError:
            last_trace = {
                "runtime": {"stage": "self_cot_generation", "attempt": attempt, "error": "timeout"}
            }
        except Exception as exc:  # noqa: BLE001 - keep pilot batches resumable.
            last_trace = {
                "runtime": {
                    "stage": "self_cot_generation",
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            }
        if attempt < attempts:
            await asyncio.sleep(min(2 * attempt, 8))
    return [], "", last_raw, last_trace


def _to_floor_example(substrate: SelfCotSubstrate) -> FloorExample:
    edited_step, recipe = _build_recipe_step(
        subject=substrate.subject,
        goal_predicate=substrate.goal_predicate,
        alternate_predicate=substrate.alternate_predicate,
        branch_predicates=substrate.branch_predicates,
        recipe=substrate.recipe,  # type: ignore[arg-type]
    )
    return FloorExample(
        task_id=substrate.task_id,
        recipe=recipe,
        subject=substrate.subject,
        goal_predicate=substrate.goal_predicate,
        alternate_predicate=substrate.alternate_predicate,
        branch_predicates=substrate.branch_predicates,
        question=substrate.question,
        original_steps=substrate.self_generated_steps,
        edited_step=edited_step,
    )


def _load_state(path: Path) -> SelfCotPilotState:
    if not path.exists():
        return SelfCotPilotState()
    try:
        return SelfCotPilotState.model_validate(json.loads(path.read_text()))
    except Exception:
        return SelfCotPilotState()


def _write_state(path: Path, state: SelfCotPilotState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.model_dump(), ensure_ascii=False, indent=2))


async def _build_substrates(
    *,
    args: argparse.Namespace,
    tasks: list[StaticTask],
    client: LLMClient,
    state: SelfCotPilotState,
) -> list[SelfCotSubstrate]:
    recipes: list[Recipe] = ["branch_follow", "goal_deferral", "focus_shift"]
    counts = Counter(substrate.recipe for substrate in state.substrates)
    seen = {(substrate.task_id, substrate.recipe) for substrate in state.substrates}
    task_by_id = {task.id: task for task in tasks}

    for task in tasks:
        if all(counts[recipe] >= args.per_recipe for recipe in recipes):
            break
        if _goal_is_direct_fact(task):
            continue
        try:
            material = _extract_floor_material(task)
        except ValueError:
            continue
        if material is None:
            continue
        subject, goal_predicate, _dataset_steps, alternate_predicate, branch_predicates = material
        for recipe in recipes:
            if counts[recipe] >= args.per_recipe or (task.id, recipe) in seen:
                continue
            steps, final_answer, raw, trace = await _generate_self_cot(
                client,
                task,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                generation_max_tokens=args.generation_max_tokens,
            )
            if len(steps) < 3 or not _answer_matches(task, raw):
                continue
            substrate = SelfCotSubstrate(
                task_id=task.id,
                recipe=recipe,
                subject=subject,
                goal_predicate=goal_predicate,
                alternate_predicate=alternate_predicate,
                branch_predicates=branch_predicates,
                question=task.question,
                self_generated_steps=steps[: args.max_steps],
                final_answer=final_answer,
                raw_generation=raw,
                generation_trace=trace,
            )
            state.substrates.append(substrate)
            counts[recipe] += 1
            seen.add((task.id, recipe))
            _write_state(Path(args.output), state)
            break

    # Drop substrates whose source task is no longer in the sampled task map.
    return [substrate for substrate in state.substrates if substrate.task_id in task_by_id]


def _summarize(args: argparse.Namespace, state: SelfCotPilotState) -> dict[str, Any]:
    counts = Counter(result.label for result in state.results)
    by_recipe: dict[str, Any] = {}
    for recipe in ["branch_follow", "goal_deferral", "focus_shift"]:
        rows = [result for result in state.results if result.recipe == recipe]
        recipe_counts = Counter(result.label for result in rows)
        by_recipe[recipe] = {
            "n": len(rows),
            "counts": dict(recipe_counts),
            "explicit_report_rate": recipe_counts["EXPLICIT_REPORT"] / len(rows) if rows else 0.0,
            "silent_fix_rate": recipe_counts["SILENT_FIX"] / len(rows) if rows else 0.0,
            "follow_rate": recipe_counts["FOLLOW"] / len(rows) if rows else 0.0,
            "other_rate": recipe_counts["OTHER"] / len(rows) if rows else 0.0,
            "path_inconsistent_share": recipe_counts["PATH_INCONSISTENT"] / len(rows)
            if rows
            else 0.0,
        }
    return {
        "config": state.config,
        "substrate_count": len(state.substrates),
        "result_count": len(state.results),
        "counts": dict(counts),
        "by_recipe": by_recipe,
    }


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    state = _load_state(output) if args.resume else SelfCotPilotState()
    state.config = {
        "benchmark": "prontoqa_self_cot_pilot",
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "mode": args.mode,
        "per_recipe": args.per_recipe,
        "seed": args.seed,
        "max_records": args.max_records,
        "task_sample_size": args.task_sample_size,
        "generation_max_tokens": args.generation_max_tokens,
        "timeout_sec": args.timeout_sec,
        "retries": args.retries,
        "reasoning_max_tokens": args.reasoning_max_tokens,
    }
    tasks = load_prontoqa_tasks(
        num_samples=max(args.task_sample_size, args.per_recipe * 9, 30),
        seed=args.seed,
        max_records=args.max_records,
    )
    target_client = LLMClient(
        model=args.target_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=args.generation_max_tokens,
        reasoning_enabled=True,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_exclude=False,
    )
    judge_client = LLMClient(
        model=args.judge_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=350,
        enable_thinking=False,
        reasoning_enabled=False,
        reasoning_exclude=True,
    )
    substrates = await _build_substrates(args=args, tasks=tasks, client=target_client, state=state)
    examples = [_to_floor_example(substrate) for substrate in substrates]
    existing = {(result.task_id, result.recipe, result.mode) for result in state.results}
    for example in examples:
        key = (example.task_id, example.recipe, args.mode)
        if key in existing:
            continue
        if args.mode == "continue":
            result = await _run_example_with_retry(
                mode="continue",
                client=target_client,
                judge_client=judge_client,
                example=example,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                variant="inconsistent",
            )
        else:
            result = await _run_example_with_retry(
                mode="explicit",
                client=target_client,
                judge_client=judge_client,
                example=example,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                variant="inconsistent",
            )
        state.results.append(result)
        existing.add(key)
        _write_state(output, state)
    _write_state(output, state)
    return _summarize(args, state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a ProntoQA B-pilot using target-model self-generated proof traces."
    )
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--mode", choices=["explicit", "continue"], default="continue")
    parser.add_argument("--target-model", default="qwen/qwen3-14b")
    parser.add_argument("--judge-model", default="qwen/qwen3-14b")
    parser.add_argument("--per-recipe", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--task-sample-size", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--generation-max-tokens", type=int, default=900)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--reasoning-max-tokens", type=int, default=160)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output",
        default="outputs/results/cot_edit_introspection_pilot/prontoqa_self_cot_b_qwen3_14b.json",
    )
    args = parser.parse_args()
    result = asyncio.run(_main_async(args))
    print(f"[saved] {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
