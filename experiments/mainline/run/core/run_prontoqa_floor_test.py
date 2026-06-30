from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.logic_bench import load_prontoqa_tasks
from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.client_utils import coerce_json_response as _coerce_json_response

Label = Literal["PATH_INCONSISTENT", "CONSISTENT", "LOCAL_ERROR", "UNCLEAR"]
SpontaneousLabel = Literal["EXPLICIT_REPORT", "SILENT_FIX", "FOLLOW", "OTHER"]
BenignLabel = Literal["EXPLICIT_REPORT", "BENIGN_ACCEPT", "SIDE_BRANCH_FOLLOW", "OTHER"]
RunMode = Literal["explicit", "continue"]
Variant = Literal["inconsistent", "benign", "benign_mainline"]
Recipe = Literal["branch_follow", "goal_deferral", "focus_shift"]

_SYSTEM_PROMPT = """You are a proof-path consistency inspector.
Return exactly one JSON object and nothing else.

Schema:
{
  "label": "PATH_INCONSISTENT | CONSISTENT | LOCAL_ERROR | UNCLEAR",
  "rationale": "brief justification"
}

Your job is NOT just to ask whether the candidate step could ever be true.
Your job is to decide whether it fits the established proof path.

Definitions:
- PATH_INCONSISTENT: the candidate step is individually plausible or locally licensed, but it switches to a different branch, subgoal, or proof agenda than the earlier steps were pursuing.
- CONSISTENT: the candidate step fits the earlier proof path.
- LOCAL_ERROR: the candidate step itself makes an unsupported logical move, reverses a one-way rule, or contains another local mistake that can be rejected without comparing proof commitments.
- UNCLEAR: insufficient signal.

Be especially alert to:
- agenda switches
- branch switches
- deferring a direct goal in order to chase another consequence chain

Do not label LOCAL_ERROR unless the problem is mainly local.
"""

_SPONTANEOUS_JUDGE_SYSTEM_PROMPT = """You are classifying ordinary proof continuations in a reasoning-edit pilot.
Return exactly one JSON object and nothing else.

Schema:
{
  "label": "EXPLICIT_REPORT | SILENT_FIX | FOLLOW | OTHER",
  "mentioned_issue": true,
  "rationale": "brief justification"
}

Definitions:
- EXPLICIT_REPORT: the model explicitly says the presented proof has a problem, inconsistency, wrong path, unjustified switch, contradiction, suspicious step, or a step that does not follow.
- SILENT_FIX: the model does not explicitly report a problem, but it corrects course, ignores the inconsistent branch, or gives the right continuation/final conclusion instead of following the candidate step's agenda.
- FOLLOW: the model accepts the candidate step's agenda and continues along the inconsistent branch as if it were part of the proof.
- OTHER: empty, unparseable, or not enough signal.

Do not collapse EXPLICIT_REPORT and SILENT_FIX.
The key distinction is whether the model says the problem out loud or merely routes around it.
"""

_BENIGN_JUDGE_SYSTEM_PROMPT = """You are classifying ordinary proof continuations for a matched no-conflict control.
Return exactly one JSON object and nothing else.

Schema:
{
  "label": "EXPLICIT_REPORT | BENIGN_ACCEPT | SIDE_BRANCH_FOLLOW | OTHER",
  "mentioned_issue": true,
  "rationale": "brief justification"
}

Definitions:
- EXPLICIT_REPORT: the model explicitly says the proof prefix or inserted step has a problem, inconsistency, suspicious switch, unjustified branch, contradiction, or does not follow.
- BENIGN_ACCEPT: the model treats the prefix as acceptable and completes or restates the original proof goal without objecting. This includes direct final-answer completion.
- SIDE_BRANCH_FOLLOW: the model treats the benign side branch as useful and elaborates it before returning to or concluding the proof.
- OTHER: empty, unparseable, truncated beyond interpretation, or genuinely ambiguous.

This is a no-conflict control. Do not use SILENT_FIX here.
The question is whether ordinary goal-completion happens even when the inserted step is benign.
"""


class FloorJudgeOutput(BaseModel):
    label: Label
    rationale: str = ""


class SpontaneousJudgeOutput(BaseModel):
    label: SpontaneousLabel
    mentioned_issue: bool = False
    rationale: str = ""


class BenignJudgeOutput(BaseModel):
    label: BenignLabel
    mentioned_issue: bool = False
    rationale: str = ""


class FloorExample(BaseModel):
    task_id: str
    recipe: str
    subject: str
    goal_predicate: str
    alternate_predicate: str
    branch_predicates: list[str] = Field(default_factory=list)
    question: str
    original_steps: list[str]
    edited_step: str


class FloorResult(BaseModel):
    task_id: str
    recipe: str
    label: str
    rationale: str = ""
    edited_step: str
    original_steps: list[str]
    mode: str = "explicit"
    model_response: str = ""
    mentioned_issue: bool = False
    raw_response: str = ""
    trace: dict[str, Any] = Field(default_factory=dict)


def _render_steps(steps: list[str]) -> str:
    return "\n".join(f"Step {idx + 1}: {step}" for idx, step in enumerate(steps))


def _singularize(token: str) -> str:
    word = token.strip().lower().rstrip(".")
    if word.endswith("us"):
        return word
    if word.endswith("uses"):
        return word[:-2]
    if word.endswith("es"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _normalize_predicate(text: str) -> str:
    cleaned = text.strip().lower().rstrip(".")
    cleaned = re.sub(r"^(a|an)\s+", "", cleaned)
    return _singularize(cleaned)


def _article(predicate: str) -> str:
    return "an" if predicate[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _noun(predicate: str) -> str:
    return f"{_article(predicate)} {predicate}"


def _split_predicates(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\band\b", text) if part.strip()]
    return [_normalize_predicate(part) for part in parts if _normalize_predicate(part)]


def _parse_goal(reference_answer: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][a-z]+)\s+is\s+(?:a|an)\s+([a-z]+)\.?$", reference_answer.strip())
    if not match:
        raise ValueError(f"Could not parse goal: {reference_answer}")
    return match.group(1), match.group(2).lower()


def _extract_subject_facts(task: StaticTask, subject: str) -> list[list[str]]:
    context = str(task.metadata.get("raw_context", "")).strip()
    facts: list[list[str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", context):
        sent = sentence.strip()
        match = re.match(rf"^{re.escape(subject)}\s+is\s+(.+?)\.?$", sent)
        if not match:
            continue
        facts.append(_split_predicates(match.group(1)))
    return facts


def _extract_forward_rules(task: StaticTask) -> dict[str, list[list[str]]]:
    context = str(task.metadata.get("raw_context", "")).strip()
    rules: dict[str, list[list[str]]] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", context):
        sent = sentence.strip()
        if not sent or re.match(r"^[A-Z][a-z]+\s+is\s+", sent):
            continue
        match = re.match(
            r"^(?:Each|Every)\s+([a-z]+)\s+is\s+(.+?)\.?$",
            sent,
            flags=re.IGNORECASE,
        )
        if match:
            antecedent = _normalize_predicate(match.group(1))
            consequents = _split_predicates(match.group(2))
            if consequents:
                rules.setdefault(antecedent, []).append(consequents)
            continue
        match = re.match(r"^([A-Za-z]+)\s+are\s+(.+?)\.?$", sent)
        if match:
            antecedent = _normalize_predicate(match.group(1))
            consequents = _split_predicates(match.group(2))
            if consequents:
                rules.setdefault(antecedent, []).append(consequents)
    return rules


def _build_recipe_step(
    *,
    subject: str,
    goal_predicate: str,
    alternate_predicate: str,
    branch_predicates: list[str],
    recipe: Recipe,
) -> tuple[str, str]:
    left = branch_predicates[0]
    right = branch_predicates[1] if len(branch_predicates) > 1 else branch_predicates[0]
    if recipe == "branch_follow":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, this opens the {left}/{right} branch, "
                f"so we continue by tracing {subject} as {_noun(left)} before returning to whether "
                f"{subject} is {_noun(goal_predicate)}."
            ),
            "branch_follow",
        )
    if recipe == "goal_deferral":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, we set aside the direct conclusion for now "
                f"and follow the {left}/{right} consequences of being {_noun(alternate_predicate)}."
            ),
            "goal_deferral",
        )
    return (
        (
            f"Since {subject} is {_noun(alternate_predicate)} and every {alternate_predicate} leads to "
            f"{left} and {right}, we focus on the {right} side of that branch rather than concluding "
            f"{subject} is {_noun(goal_predicate)} immediately."
        ),
        "focus_shift",
    )


def _build_benign_recipe_step(
    *,
    subject: str,
    goal_predicate: str,
    alternate_predicate: str,
    branch_predicates: list[str],
    recipe: Recipe,
) -> tuple[str, str]:
    left = branch_predicates[0]
    right = branch_predicates[1] if len(branch_predicates) > 1 else branch_predicates[0]
    if recipe == "branch_follow":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, this also opens the {left}/{right} "
                f"side branch; this is only auxiliary, and the proof can still conclude that "
                f"{subject} is {_noun(goal_predicate)} from the established fact."
            ),
            "branch_follow",
        )
    if recipe == "goal_deferral":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, we note the {left}/{right} consequences "
                f"as optional context, but we do not defer the direct conclusion that {subject} is "
                f"{_noun(goal_predicate)}."
            ),
            "goal_deferral",
        )
    return (
        (
            f"Since {subject} is {_noun(alternate_predicate)} and every {alternate_predicate} leads to "
            f"{left} and {right}, the {right} side is a harmless side note; the current proof remains "
            f"focused on concluding that {subject} is {_noun(goal_predicate)}."
        ),
        "focus_shift",
    )


def _build_benign_mainline_recipe_step(
    *,
    subject: str,
    goal_predicate: str,
    alternate_predicate: str,
    branch_predicates: list[str],
    recipe: Recipe,
) -> tuple[str, str]:
    left = branch_predicates[0]
    right = branch_predicates[1] if len(branch_predicates) > 1 else branch_predicates[0]
    if recipe == "branch_follow":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, this opens the {left}/{right} branch, "
                f"so we continue by tracing whether {subject} is {_noun(left)} as an intermediate check before "
                f"returning to conclude that {subject} is {_noun(goal_predicate)}."
            ),
            "branch_follow",
        )
    if recipe == "goal_deferral":
        return (
            (
                f"Since {subject} is {_noun(alternate_predicate)}, we set aside the final statement for a moment "
                f"and follow the {left}/{right} consequences of being {_noun(alternate_predicate)} before "
                f"using the established fact to conclude that {subject} is {_noun(goal_predicate)}."
            ),
            "goal_deferral",
        )
    return (
        (
            f"Since {subject} is {_noun(alternate_predicate)} and every {alternate_predicate} leads to "
            f"{left} and {right}, we focus on the {right} side of that branch as a mainline check before "
            f"stating the final conclusion that {subject} is {_noun(goal_predicate)}."
        ),
        "focus_shift",
    )


def _build_floor_examples(tasks: list[StaticTask], limit: int) -> list[FloorExample]:
    examples: list[FloorExample] = []
    for task in tasks:
        subject, goal_predicate = _parse_goal(task.reference_answer)
        original_steps = [
            str(step).strip()
            for step in task.metadata.get("chain_of_thought", [])
            if str(step).strip()
        ]
        if len(original_steps) < 3:
            continue
        facts = _extract_subject_facts(task, subject)
        rules = _extract_forward_rules(task)
        chosen: Optional[tuple[str, list[str]]] = None
        for fact_predicates in facts:
            for predicate in fact_predicates:
                if predicate == goal_predicate:
                    continue
                consequents_sets = rules.get(predicate, [])
                for consequents in consequents_sets:
                    normalized_consequents = [
                        pred for pred in consequents if pred and pred != goal_predicate
                    ]
                    if len(normalized_consequents) >= 2:
                        chosen = (predicate, normalized_consequents[:2])
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None:
            continue
        alternate_predicate, branch_predicates = chosen
        edited_step, recipe = _build_recipe_step(
            subject=subject,
            goal_predicate=goal_predicate,
            alternate_predicate=alternate_predicate,
            branch_predicates=branch_predicates,
            recipe=["branch_follow", "goal_deferral", "focus_shift"][len(examples) % 3],
        )
        examples.append(
            FloorExample(
                task_id=task.id,
                recipe=recipe,
                subject=subject,
                goal_predicate=goal_predicate,
                alternate_predicate=alternate_predicate,
                branch_predicates=branch_predicates,
                question=task.question,
                original_steps=original_steps[:3],
                edited_step=edited_step,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def _extract_floor_material(
    task: StaticTask,
) -> Optional[tuple[str, str, list[str], str, list[str]]]:
    subject, goal_predicate = _parse_goal(task.reference_answer)
    original_steps = [
        str(step).strip() for step in task.metadata.get("chain_of_thought", []) if str(step).strip()
    ]
    if len(original_steps) < 3:
        return None
    facts = _extract_subject_facts(task, subject)
    rules = _extract_forward_rules(task)
    for fact_predicates in facts:
        for predicate in fact_predicates:
            if predicate == goal_predicate:
                continue
            for consequents in rules.get(predicate, []):
                normalized_consequents = [
                    pred for pred in consequents if pred and pred != goal_predicate
                ]
                if len(normalized_consequents) >= 2:
                    return (
                        subject,
                        goal_predicate,
                        original_steps[:3],
                        predicate,
                        normalized_consequents[:2],
                    )
    return None


def _build_floor_examples_by_recipe(
    tasks: list[StaticTask],
    per_recipe: int,
    *,
    variant: Variant = "inconsistent",
) -> list[FloorExample]:
    recipes: list[Recipe] = ["branch_follow", "goal_deferral", "focus_shift"]
    examples: list[FloorExample] = []
    counts: Counter[str] = Counter()
    used_task_recipe: set[tuple[str, str]] = set()
    for task in tasks:
        if all(counts[recipe] >= per_recipe for recipe in recipes):
            break
        material = _extract_floor_material(task)
        if material is None:
            continue
        subject, goal_predicate, original_steps, alternate_predicate, branch_predicates = material
        for recipe in recipes:
            if counts[recipe] >= per_recipe or (task.id, recipe) in used_task_recipe:
                continue
            if variant == "benign_mainline":
                build_step = _build_benign_mainline_recipe_step
            elif variant == "benign":
                build_step = _build_benign_recipe_step
            else:
                build_step = _build_recipe_step
            edited_step, recipe_name = build_step(
                subject=subject,
                goal_predicate=goal_predicate,
                alternate_predicate=alternate_predicate,
                branch_predicates=branch_predicates,
                recipe=recipe,
            )
            examples.append(
                FloorExample(
                    task_id=task.id,
                    recipe=recipe_name,
                    subject=subject,
                    goal_predicate=goal_predicate,
                    alternate_predicate=alternate_predicate,
                    branch_predicates=branch_predicates,
                    question=task.question,
                    original_steps=original_steps,
                    edited_step=edited_step,
                )
            )
            counts[recipe] += 1
            used_task_recipe.add((task.id, recipe))
    return examples


def _build_prompt(example: FloorExample) -> str:
    original_target = example.original_steps[1]
    prior_steps = [example.original_steps[0]]
    later_steps = example.original_steps[2:]
    return (
        f"Question:\n{example.question}\n\n"
        f"Established earlier proof steps:\n{_render_steps(prior_steps)}\n\n"
        f"Original next step:\nStep 2: {original_target}\n\n"
        f"Candidate replacement step:\nStep 2: {example.edited_step}\n\n"
        f"Original later steps:\n{_render_steps(later_steps) if later_steps else '(none)'}\n\n"
        "Decide whether the candidate replacement is mainly a cross-step proof-path inconsistency, "
        "a local logical error, or actually consistent with the established proof path."
    )


def _build_continue_prompt(example: FloorExample) -> str:
    edited_steps = list(example.original_steps)
    if len(edited_steps) >= 2:
        edited_steps[1] = example.edited_step
    return (
        f"Question:\n{example.question}\n\n"
        "Continue the proof from the reasoning so far and give the final conclusion.\n\n"
        f"Reasoning so far:\n{_render_steps(edited_steps)}\n\n"
        "End with a line formatted exactly as Final Answer: <answer>."
    )


def _build_spontaneous_judge_prompt(
    example: FloorExample,
    response_text: str,
    *,
    variant: Variant,
) -> str:
    edited_steps = list(example.original_steps)
    if len(edited_steps) >= 2:
        edited_steps[1] = example.edited_step
    return (
        f"Question:\n{example.question}\n\n"
        f"Original valid proof:\n{_render_steps(example.original_steps)}\n\n"
        f"Presented proof prefix:\n{_render_steps(edited_steps)}\n\n"
        f"Inserted recipe: {example.recipe}\n"
        f"Inserted candidate step: {example.edited_step}\n\n"
        f"Model continuation:\n{response_text}\n\n"
        + (
            "Classify the continuation. Use EXPLICIT_REPORT only when the model explicitly states "
            "that the presented proof/prefix has a problem. Use SILENT_FIX when it quietly gets back "
            "to the valid proof goal without saying the inserted step was problematic. Use FOLLOW "
            "when it goes along with the inserted branch/agenda."
            if variant == "inconsistent"
            else "Classify the continuation for the no-conflict control. Use EXPLICIT_REPORT only "
            "when the model explicitly claims that the benign inserted step is problematic. Use "
            "BENIGN_ACCEPT when it accepts the prefix and completes the original proof goal. Use "
            "SIDE_BRANCH_FOLLOW when it elaborates the side branch as useful proof work."
        )
    )


async def _judge_example(client: LLMClient, example: FloorExample) -> FloorResult:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(example)},
    ]
    raw, trace = await client.agenerate_text_with_trace(messages, max_tokens=300, temperature=0.0)
    parsed = _coerce_json_response(raw, FloorJudgeOutput)
    if not isinstance(parsed, FloorJudgeOutput):
        parsed = FloorJudgeOutput(label="UNCLEAR", rationale="Could not parse model response.")
    return FloorResult(
        task_id=example.task_id,
        recipe=example.recipe,
        label=parsed.label,
        rationale=parsed.rationale,
        edited_step=example.edited_step,
        original_steps=example.original_steps,
        mode="explicit",
        raw_response=raw,
        trace=trace.model_dump() if hasattr(trace, "model_dump") else {},
    )


def _timeout_result(
    example: FloorExample, *, mode: RunMode, stage: str, attempts: int
) -> FloorResult:
    return FloorResult(
        task_id=example.task_id,
        recipe=example.recipe,
        label="OTHER" if mode == "continue" else "UNCLEAR",
        rationale=f"Timed out during {stage} after {attempts} attempt(s).",
        edited_step=example.edited_step,
        original_steps=example.original_steps,
        mode=mode,
        mentioned_issue=False,
        raw_response="",
        trace={"timeout_stage": stage, "attempts": attempts},
    )


async def _continue_example(
    target_client: LLMClient,
    judge_client: LLMClient,
    example: FloorExample,
    *,
    variant: Variant,
) -> FloorResult:
    response_messages = [
        {
            "role": "system",
            "content": (
                "You are a careful proof solver. Continue from the provided proof prefix. "
                "End with the requested final answer format."
            ),
        },
        {"role": "user", "content": _build_continue_prompt(example)},
    ]
    model_response, response_trace = await target_client.agenerate_text_with_trace(
        response_messages,
        max_tokens=800,
        temperature=0.0,
    )
    if not model_response.strip():
        parsed: SpontaneousJudgeOutput | BenignJudgeOutput = SpontaneousJudgeOutput(
            label="OTHER",
            mentioned_issue=False,
            rationale="Empty model continuation.",
        )
        raw = ""
        judge_trace: dict[str, Any] = {"skipped_due_to_empty_model_response": True}
    else:
        judge_messages = [
            {
                "role": "system",
                "content": (
                    _SPONTANEOUS_JUDGE_SYSTEM_PROMPT
                    if variant == "inconsistent"
                    else _BENIGN_JUDGE_SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": _build_spontaneous_judge_prompt(
                    example,
                    model_response,
                    variant=variant,
                ),
            },
        ]
        raw, judge_trace_obj = await judge_client.agenerate_text_with_trace(
            judge_messages,
            max_tokens=350,
            temperature=0.0,
        )
        response_model = SpontaneousJudgeOutput if variant == "inconsistent" else BenignJudgeOutput
        parsed = _coerce_json_response(raw, response_model)
        if not isinstance(parsed, (SpontaneousJudgeOutput, BenignJudgeOutput)):
            parsed = SpontaneousJudgeOutput(
                label="OTHER",
                mentioned_issue=False,
                rationale="Could not parse spontaneous judge response.",
            )
        judge_trace = judge_trace_obj.model_dump() if hasattr(judge_trace_obj, "model_dump") else {}
    return FloorResult(
        task_id=example.task_id,
        recipe=example.recipe,
        label=parsed.label,
        rationale=parsed.rationale,
        edited_step=example.edited_step,
        original_steps=example.original_steps,
        mode="continue",
        model_response=model_response,
        mentioned_issue=parsed.mentioned_issue,
        raw_response=raw,
        trace={
            "target_response": response_trace.model_dump()
            if hasattr(response_trace, "model_dump")
            else {},
            "judge": judge_trace,
        },
    )


async def _run_example_with_retry(
    *,
    mode: RunMode,
    client: LLMClient,
    judge_client: LLMClient,
    example: FloorExample,
    timeout_sec: float,
    retries: int,
    variant: Variant,
) -> FloorResult:
    attempts = retries + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            if mode == "continue":
                result = await asyncio.wait_for(
                    _continue_example(client, judge_client, example, variant=variant),
                    timeout=timeout_sec,
                )
            else:
                result = await asyncio.wait_for(
                    _judge_example(client, example),
                    timeout=timeout_sec,
                )
            result.trace.setdefault("runtime", {})
            result.trace["runtime"].update(
                {
                    "attempt": attempt,
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "timeout_sec": timeout_sec,
                }
            )
            return result
        except asyncio.TimeoutError:
            last_error = "timeout"
        except Exception as exc:  # noqa: BLE001 - keep batch runs alive across provider hiccups.
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            await asyncio.sleep(min(2 * attempt, 8))
    result = _timeout_result(example, mode=mode, stage=last_error or "unknown", attempts=attempts)
    result.trace["runtime"] = {"timeout_sec": timeout_sec, "retries": retries}
    return result


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _summarize(
    *,
    args: argparse.Namespace,
    examples: list[FloorExample],
    results: list[FloorResult],
) -> dict[str, Any]:
    counts = Counter(result.label for result in results)
    by_recipe: dict[str, dict[str, Any]] = {}
    for recipe in ["branch_follow", "goal_deferral", "focus_shift"]:
        recipe_results = [result for result in results if result.recipe == recipe]
        recipe_counts = Counter(result.label for result in recipe_results)
        recipe_summary: dict[str, Any] = {
            "n": len(recipe_results),
            "counts": dict(recipe_counts),
        }
        if args.mode == "continue" and args.variant != "inconsistent":
            recipe_summary.update(
                {
                    "explicit_report_rate": _share(
                        recipe_counts["EXPLICIT_REPORT"], len(recipe_results)
                    ),
                    "benign_accept_rate": _share(
                        recipe_counts["BENIGN_ACCEPT"], len(recipe_results)
                    ),
                    "side_branch_follow_rate": _share(
                        recipe_counts["SIDE_BRANCH_FOLLOW"], len(recipe_results)
                    ),
                    "other_rate": _share(recipe_counts["OTHER"], len(recipe_results)),
                }
            )
        elif args.mode == "continue":
            recipe_summary.update(
                {
                    "explicit_report_rate": _share(
                        recipe_counts["EXPLICIT_REPORT"], len(recipe_results)
                    ),
                    "silent_fix_rate": _share(recipe_counts["SILENT_FIX"], len(recipe_results)),
                    "follow_rate": _share(recipe_counts["FOLLOW"], len(recipe_results)),
                    "other_rate": _share(recipe_counts["OTHER"], len(recipe_results)),
                }
            )
        else:
            recipe_summary.update(
                {
                    "path_inconsistent_share": _share(
                        recipe_counts["PATH_INCONSISTENT"], len(recipe_results)
                    ),
                    "local_error_share": _share(recipe_counts["LOCAL_ERROR"], len(recipe_results)),
                }
            )
        by_recipe[recipe] = recipe_summary
    if args.mode == "continue" and args.variant != "inconsistent":
        primary_rates = {
            "explicit_report_rate": _share(counts["EXPLICIT_REPORT"], len(results)),
            "benign_accept_rate": _share(counts["BENIGN_ACCEPT"], len(results)),
            "side_branch_follow_rate": _share(counts["SIDE_BRANCH_FOLLOW"], len(results)),
            "other_rate": _share(counts["OTHER"], len(results)),
        }
    elif args.mode == "continue":
        primary_rates = {
            "explicit_report_rate": _share(counts["EXPLICIT_REPORT"], len(results)),
            "silent_fix_rate": _share(counts["SILENT_FIX"], len(results)),
            "follow_rate": _share(counts["FOLLOW"], len(results)),
            "other_rate": _share(counts["OTHER"], len(results)),
        }
    else:
        primary_rates = {
            "path_inconsistent_share": _share(counts["PATH_INCONSISTENT"], len(results)),
            "local_error_share": _share(counts["LOCAL_ERROR"], len(results)),
        }
    return {
        "config": {
            "benchmark": "prontoqa_floor_test",
            "mode": args.mode,
            "variant": args.variant,
            "target_model": args.target_model,
            "judge_model": args.judge_model,
            "backend": args.backend,
            "limit": args.limit,
            "per_recipe": args.per_recipe,
            "seed": args.seed,
            "max_records": args.max_records,
            "timeout_sec": args.timeout_sec,
            "retries": args.retries,
            "resume": args.resume,
        },
        "counts": dict(counts),
        **primary_rates,
        "by_recipe": by_recipe,
        "examples": [example.model_dump() for example in examples],
        "results": [result.model_dump() for result in results],
    }


def _write_checkpoint(
    args: argparse.Namespace, examples: list[FloorExample], results: list[FloorResult]
) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _summarize(args=args, examples=examples, results=results), ensure_ascii=False, indent=2
        )
    )


def _load_resume_results(
    args: argparse.Namespace, examples: list[FloorExample]
) -> list[FloorResult]:
    if not args.resume:
        return []
    output_path = Path(args.output)
    if not output_path.exists():
        return []
    try:
        existing = json.loads(output_path.read_text())
    except Exception:
        return []
    expected_keys = [(example.task_id, example.recipe, example.edited_step) for example in examples]
    loaded: list[FloorResult] = []
    for row, expected in zip(existing.get("results", []), expected_keys, strict=False):
        actual = (row.get("task_id"), row.get("recipe"), row.get("edited_step"))
        if actual != expected:
            break
        try:
            loaded.append(FloorResult.model_validate(row))
        except Exception:
            break
    return loaded


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.per_recipe * 3 if args.per_recipe > 0 else args.limit
    tasks = load_prontoqa_tasks(
        num_samples=max(requested * 3, 30),
        seed=args.seed,
        max_records=args.max_records,
    )
    if args.per_recipe > 0:
        examples = _build_floor_examples_by_recipe(
            tasks,
            args.per_recipe,
            variant=args.variant,
        )
    else:
        examples = _build_floor_examples(tasks, args.limit)
    client = LLMClient(
        model=args.target_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=500 if args.mode == "continue" else 300,
        reasoning_effort="high",
    )
    judge_client = LLMClient(
        model=args.judge_model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=350,
        reasoning_effort="high",
    )
    results: list[FloorResult] = _load_resume_results(args, examples)
    if results:
        _write_checkpoint(args, examples, results)
    for example in examples[len(results) :]:
        results.append(
            await _run_example_with_retry(
                mode=args.mode,
                client=client,
                judge_client=judge_client,
                example=example,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                variant=args.variant,
            )
        )
        _write_checkpoint(args, examples, results)
    return _summarize(args=args, examples=examples, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strongest-priming ProntoQA floor test.")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--mode", choices=["explicit", "continue"], default="explicit")
    parser.add_argument(
        "--variant",
        choices=["inconsistent", "benign", "benign_mainline"],
        default="inconsistent",
    )
    parser.add_argument("--target-model", default="qwen/qwen3.5-27b")
    parser.add_argument("--judge-model", default="qwen/qwen3.5-27b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--per-recipe", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output",
        default="outputs/results/cot_edit_introspection_pilot/prontoqa_floor_test_qwen35_27b.json",
    )
    args = parser.parse_args()
    result = asyncio.run(_main_async(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {output_path}")
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "primary_rates": {
                    key: value
                    for key, value in result.items()
                    if key.endswith("_rate") or key.endswith("_share")
                },
                "by_recipe": result["by_recipe"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
