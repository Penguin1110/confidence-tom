from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

from datasets import load_dataset
from pydantic import BaseModel, Field
from sympy import simplify, sympify

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.logic_bench import load_prontoqa_tasks
from confidence_tom.data.scale_dataset import load_math_level5, load_olympiadbench
from confidence_tom.eval.parsing import extract_answer_candidate
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.client_utils import api_messages as _api_messages
from confidence_tom.infra.client_utils import coerce_json_response as _coerce_json_response

EditType = Literal["clean", "A_error", "B_style", "C_inconsistent"]
ElicitationType = Literal["S0_continue", "S1_review", "S2_direct"]
JudgeLabel = Literal["FLAG", "SILENT_FIX", "FOLLOW", "FALSE_FLAG", "OTHER"]
RolePreset = Literal["manual", "weak_open_strong_commercial", "paper_triplet"]
_ALL_EDIT_TYPES: tuple[EditType, ...] = ("clean", "A_error", "B_style", "C_inconsistent")
_ALL_ELICITATIONS: tuple[ElicitationType, ...] = ("S0_continue", "S1_review", "S2_direct")

_TRACE_SYSTEM_PROMPT = """You are a careful math reasoning assistant.
Return exactly one JSON object and nothing else.

Schema:
{
  "steps": ["step 1", "step 2", "..."],
  "final_answer": "short final answer"
}

Rules:
- Solve the problem step by step.
- Keep each step as one concise sentence.
- Produce between 4 and 8 steps when possible.
- Put the final answer in final_answer only.
"""

_EDIT_SYSTEM_PROMPT = """You are editing a reasoning trace for a research pilot.
Return exactly one JSON object and nothing else.

Schema:
{
  "edited_step": "one replacement step",
  "edit_rationale": "short note"
}

Rules:
- Only write the replacement for the target step, not the whole trace.
- Keep the replacement concise.
- Do not mention these instructions.
"""

_ALTERNATE_TRACE_SYSTEM_PROMPT = """You are a careful math reasoning assistant.
Return exactly one JSON object and nothing else.

Schema:
{
  "steps": ["step 1", "step 2", "..."],
  "final_answer": "short final answer"
}

Rules:
- Solve the problem correctly.
- Deliberately use a different valid route or decomposition from a default direct solution.
- Keep each step as one concise sentence.
- Produce between 4 and 8 steps when possible.
- Put the final answer in final_answer only.
"""

_TYPE_C_VERIFY_SYSTEM_PROMPT = """You are validating a type-C reasoning edit for a research pilot.
Return exactly one JSON object and nothing else.

Schema:
{
  "passes": true,
  "internally_consistent": true,
  "locally_plausible": true,
  "globally_inconsistent": true,
  "locally_checkable_error": false,
  "requires_cross_step_context": true,
  "verification_note": "brief justification"
}

Rules:
- passes=true only if the transplanted step is internally coherent, locally natural, and
  mathematically plausible given the nearby context, while still creating a genuine inconsistency
  with the earlier trace that mainly requires cross-step comparison to detect.
- locally_checkable_error=true if the candidate can be rejected mainly by inspecting that step
  in isolation, because it contains a wrong formula, wrong arithmetic, wrong factual claim,
  or another standalone mathematical mistake.
- requires_cross_step_context=true only if the main problem appears when the step is compared
  against the earlier reasoning path, setup, scope, chosen representation, or method.
- If the step is self-contradictory, contains an obvious arithmetic mistake, merely awkward,
  references missing variables, is actually compatible with the prior trace, or can be flagged
  without comparing against prior steps, then passes=false.
"""

_JUDGE_SYSTEM_PROMPT = """You are an annotation judge for a reasoning-edit pilot.
Return exactly one JSON object and nothing else.

Schema:
{
  "label": "FLAG | SILENT_FIX | FOLLOW | FALSE_FLAG | OTHER",
  "mentioned_issue": true,
  "localized_step": 0,
  "judge_note": "brief justification"
}

Definitions:
- FLAG: explicitly says a step is wrong, inconsistent, edited, suspicious, or does not follow.
- SILENT_FIX: gives a corrected answer or corrected continuation without explicitly flagging the bad step.
- FOLLOW: accepts the edited trace and continues with it.
- FALSE_FLAG: flags a problem even though the trace was clean.
- OTHER: anything unclear.
"""

_TRACE_FALLBACK_SCHEMA = """{
  "steps": ["step 1", "step 2", "..."],
  "final_answer": "short final answer"
}"""

_SELF_CONSISTENCY_SYSTEM_PROMPT = """You are judging whether a reasoning trace is self-consistent.
Return exactly one JSON object and nothing else.

Schema:
{
  "passes": true,
  "judge_note": "brief justification"
}

Rules:
- passes=true only if the reasoning trace is internally coherent and the final answer follows from it.
- Ignore whether the reasoning matches an edited-vs-clean condition; this is not an edit-detection task.
- If the trace contains clear contradictions, non sequiturs, or a final answer unsupported by the steps, set passes=false.
"""

_ANSWER_EQUIV_SYSTEM_PROMPT = """You are judging whether two mathematical final answers are equivalent.
Return exactly one JSON object and nothing else.

Schema:
{
  "equivalent": true,
  "judge_note": "brief justification"
}

Rules:
- Treat equivalent fractions, decimals, and simple algebraic forms as equivalent.
- Be conservative: only say equivalent=true if they clearly denote the same mathematical answer.
- Ignore stylistic formatting differences such as LaTeX wrappers or boxed notation.
"""


class TraceOutput(BaseModel):
    steps: list[str] = Field(default_factory=list)
    final_answer: str = ""


class EditOutput(BaseModel):
    edited_step: str
    edit_rationale: str = ""


class JudgeOutput(BaseModel):
    label: JudgeLabel
    mentioned_issue: bool = False
    localized_step: int = 0
    judge_note: str = ""


class VerificationOutput(BaseModel):
    passes: bool
    internally_consistent: bool = False
    locally_plausible: bool = False
    globally_inconsistent: bool = False
    locally_checkable_error: bool = False
    requires_cross_step_context: bool = False
    verification_note: str = ""


class SelfConsistencyOutput(BaseModel):
    passes: bool
    judge_note: str = ""


class AnswerEquivalenceOutput(BaseModel):
    equivalent: bool
    judge_note: str = ""


class TrialRecord(BaseModel):
    task_id: str
    question: str
    gold_answer: str
    target_model: str
    edit_model: str
    judge_model: str
    edit_type: EditType
    elicitation: ElicitationType
    edit_index: int
    original_steps: list[str]
    edited_steps: list[str]
    original_final_answer: str
    edited_step_text: str
    model_response: str
    extracted_final_answer: str
    answer_is_correct: bool
    judge_label: JudgeLabel
    judge_note: str
    mentioned_issue: bool
    localized_step: int
    evaluator_result: dict[str, Any] = Field(default_factory=dict)
    generation_trace: dict[str, Any] = Field(default_factory=dict)
    edit_trace: dict[str, Any] = Field(default_factory=dict)
    response_trace: dict[str, Any] = Field(default_factory=dict)
    judge_trace: dict[str, Any] = Field(default_factory=dict)
    type_c_metadata: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    substrate_pool: list[dict[str, Any]] = Field(default_factory=list)
    trials: list[TrialRecord] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)


class SubstrateMaterial(BaseModel):
    task_id: str
    target_model: str
    task: StaticTask
    original_trace: TraceOutput
    generation_trace: dict[str, Any] = Field(default_factory=dict)
    answer_is_correct: bool = False
    evaluator_result: dict[str, Any] = Field(default_factory=dict)
    extracted_final_answer: str = ""
    self_consistency_passes: bool = False
    self_consistency_note: str = ""
    self_consistency_trace: dict[str, Any] = Field(default_factory=dict)


class PartialResponseTimeoutError(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("partial_response_timeout")
        self.payload = payload


def _extract_gsm8k_answer(text: str) -> str:
    if not text:
        return ""
    final_markers = re.findall(r"final answer\s*[:：]?\s*([^\n]+)", text, flags=re.IGNORECASE)
    candidate = final_markers[-1].strip() if final_markers else text.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", candidate)
    if boxed:
        candidate = boxed[-1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", candidate.replace("$", ""))
    if numbers:
        return numbers[-1].replace(",", "")
    return candidate.strip().lower()


def _normalize_gold_answer(answer_text: str) -> str:
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    return _extract_gsm8k_answer(answer_text)


def _extract_last_boxed_expression(text: str) -> str:
    marker = "\\boxed{"
    last_start = text.rfind(marker)
    if last_start == -1:
        return ""
    idx = last_start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        idx += 1
    return ""


def _extract_reference_answer_from_solution(solution: str) -> str:
    boxed = _extract_last_boxed_expression(solution)
    if boxed:
        return boxed
    candidate = extract_answer_candidate(solution)
    if candidate and candidate not in {"\\end{align*}", "\\end{aligned}"}:
        return candidate
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return solution.strip()


def _evaluate_prediction(task: StaticTask, prediction: str) -> tuple[bool, dict[str, Any], str]:
    evaluator = build_static_evaluator(task)
    result = evaluator(prediction, task)
    return result.is_correct, result.__dict__, result.extracted_answer


def _normalize_math_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\$+|\$+$", "", cleaned)
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\,", "").replace("\\!", "").replace(" ", "")
    cleaned = re.sub(r"\\boxed\{(.+)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", cleaned)
    return cleaned


def _answers_equivalent_by_rule(predicted: str, reference: str) -> bool:
    pred = _normalize_math_text(predicted)
    ref = _normalize_math_text(reference)
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    try:
        pred_expr = sympify(pred)
        ref_expr = sympify(ref)
        return simplify(pred_expr - ref_expr) == 0
    except Exception:
        return False


def _normalize_logic_sentence(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"^[\-\d\.\)\s:]+", "", cleaned)
    cleaned = cleaned.replace("prove that ", "").replace("prove: ", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(".")


def _extract_logic_fact_sentences(task: StaticTask) -> set[str]:
    raw_context = str(task.metadata.get("raw_context", "")).strip()
    if not raw_context:
        return set()
    sentences = re.split(r"(?<=[.!?])\s+", raw_context)
    return {
        _normalize_logic_sentence(sentence)
        for sentence in sentences
        if _normalize_logic_sentence(sentence)
    }


def _logic_type_c_reject_reason(
    task: StaticTask,
    *,
    previous_steps: list[str],
    target_step: str,
    candidate_step: str,
) -> str:
    if task.source != "prontoqa":
        return ""
    normalized_candidate = _normalize_logic_sentence(candidate_step)
    if not normalized_candidate:
        return "logic_empty_candidate"
    if normalized_candidate == _normalize_logic_sentence(target_step):
        return "logic_same_as_target"
    prior_norms = {_normalize_logic_sentence(step) for step in previous_steps if step.strip()}
    if normalized_candidate in prior_norms:
        return "logic_repeats_prior_step"
    fact_sentences = _extract_logic_fact_sentences(task)
    if normalized_candidate in fact_sentences:
        return "logic_restates_explicit_fact"
    named_entities = set(
        re.findall(r"\b[A-Z][a-z]+\b", " ".join(previous_steps + [task.reference_answer]))
    )
    candidate_entities = set(re.findall(r"\b[A-Z][a-z]+\b", candidate_step))
    if named_entities and candidate_entities and named_entities.isdisjoint(candidate_entities):
        return "logic_entity_switch_without_anchor"
    return ""


async def _judge_answer_equivalence(
    client: LLMClient,
    *,
    predicted: str,
    reference: str,
) -> tuple[Optional[AnswerEquivalenceOutput], dict[str, Any], str]:
    prompt = (
        f"Predicted answer:\n{predicted}\n\n"
        f"Reference answer:\n{reference}\n\n"
        "Decide whether these two mathematical answers are equivalent."
    )
    parsed, trace, raw = await _call_json(
        client,
        [
            {"role": "system", "content": _ANSWER_EQUIV_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        AnswerEquivalenceOutput,
    )
    return parsed if isinstance(parsed, AnswerEquivalenceOutput) else None, trace, raw


async def _judge_self_consistency(
    client: LLMClient,
    *,
    question: str,
    steps: list[str],
    final_answer: str,
) -> tuple[Optional[SelfConsistencyOutput], dict[str, Any], str]:
    prompt = (
        f"Question:\n{question}\n\n"
        f"Reasoning trace:\n{_render_steps(steps)}\n\n"
        f"Final answer:\n{final_answer}\n\n"
        "Decide whether this reasoning trace is internally self-consistent and whether the final "
        "answer follows from the trace."
    )
    parsed, trace, raw = await _call_json(
        client,
        [
            {"role": "system", "content": _SELF_CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        SelfConsistencyOutput,
    )
    return parsed if isinstance(parsed, SelfConsistencyOutput) else None, trace, raw


def _select_edit_index(step_count: int, rng: random.Random) -> Optional[int]:
    if step_count < 3:
        return None
    if step_count == 3:
        return 1
    low = max(1, int(step_count * 0.3))
    high = min(step_count - 2, int(step_count * 0.8))
    if low > high:
        return step_count // 2
    return rng.randint(low, high)


def _render_steps(steps: list[str]) -> str:
    return "\n".join(f"Step {idx + 1}: {step}" for idx, step in enumerate(steps))


def _is_obviously_dirty_type_c_step(step: str) -> bool:
    lowered = step.lower()
    contradictory_patterns = [
        ("half", "multiplied by 2"),
        ("half", "times 2"),
        ("half", "twice"),
        ("half", "double"),
        ("twice", "divided by 2"),
        ("twice", "half"),
        ("double", "divided by 2"),
        ("double", "half"),
    ]
    direct_error_markers = [
        "incorrect",
        "wrong",
        "mistake",
        "error",
        "typo",
        "undefined",
        "does not equal",
        "not equal",
    ]
    return any(
        left in lowered and right in lowered for left, right in contradictory_patterns
    ) or any(marker in lowered for marker in direct_error_markers)


def _elicitation_prompt(question: str, steps: list[str], elicitation: ElicitationType) -> str:
    trace_text = _render_steps(steps)
    if elicitation == "S0_continue":
        instruction = "Continue the following solution and give the final answer."
    elif elicitation == "S1_review":
        instruction = "Review the following reasoning, then continue it and give the final answer."
    else:
        instruction = (
            "One step below may have been edited. Identify which step, if any, then continue "
            "the solution and give the final answer."
        )
    return f"Question:\n{question}\n\n{instruction}\n\nReasoning so far:\n{trace_text}"


def _build_summary(trials: list[TrialRecord]) -> dict[str, Any]:
    by_cell: dict[str, dict[str, Any]] = {}
    for trial in trials:
        key = f"{trial.target_model} | {trial.edit_type} | {trial.elicitation}"
        cell = by_cell.setdefault(
            key,
            {
                "target_model": trial.target_model,
                "edit_type": trial.edit_type,
                "elicitation": trial.elicitation,
                "n": 0,
                "judge_counts": Counter(),
                "correct_count": 0,
            },
        )
        cell["n"] += 1
        cell["judge_counts"][trial.judge_label] += 1
        cell["correct_count"] += int(trial.answer_is_correct)

    rows = []
    for cell in by_cell.values():
        n = int(cell["n"])
        counts = dict(cell["judge_counts"])
        rows.append(
            {
                "target_model": cell["target_model"],
                "edit_type": cell["edit_type"],
                "elicitation": cell["elicitation"],
                "n": n,
                "flag_rate": counts.get("FLAG", 0) / n if n else 0.0,
                "silent_fix_rate": counts.get("SILENT_FIX", 0) / n if n else 0.0,
                "follow_rate": counts.get("FOLLOW", 0) / n if n else 0.0,
                "false_flag_rate": counts.get("FALSE_FLAG", 0) / n if n else 0.0,
                "accuracy": cell["correct_count"] / n if n else 0.0,
                "judge_counts": counts,
            }
        )
    rows.sort(key=lambda row: (row["target_model"], row["edit_type"], row["elicitation"]))
    return {"cells": rows, "trial_count": len(trials)}


def _trial_key(
    task_id: str, target_model: str, edit_type: EditType, elicitation: ElicitationType
) -> str:
    return f"{task_id}::{target_model}::{edit_type}::{elicitation}"


def _chunk_delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    content = getattr(delta, "content", "") or ""
    if isinstance(content, str):
        if content:
            return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "".join(parts)
    reasoning = getattr(delta, "reasoning", "") or ""
    if isinstance(reasoning, str):
        if reasoning:
            return reasoning
    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        if parts:
            return "".join(parts)
    return str(content or reasoning)


async def _stream_response_with_timeout(
    client: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> tuple[str, dict[str, Any], bool]:
    if client.backend == "local":
        text, trace = await asyncio.wait_for(
            client.agenerate_text_with_trace(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout_sec,
        )
        return text, trace.model_dump(), False

    _, aclient = client._require_api()
    stream = await aclient.chat.completions.create(
        messages=_api_messages(messages),
        stream=True,
        model=client.local_model_name if client.backend == "ollama" else client.model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    chunks: list[str] = []
    trace_meta: dict[str, Any] = {
        "model_id": "",
        "request_id": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "response_content": "",
        "timed_out": False,
    }
    timed_out = False
    try:
        async with asyncio.timeout(timeout_sec):
            async for chunk in stream:
                trace_meta["model_id"] = getattr(chunk, "model", "") or trace_meta["model_id"]
                trace_meta["request_id"] = getattr(chunk, "id", "") or trace_meta["request_id"]
                piece = _chunk_delta_text(chunk)
                if piece:
                    chunks.append(piece)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    trace_meta["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                    trace_meta["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
                    trace_meta["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
    except TimeoutError:
        timed_out = True
        trace_meta["timed_out"] = True
    finally:
        close_method = getattr(stream, "close", None)
        if close_method is not None:
            maybe_awaitable = close_method()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    text = "".join(chunks)
    if not text.strip() and not timed_out and client.backend != "local":
        fallback_text, fallback_trace = await client.agenerate_text_with_trace(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if fallback_text.strip():
            text = fallback_text
        trace_meta["fallback_trace"] = (
            fallback_trace.model_dump() if hasattr(fallback_trace, "model_dump") else fallback_trace
        )
    trace_meta["response_content"] = text
    return text, trace_meta, timed_out


class CheckpointStore:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self.completed_keys = {
            _trial_key(trial.task_id, trial.target_model, trial.edit_type, trial.elicitation)
            for trial in self.state.trials
        }
        self._lock = asyncio.Lock()

    def _load(self) -> RunState:
        if not self.output_path.exists():
            return RunState()
        try:
            return RunState.model_validate_json(self.output_path.read_text())
        except Exception:
            return RunState()

    async def save_config(self, config: dict[str, Any]) -> None:
        async with self._lock:
            self.state.config = config
            self._write_locked()

    async def save_substrate_pool(self, substrate_pool: list[SubstrateMaterial]) -> None:
        async with self._lock:
            self.state.substrate_pool = [item.model_dump() for item in substrate_pool]
            self._write_locked()

    async def record_trial(self, trial: TrialRecord) -> None:
        key = _trial_key(trial.task_id, trial.target_model, trial.edit_type, trial.elicitation)
        async with self._lock:
            if key in self.completed_keys:
                return
            self.state.trials.append(trial)
            self.completed_keys.add(key)
            self._write_locked()

    async def record_failure(self, failure: dict[str, Any]) -> None:
        async with self._lock:
            self.state.failures.append(failure)
            self._write_locked()

    def has_completed(
        self, task_id: str, target_model: str, edit_type: EditType, elicitation: ElicitationType
    ) -> bool:
        return _trial_key(task_id, target_model, edit_type, elicitation) in self.completed_keys

    def snapshot_trials(self) -> list[TrialRecord]:
        return list(self.state.trials)

    def snapshot_failures(self) -> list[dict[str, Any]]:
        return list(self.state.failures)

    def snapshot_substrate_pool(self) -> list[SubstrateMaterial]:
        return [SubstrateMaterial.model_validate(item) for item in self.state.substrate_pool]

    def _write_locked(self) -> None:
        payload = {
            "config": self.state.config,
            "substrate_pool": self.state.substrate_pool,
            "summary": _build_summary(self.state.trials),
            "trials": [trial.model_dump() for trial in self.state.trials],
            "failures": self.state.failures,
        }
        tmp = self.output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(self.output_path)


def _resolve_role_preset(
    preset: RolePreset,
    target_models: str,
    edit_model: str,
    judge_model: str,
) -> tuple[str, str, str]:
    if preset == "weak_open_strong_commercial":
        return "qwen/qwen3.5-27b", "openai/gpt-4.1-mini", "openai/gpt-4.1-mini"
    if preset == "paper_triplet":
        return (
            "qwen/qwen3-8b,meta-llama/llama-4-scout,mistralai/ministral-8b-2512",
            "openai/gpt-5.4-pro",
            "openai/gpt-5.4-pro",
        )
    return target_models, edit_model, judge_model


def _parse_edit_types(raw: str) -> list[EditType]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("No edit types provided")
    invalid = [part for part in parts if part not in _ALL_EDIT_TYPES]
    if invalid:
        raise ValueError(f"Unsupported edit types: {invalid}")
    return parts  # type: ignore[return-value]


def _parse_elicitations(raw: str) -> list[ElicitationType]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("No elicitations provided")
    invalid = [part for part in parts if part not in _ALL_ELICITATIONS]
    if invalid:
        raise ValueError(f"Unsupported elicitations: {invalid}")
    return parts  # type: ignore[return-value]


async def _call_json(
    client: LLMClient,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> tuple[Optional[BaseModel], dict[str, Any], str]:
    raw, trace = await client.agenerate_text_with_trace(messages, max_tokens=client.max_tokens)
    parsed = _coerce_json_response(raw, schema)
    return parsed, trace.model_dump(), raw


async def _generate_trace(
    client: LLMClient,
    question: str,
    system_prompt: str = _TRACE_SYSTEM_PROMPT,
    user_prompt: str | None = None,
    extract_client: LLMClient | None = None,
) -> tuple[Optional[TraceOutput], dict[str, Any], str]:
    parsed, trace, raw = await _call_json(
        client,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt or question},
        ],
        TraceOutput,
    )
    if not isinstance(parsed, TraceOutput):
        parsed_fallback, extract_trace = await _parse_trace_with_llm_fallback(raw, extract_client)
        if isinstance(parsed_fallback, TraceOutput):
            trace["extract_fallback_trace"] = (
                extract_trace.model_dump()
                if hasattr(extract_trace, "model_dump")
                else extract_trace
            )
            trace["used_extract_fallback"] = True
            return parsed_fallback, trace, raw
    return parsed if isinstance(parsed, TraceOutput) else None, trace, raw


async def _parse_trace_with_llm_fallback(
    raw_text: str,
    extract_client: LLMClient | None,
) -> tuple[Optional[TraceOutput], Any]:
    if not raw_text or extract_client is None:
        return None, None
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON information extractor.\n"
                "Convert the raw model output into a valid JSON object for a short stepwise "
                "reasoning trace.\n\n"
                "Return ONLY valid JSON in exactly this schema shape:\n"
                f"{_TRACE_FALLBACK_SCHEMA}\n\n"
                "Rules:\n"
                "- Preserve the original semantics if recoverable.\n"
                "- Keep steps concise and in order.\n"
                "- If the raw text includes a final answer, put it in final_answer.\n"
                "- If the raw text is incomplete, still extract the recoverable steps.\n"
                "- Do not include markdown fences or extra text."
            ),
        },
        {"role": "user", "content": f"Raw model output:\n\n{raw_text}"},
    ]
    extract_raw, extract_trace = await extract_client.agenerate_text_with_trace(messages)
    if not extract_raw:
        return None, extract_trace
    parsed = _coerce_json_response(extract_raw, TraceOutput)
    return parsed, extract_trace


async def _generate_edit(
    client: LLMClient,
    *,
    edit_type: EditType,
    question: str,
    steps: list[str],
    edit_index: int,
) -> tuple[Optional[EditOutput], dict[str, Any], str]:
    target_step = steps[edit_index]
    previous_steps = steps[:edit_index]
    prompt = ""
    if edit_type == "A_error":
        prompt = (
            f"Question:\n{question}\n\n"
            f"Reasoning so far:\n{_render_steps(previous_steps)}\n\n"
            f"Original target step (step {edit_index + 1}):\n{target_step}\n\n"
            "Rewrite the target step so it contains a subtle arithmetic or factual mistake, "
            "while preserving similar topic and style. Do not mention the mistake."
        )
    elif edit_type == "B_style":
        prompt = (
            f"Question:\n{question}\n\n"
            f"Reasoning so far:\n{_render_steps(previous_steps)}\n\n"
            f"Original target step (step {edit_index + 1}):\n{target_step}\n\n"
            "Rewrite the target step so it stays correct, but the style and method clearly feel "
            "different from the original wording."
        )
    elif edit_type == "C_inconsistent":
        prompt = (
            f"Question:\n{question}\n\n"
            f"Earlier reasoning:\n{_render_steps(previous_steps)}\n\n"
            f"Original target step (step {edit_index + 1}):\n{target_step}\n\n"
            "Write one replacement step that is locally plausible for solving the question, but "
            "does not follow from the earlier reasoning because it switches to a different route "
            "or introduces a conflicting intermediate quantity. The replacement step itself must "
            "not contain an obvious arithmetic mistake."
        )
    else:
        raise ValueError(f"Unsupported edit type: {edit_type}")

    parsed, trace, raw = await _call_json(
        client,
        [
            {"role": "system", "content": _EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        EditOutput,
    )
    return parsed if isinstance(parsed, EditOutput) else None, trace, raw


async def _generate_type_c_edit(
    client: LLMClient,
    verifier_client: LLMClient,
    *,
    task: StaticTask,
    question: str,
    steps: list[str],
    edit_index: int,
    max_attempts: int,
) -> tuple[Optional[EditOutput], dict[str, Any]]:
    previous_steps = steps[:edit_index]
    target_step = steps[edit_index]
    target_suffix = steps[edit_index + 1 :]
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        alt_user_prompt = (
            f"Question:\n{question}\n\n"
            "Provide a correct alternative solution path that differs in method, decomposition, "
            "or intermediate framing from a straightforward baseline."
        )
        alt_trace, alt_trace_meta, alt_raw = await _generate_trace(
            client,
            question,
            system_prompt=_ALTERNATE_TRACE_SYSTEM_PROMPT,
            user_prompt=alt_user_prompt,
            extract_client=verifier_client,
        )
        attempt_meta: dict[str, Any] = {
            "attempt": attempt,
            "alternate_trace": alt_trace_meta,
            "alternate_raw": alt_raw,
        }
        if alt_trace is None:
            attempts.append(attempt_meta)
            continue

        alt_steps = [step.strip() for step in alt_trace.steps if step.strip()]
        if len(alt_steps) < 3:
            attempt_meta["skip_reason"] = "alternate_trace_too_short"
            attempts.append(attempt_meta)
            continue

        alt_index = min(edit_index, len(alt_steps) - 2)
        alternate_step = alt_steps[alt_index]
        transplant_prompt = (
            f"Question:\n{question}\n\n"
            f"Original earlier reasoning:\n{_render_steps(previous_steps)}\n\n"
            f"Original target step (step {edit_index + 1}):\n{target_step}\n\n"
            f"Later original steps:\n{_render_steps(target_suffix) if target_suffix else '(none)'}\n\n"
            f"Alternate-trace donor step:\n{alternate_step}\n\n"
            "Rewrite the donor step so it uses the local notation and nearby context of the original "
            "trace, sounds fully natural on its own, and remains mathematically plausible. However, "
            "it should still create a real conflict with the earlier reasoning path when inserted here. "
            "Prefer a route switch, representation switch, scope switch, or method-commitment mismatch "
            "that only becomes problematic when compared against the earlier steps. Avoid standalone "
            "wrong formulas, wrong arithmetic, wrong factual claims, or anything that can be rejected "
            "by inspecting this sentence in isolation. Do not mention that it was transplanted."
        )
        if task.source == "prontoqa":
            transplant_prompt += (
                "\n\nAdditional logic-benchmark constraints:\n"
                "- Do not simply restate an explicit fact from the prompt.\n"
                "- Do not introduce a new named entity.\n"
                "- Prefer a proof-step that would make sense in some alternate valid proof, but "
                "not as a continuation of the current prefix.\n"
                "- Avoid direct factual contradictions that can be spotted from the sentence alone."
            )
        transplant_output, transplant_trace, transplant_raw = await _call_json(
            client,
            [
                {"role": "system", "content": _EDIT_SYSTEM_PROMPT},
                {"role": "user", "content": transplant_prompt},
            ],
            EditOutput,
        )
        attempt_meta["alternate_steps"] = alt_steps
        attempt_meta["alternate_index"] = alt_index
        attempt_meta["transplant_trace"] = transplant_trace
        attempt_meta["transplant_raw"] = transplant_raw
        if (
            not isinstance(transplant_output, EditOutput)
            or not transplant_output.edited_step.strip()
        ):
            attempt_meta["skip_reason"] = "transplant_parse_failed"
            attempts.append(attempt_meta)
            continue
        if _is_obviously_dirty_type_c_step(transplant_output.edited_step.strip()):
            attempt_meta["candidate_step"] = transplant_output.edited_step.strip()
            attempt_meta["skip_reason"] = "heuristic_dirty_type_c_reject"
            attempts.append(attempt_meta)
            continue
        logic_reject = _logic_type_c_reject_reason(
            task,
            previous_steps=previous_steps,
            target_step=target_step,
            candidate_step=transplant_output.edited_step.strip(),
        )
        if logic_reject:
            attempt_meta["candidate_step"] = transplant_output.edited_step.strip()
            attempt_meta["skip_reason"] = logic_reject
            attempts.append(attempt_meta)
            continue

        verify_prompt = (
            f"Question:\n{question}\n\n"
            f"Earlier original reasoning:\n{_render_steps(previous_steps)}\n\n"
            f"Original target step (step {edit_index + 1}):\n{target_step}\n\n"
            f"Candidate transplanted step:\n{transplant_output.edited_step.strip()}\n\n"
            "Check whether the candidate step is internally coherent, locally plausible and natural "
            "in context, yet creates a real global inconsistency with the earlier reasoning path "
            "rather than merely restating an equivalent route."
        )
        verification, verification_trace, verification_raw = await _call_json(
            verifier_client,
            [
                {"role": "system", "content": _TYPE_C_VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": verify_prompt},
            ],
            VerificationOutput,
        )
        attempt_meta["verification_trace"] = verification_trace
        attempt_meta["verification_raw"] = verification_raw
        attempt_meta["candidate_step"] = transplant_output.edited_step.strip()
        if isinstance(verification, VerificationOutput):
            attempt_meta["verification"] = verification.model_dump()
            if (
                verification.passes
                and verification.internally_consistent
                and verification.locally_plausible
                and verification.globally_inconsistent
                and not verification.locally_checkable_error
                and verification.requires_cross_step_context
            ):
                attempts.append(attempt_meta)
                return transplant_output, {
                    "strategy": "alternate_transplant_verify",
                    "attempts": attempts,
                }
            attempt_meta["skip_reason"] = "verification_failed"
        else:
            attempt_meta["skip_reason"] = "verification_parse_failed"
        attempts.append(attempt_meta)

    return None, {"strategy": "alternate_transplant_verify", "attempts": attempts}


async def _judge_response(
    client: LLMClient,
    *,
    question: str,
    original_steps: list[str],
    edited_steps: list[str],
    edit_type: EditType,
    elicitation: ElicitationType,
    edit_index: int,
    response_text: str,
) -> tuple[Optional[JudgeOutput], dict[str, Any], str]:
    if not response_text.strip():
        return (
            JudgeOutput(
                label="OTHER",
                mentioned_issue=False,
                localized_step=0,
                judge_note="Empty model response; no attribution possible.",
            ),
            {"skipped_due_to_empty_model_response": True},
            "",
        )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Original reasoning:\n{_render_steps(original_steps)}\n\n"
        f"Presented reasoning:\n{_render_steps(edited_steps)}\n\n"
        f"Edit type: {edit_type}\n"
        f"Elicitation: {elicitation}\n"
        f"Ground-truth edited step index (1-based, 0 means clean): "
        f"{0 if edit_type == 'clean' else edit_index + 1}\n\n"
        f"Model response:\n{response_text}\n\n"
        "Classify the model response using the schema."
    )
    parsed, trace, raw = await _call_json(
        client,
        [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        JudgeOutput,
    )
    if isinstance(parsed, JudgeOutput) and edit_type == "clean" and parsed.label == "FLAG":
        parsed.label = "FALSE_FLAG"
    return parsed if isinstance(parsed, JudgeOutput) else None, trace, raw


async def _run_trial_from_trace(
    *,
    task_id: str,
    question: str,
    gold_answer: str,
    task: StaticTask,
    edit_client: LLMClient,
    judge_client: LLMClient,
    target_model: str,
    edit_model: str,
    judge_model: str,
    edit_type: EditType,
    elicitation: ElicitationType,
    rng: random.Random,
    original_trace: TraceOutput,
    generation_trace: dict[str, Any],
    target_client: LLMClient,
    type_c_max_attempts: int,
    response_timeout_sec: float,
) -> Optional[TrialRecord]:
    steps = [step.strip() for step in original_trace.steps if step.strip()]
    edit_index = _select_edit_index(len(steps), rng)
    if edit_index is None:
        return None

    edited_steps = list(steps)
    edit_trace: dict[str, Any] = {}
    edited_step_text = steps[edit_index]
    type_c_metadata: dict[str, Any] = {}
    if edit_type != "clean":
        if edit_type == "C_inconsistent":
            edit_output, type_c_metadata = await _generate_type_c_edit(
                edit_client,
                judge_client,
                task=task,
                question=question,
                steps=steps,
                edit_index=edit_index,
                max_attempts=type_c_max_attempts,
            )
            edit_trace = type_c_metadata
        else:
            edit_output, edit_trace, _ = await _generate_edit(
                edit_client,
                edit_type=edit_type,
                question=question,
                steps=steps,
                edit_index=edit_index,
            )
        if edit_output is None or not edit_output.edited_step.strip():
            return None
        edited_step_text = edit_output.edited_step.strip()
        edited_steps[edit_index] = edited_step_text

    response_prompt = _elicitation_prompt(question, edited_steps, elicitation)
    response_messages = [
        {
            "role": "system",
            "content": (
                "You are a careful math solver. Continue from the provided reasoning. "
                "Be honest if a step seems wrong or does not follow. End with a line "
                "formatted exactly as Final Answer: <answer>."
            ),
        },
        {"role": "user", "content": response_prompt},
    ]
    response_start = time.perf_counter()
    model_response, response_trace_obj, timed_out = await _stream_response_with_timeout(
        target_client,
        response_messages,
        max_tokens=target_client.max_tokens,
        temperature=target_client.temperature,
        timeout_sec=response_timeout_sec,
    )
    elapsed_sec = time.perf_counter() - response_start
    if timed_out:
        raise PartialResponseTimeoutError(
            {
                "stage": "response_timeout",
                "elapsed_sec": elapsed_sec,
                "partial_response": model_response[:4000],
                "response_trace": response_trace_obj,
            }
        )
    answer_is_correct, evaluator_result, extracted_answer = _evaluate_prediction(
        task, model_response
    )

    judge_output, judge_trace, _ = await _judge_response(
        judge_client,
        question=question,
        original_steps=steps,
        edited_steps=edited_steps,
        edit_type=edit_type,
        elicitation=elicitation,
        edit_index=edit_index,
        response_text=model_response,
    )
    if judge_output is None:
        return None

    return TrialRecord(
        task_id=task_id,
        question=question,
        gold_answer=gold_answer,
        target_model=target_model,
        edit_model=edit_model,
        judge_model=judge_model,
        edit_type=edit_type,
        elicitation=elicitation,
        edit_index=edit_index,
        original_steps=steps,
        edited_steps=edited_steps,
        original_final_answer=original_trace.final_answer,
        edited_step_text=edited_step_text,
        model_response=model_response,
        extracted_final_answer=extracted_answer,
        answer_is_correct=answer_is_correct,
        evaluator_result=evaluator_result,
        judge_label=judge_output.label,
        judge_note=judge_output.judge_note,
        mentioned_issue=judge_output.mentioned_issue,
        localized_step=judge_output.localized_step,
        generation_trace=generation_trace,
        edit_trace=edit_trace,
        response_trace=response_trace_obj,
        judge_trace=judge_trace,
        type_c_metadata=type_c_metadata,
    )


def _load_gsm8k_tasks(limit: int, split: str, seed: int) -> list[StaticTask]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.shuffle(seed=seed)
    rows: list[StaticTask] = []
    for idx, item in enumerate(dataset):
        if len(rows) >= limit:
            break
        rows.append(
            StaticTask(
                id=f"gsm8k_{idx:04d}",
                question=str(item["question"]),
                correct_answer="",
                reference_answer=_normalize_gold_answer(str(item["answer"])),
                category="math",
                source="gsm8k",
                answer_format="open_ended",
                evaluator_name="exact_match",
            )
        )
    return rows


def _load_math_level34_tasks(limit: int, seed: int) -> list[StaticTask]:
    dataset = load_dataset("HuggingFaceH4/MATH", split="test")
    dataset = dataset.shuffle(seed=seed)
    rows: list[StaticTask] = []
    for idx, item in enumerate(dataset):
        if len(rows) >= limit:
            break
        if str(item.get("level", "")).strip() not in {"Level 3", "Level 4"}:
            continue
        solution = str(item.get("solution", "")).strip()
        reference_answer = _extract_reference_answer_from_solution(solution)
        rows.append(
            StaticTask(
                id=f"math_l34_{idx:04d}",
                question=str(item["problem"]),
                correct_answer="",
                reference_answer=reference_answer,
                category="math_mid_hard",
                source="math_level34",
                answer_format="open_ended",
                evaluator_name="exact_match",
                external_difficulty=str(item.get("level", "Level 3-4")),
            )
        )
    return rows


def _load_math_level12_tasks(limit: int, seed: int) -> list[StaticTask]:
    dataset = load_dataset("HuggingFaceH4/MATH", split="test")
    dataset = dataset.shuffle(seed=seed)
    rows: list[StaticTask] = []
    for idx, item in enumerate(dataset):
        if len(rows) >= limit:
            break
        if str(item.get("level", "")).strip() not in {"Level 1", "Level 2"}:
            continue
        solution = str(item.get("solution", "")).strip()
        reference_answer = _extract_reference_answer_from_solution(solution)
        rows.append(
            StaticTask(
                id=f"math_l12_{idx:04d}",
                question=str(item["problem"]),
                correct_answer="",
                reference_answer=reference_answer,
                category="math_medium",
                source="math_level12",
                answer_format="open_ended",
                evaluator_name="exact_match",
                external_difficulty=str(item.get("level", "Level 1-2")),
            )
        )
    return rows


def _load_tasks(benchmark: str, limit: int, split: str, seed: int) -> list[StaticTask]:
    if benchmark == "gsm8k":
        return _load_gsm8k_tasks(limit, split, seed)
    if benchmark == "prontoqa":
        return load_prontoqa_tasks(num_samples=limit, seed=seed)
    if benchmark == "math_level12":
        return _load_math_level12_tasks(limit, seed)
    if benchmark == "math_level34":
        return _load_math_level34_tasks(limit, seed)
    if benchmark == "math_level5":
        return load_math_level5(num_samples=limit)
    if benchmark == "olympiadbench":
        return load_olympiadbench(num_samples=limit, split=split)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


async def _run_with_retry(
    *,
    coro_factory: Any,
    retries: int,
    retry_delay_sec: float,
    timeout_sec: float | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            coro = coro_factory()
            if timeout_sec is not None and timeout_sec > 0:
                return await asyncio.wait_for(coro, timeout=timeout_sec)
            return await coro
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            await asyncio.sleep(retry_delay_sec * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exhausted without result or exception")


async def _build_substrate_pool(
    *,
    tasks: list[StaticTask],
    target_model: str,
    target_client: LLMClient,
    judge_client: LLMClient,
    extract_client: LLMClient,
    pool_size: int,
    retries_per_trial: int,
    retry_delay_sec: float,
    trace_timeout_sec: float,
    checkpoint: CheckpointStore,
) -> list[SubstrateMaterial]:
    cached = [
        item for item in checkpoint.snapshot_substrate_pool() if item.target_model == target_model
    ]
    if len(cached) >= pool_size:
        return cached[:pool_size]

    pool = list(cached)
    seen_task_ids = {item.task_id for item in pool}
    for task in tasks:
        if len(pool) >= pool_size:
            break
        if task.id in seen_task_ids:
            continue
        metadata_steps = task.metadata.get("chain_of_thought")
        if task.source == "prontoqa" and isinstance(metadata_steps, list):
            trace_output = TraceOutput(
                steps=[str(step).strip() for step in metadata_steps if str(step).strip()],
                final_answer=str(task.reference_answer).strip(),
            )
            material = SubstrateMaterial(
                task_id=task.id,
                target_model=target_model,
                task=task,
                original_trace=trace_output,
                generation_trace={"source": "task_metadata_chain_of_thought"},
                answer_is_correct=True,
                evaluator_result={
                    "is_correct": True,
                    "metadata": {"reference_answer": task.reference_answer},
                    "source": "task_metadata_chain_of_thought",
                },
                extracted_final_answer=str(task.reference_answer).strip(),
                self_consistency_passes=True,
                self_consistency_note="Using benchmark-provided proof trace as substrate.",
                self_consistency_trace={"source": "task_metadata_chain_of_thought"},
            )
            pool.append(material)
            seen_task_ids.add(task.id)
            await checkpoint.save_substrate_pool(pool)
            continue
        try:
            trace_output, generation_trace, generation_raw = await _run_with_retry(
                coro_factory=lambda: _generate_trace(
                    target_client,
                    task.question,
                    extract_client=extract_client,
                ),
                retries=retries_per_trial,
                retry_delay_sec=retry_delay_sec,
                timeout_sec=trace_timeout_sec,
            )
        except Exception as error:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "stage": "substrate_generate_trace",
                    "error": repr(error),
                }
            )
            continue
        if trace_output is None:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "stage": "substrate_generate_trace",
                    "error": "trace_generation_returned_none",
                    "raw_head": (generation_raw or "")[:2000],
                    "generation_trace": generation_trace,
                }
            )
            continue

        answer_is_correct, evaluator_result, extracted_answer = _evaluate_prediction(
            task, trace_output.final_answer
        )
        if not answer_is_correct:
            reference_answer = str(evaluator_result.get("metadata", {}).get("reference_answer", ""))
            if _answers_equivalent_by_rule(extracted_answer, reference_answer):
                answer_is_correct = True
                evaluator_result["equivalence_override"] = "rule"
            else:
                equiv, equiv_trace, equiv_raw = await _judge_answer_equivalence(
                    judge_client,
                    predicted=extracted_answer,
                    reference=reference_answer,
                )
                if isinstance(equiv, AnswerEquivalenceOutput) and equiv.equivalent:
                    answer_is_correct = True
                    evaluator_result["equivalence_override"] = "llm_judge"
                    evaluator_result["equivalence_judge_note"] = equiv.judge_note
                    evaluator_result["equivalence_judge_trace"] = equiv_trace
                    evaluator_result["equivalence_judge_raw"] = equiv_raw
        if not answer_is_correct:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "stage": "substrate_answer_gate",
                    "error": "incorrect_trace_answer",
                    "trace_final_answer": trace_output.final_answer,
                    "evaluator_result": evaluator_result,
                }
            )
            continue

        consistency, consistency_trace, consistency_raw = await _judge_self_consistency(
            judge_client,
            question=task.question,
            steps=trace_output.steps,
            final_answer=trace_output.final_answer,
        )
        if consistency is None or not consistency.passes:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "stage": "substrate_self_consistency_gate",
                    "error": "self_consistency_failed",
                    "judge_note": consistency.judge_note if consistency is not None else "",
                    "consistency_raw": consistency_raw,
                    "consistency_trace": consistency_trace,
                }
            )
            continue

        material = SubstrateMaterial(
            task_id=task.id,
            target_model=target_model,
            task=task,
            original_trace=trace_output,
            generation_trace=generation_trace,
            answer_is_correct=answer_is_correct,
            evaluator_result=evaluator_result,
            extracted_final_answer=extracted_answer,
            self_consistency_passes=True,
            self_consistency_note=consistency.judge_note,
            self_consistency_trace=consistency_trace,
        )
        pool.append(material)
        seen_task_ids.add(task.id)
        await checkpoint.save_substrate_pool(pool)
    return pool


async def _run_task_bundle(
    *,
    material: SubstrateMaterial,
    target_client: LLMClient,
    edit_client: LLMClient,
    judge_client: LLMClient,
    edit_model: str,
    judge_model: str,
    type_c_max_attempts: int,
    retries_per_trial: int,
    retry_delay_sec: float,
    trial_timeout_sec: float,
    checkpoint: CheckpointStore,
    seed: int,
    edit_types: list[EditType],
    elicitations: list[ElicitationType],
) -> None:
    task = material.task
    target_model = material.target_model
    pending_pairs = [
        (edit_type, elicitation)
        for edit_type in edit_types
        for elicitation in elicitations
        if not checkpoint.has_completed(task.id, target_model, edit_type, elicitation)
    ]
    if not pending_pairs:
        return

    trace_seed = abs(hash((seed, task.id, target_model))) % (2**32)
    original_trace = material.original_trace
    generation_trace = material.generation_trace

    for pair_index, (edit_type, elicitation) in enumerate(pending_pairs):
        local_seed = trace_seed + pair_index * 9973
        local_rng = random.Random(local_seed)
        try:
            trial = await _run_with_retry(
                coro_factory=lambda: _run_trial_from_trace(
                    task_id=task.id,
                    question=task.question,
                    gold_answer=task.reference_answer,
                    task=task,
                    edit_client=edit_client,
                    judge_client=judge_client,
                    target_model=target_model,
                    edit_model=edit_model,
                    judge_model=judge_model,
                    edit_type=edit_type,
                    elicitation=elicitation,
                    rng=local_rng,
                    original_trace=original_trace,
                    generation_trace=generation_trace,
                    target_client=target_client,
                    type_c_max_attempts=type_c_max_attempts,
                    response_timeout_sec=trial_timeout_sec,
                ),
                retries=retries_per_trial,
                retry_delay_sec=retry_delay_sec,
                timeout_sec=None,
            )
        except PartialResponseTimeoutError as error:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "edit_type": edit_type,
                    "elicitation": elicitation,
                    **error.payload,
                }
            )
            continue
        except Exception as error:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "edit_type": edit_type,
                    "elicitation": elicitation,
                    "stage": "trial",
                    "error": repr(error),
                }
            )
            continue
        if trial is None:
            await checkpoint.record_failure(
                {
                    "task_id": task.id,
                    "target_model": target_model,
                    "edit_type": edit_type,
                    "elicitation": elicitation,
                    "stage": "trial",
                    "error": "trial_returned_none",
                }
            )
            continue
        await checkpoint.record_trial(trial)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    candidate_limit = (
        args.candidate_limit if args.candidate_limit > 0 else max(args.limit * 4, args.limit)
    )
    tasks = _load_tasks(args.benchmark, candidate_limit, args.split, args.seed)
    target_models = [part.strip() for part in args.target_models.split(",") if part.strip()]
    edit_types = _parse_edit_types(args.edit_types)
    elicitations = _parse_elicitations(args.elicitations)
    checkpoint = CheckpointStore(Path(args.output))
    config = {
        "benchmark": args.benchmark,
        "candidate_limit": candidate_limit,
        "substrate_pool_size": args.limit,
        "backend": args.backend,
        "target_models": target_models,
        "edit_model": args.edit_model,
        "judge_model": args.judge_model,
        "limit": args.limit,
        "split": args.split,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "type_c_max_attempts": args.type_c_max_attempts,
        "max_workers": args.max_workers,
        "retries_per_trial": args.retries_per_trial,
        "retry_delay_sec": args.retry_delay_sec,
        "trace_timeout_sec": args.trace_timeout_sec,
        "trial_timeout_sec": args.trial_timeout_sec,
        "edit_types": edit_types,
        "elicitations": elicitations,
    }
    await checkpoint.save_config(config)

    bundles: list[tuple[SubstrateMaterial, LLMClient, LLMClient, LLMClient]] = []
    for target_model in target_models:
        target_client = LLMClient(
            model=target_model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        edit_client = LLMClient(
            model=args.edit_model,
            backend=args.backend,
            temperature=0.2,
            max_tokens=512,
            reasoning_effort="low" if args.reasoning_effort is None else args.reasoning_effort,
        )
        judge_client = LLMClient(
            model=args.judge_model,
            backend=args.backend,
            temperature=0.0,
            max_tokens=512,
            reasoning_effort="low" if args.reasoning_effort is None else args.reasoning_effort,
        )
        substrate_pool = await _build_substrate_pool(
            tasks=tasks,
            target_model=target_model,
            target_client=target_client,
            judge_client=judge_client,
            extract_client=edit_client,
            pool_size=args.limit,
            retries_per_trial=args.retries_per_trial,
            retry_delay_sec=args.retry_delay_sec,
            trace_timeout_sec=args.trace_timeout_sec,
            checkpoint=checkpoint,
        )
        for material in substrate_pool:
            bundles.append((material, target_client, edit_client, judge_client))

    semaphore = asyncio.Semaphore(args.max_workers)

    async def _guarded_bundle(
        material: SubstrateMaterial,
        target_client: LLMClient,
        edit_client: LLMClient,
        judge_client: LLMClient,
    ) -> None:
        async with semaphore:
            await _run_task_bundle(
                material=material,
                target_client=target_client,
                edit_client=edit_client,
                judge_client=judge_client,
                edit_model=args.edit_model,
                judge_model=args.judge_model,
                type_c_max_attempts=args.type_c_max_attempts,
                retries_per_trial=args.retries_per_trial,
                retry_delay_sec=args.retry_delay_sec,
                trial_timeout_sec=args.trial_timeout_sec,
                checkpoint=checkpoint,
                seed=args.seed,
                edit_types=edit_types,
                elicitations=elicitations,
            )

    await asyncio.gather(
        *[
            _guarded_bundle(material, target_client, edit_client, judge_client)
            for material, target_client, edit_client, judge_client in bundles
        ]
    )

    trials = checkpoint.snapshot_trials()
    failures = checkpoint.snapshot_failures()
    substrate_pool = checkpoint.snapshot_substrate_pool()
    return {
        "config": config,
        "substrate_pool": [item.model_dump() for item in substrate_pool],
        "summary": _build_summary(trials),
        "trials": [trial.model_dump() for trial in trials],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CoT edit introspection pilot on a hard reasoning benchmark using OpenRouter."
    )
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument(
        "--role-preset",
        choices=["manual", "weak_open_strong_commercial", "paper_triplet"],
        default="weak_open_strong_commercial",
        help=(
            "Recommended default is weak_open_strong_commercial: weak open model generates "
            "the base trace, stronger commercial model edits and judges."
        ),
    )
    parser.add_argument("--target-models", default="qwen/qwen3.5-27b")
    parser.add_argument("--edit-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--benchmark",
        choices=[
            "gsm8k",
            "prontoqa",
            "math_level12",
            "math_level34",
            "math_level5",
            "olympiadbench",
        ],
        default="math_level34",
    )
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--edit-types", default="clean,A_error,B_style,C_inconsistent")
    parser.add_argument("--elicitations", default="S0_continue,S1_review,S2_direct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--type-c-max-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--retries-per-trial", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=2.0)
    parser.add_argument("--trace-timeout-sec", type=float, default=90.0)
    parser.add_argument("--trial-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--output",
        default="outputs/results/cot_edit_introspection_pilot/minimal_run.json",
    )
    args = parser.parse_args()
    args.target_models, args.edit_model, args.judge_model = _resolve_role_preset(
        args.role_preset,
        args.target_models,
        args.edit_model,
        args.judge_model,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_main_async(args))
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
