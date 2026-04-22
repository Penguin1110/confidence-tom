from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from confidence_tom.infra.client import LLMClient
from confidence_tom.intervention.models import (
    ExtractedFinalAnswerOutput,
    NextStepOutput,
    SegmentedTraceOutput,
    StepwiseWorkerOutput,
)

T = TypeVar("T", bound=BaseModel)


def _extract_first_json_object(raw: str) -> str:
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


def _coerce_json_response_local(raw: str, response_model: Type[T]) -> Optional[T]:
    candidate = _extract_first_json_object(raw.strip())
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    try:
        return response_model.model_validate(data)
    except Exception:
        return None


_NEXT_STEP_SCHEMA = """{
  "next_step": {
    "step": 1,
    "subgoal": "...",
    "reasoning": "...",
    "partial_answer": "...",
    "step_confidence": 72,
    "assumptions": ["..."],
    "uncertainty_note": "...",
    "is_revision": false,
    "revision_target": "",
    "intermediate_result": "...",
    "verification_status": "none"
  },
  "done": false,
  "final_answer": "",
  "final_confidence": 0,
  "parse_incomplete": false,
  "parse_incomplete_note": ""
}"""


_STEPWISE_SCHEMA = """{
  "steps": [
    {
      "step": 1,
      "subgoal": "...",
      "reasoning": "...",
      "partial_answer": "...",
      "step_confidence": 72,
      "assumptions": ["..."],
      "uncertainty_note": "...",
      "is_revision": false,
      "revision_target": "",
      "intermediate_result": "...",
      "verification_status": "verified"
    }
  ],
  "final_answer": "...",
  "final_confidence": 82,
  "parse_incomplete": false,
  "parse_incomplete_note": ""
}"""

_SEGMENTED_TRACE_SCHEMA = """{
  "segments": [
    {
      "segment_id": "seg_1",
      "index": 1,
      "text": "..."
    }
  ],
  "final_answer": "...",
  "parse_incomplete": false,
  "parse_incomplete_note": ""
}"""

_FINAL_ANSWER_SCHEMA = """{
  "final_answer": "...",
  "parse_incomplete": false,
  "parse_incomplete_note": ""
}"""


def _schema_text(response_model: Type[Any]) -> str:
    if response_model is NextStepOutput:
        return _NEXT_STEP_SCHEMA
    if response_model is StepwiseWorkerOutput:
        return _STEPWISE_SCHEMA
    if response_model is SegmentedTraceOutput:
        return _SEGMENTED_TRACE_SCHEMA
    if response_model is ExtractedFinalAnswerOutput:
        return _FINAL_ANSWER_SCHEMA
    raise ValueError(f"Unsupported response_model: {response_model}")


def _task_label(response_model: Type[Any]) -> str:
    if response_model is NextStepOutput:
        return "single-step math reasoning state"
    if response_model is StepwiseWorkerOutput:
        return "full stepwise math reasoning trace"
    if response_model is SegmentedTraceOutput:
        return "segmented full reasoning trace"
    if response_model is ExtractedFinalAnswerOutput:
        return "final-answer extraction"
    return "structured reasoning trace"


def _extract_messages_from_raw(raw: str, response_model: Type[Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict JSON information extractor.\n"
                "Convert the raw model output into a valid JSON object for a "
                f"{_task_label(response_model)}.\n\n"
                "Return ONLY valid JSON in exactly this schema shape:\n"
                f"{_schema_text(response_model)}\n\n"
                "Rules:\n"
                "- Preserve the original semantics if recoverable.\n"
                "- Normalize minor enum/style mismatches.\n"
                "- If a field is missing, use a safe default.\n"
                "- Do NOT invent mathematical facts, new steps, or a final "
                "answer that is not explicitly supported by the raw text.\n"
                "- For segmented traces, only regroup existing reasoning into "
                "coherent segments; do not add new reasoning content.\n"
                "- For segmented traces, prefer coarse, semantically self-"
                "contained reasoning chunks rather than sentence-level splits.\n"
                "- Do NOT create standalone segments for section headers, "
                "transition phrases, or isolated formula blocks when they "
                "clearly belong with adjacent reasoning.\n"
                "- Merge short setup lines, short transitions, and displayed "
                "equations into the surrounding reasoning chunk when possible.\n"
                "- A good segment usually corresponds to one subgoal or one "
                "meaningful reasoning unit, not one line.\n"
                "- For final-answer extraction, only return an answer "
                "explicitly supported by the raw text.\n"
                "- Only set done=true or final_answer when the raw text "
                "clearly indicates the step finishes the task.\n"
                "- If the raw text is missing important information needed for "
                "a reliable extraction, set parse_incomplete=true and explain "
                "briefly in parse_incomplete_note.\n"
                "- Do not include markdown fences or extra text."
            ),
        },
        {"role": "user", "content": f"Raw model output:\n\n{raw}"},
    ]


async def parse_with_llm_fallback(
    raw_text: str,
    response_model: Type[T],
    extract_client: Optional[LLMClient],
) -> tuple[Optional[T], Any]:
    if not raw_text or extract_client is None:
        return None, None

    extract_raw, extract_trace = await extract_client.agenerate_text_with_trace(
        _extract_messages_from_raw(raw_text, response_model)
    )
    if not extract_raw:
        return None, extract_trace

    parsed = _coerce_json_response_local(extract_raw, response_model)
    return parsed, extract_trace
