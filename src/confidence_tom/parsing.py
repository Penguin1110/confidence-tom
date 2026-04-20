"""Robust parsing utilities for MC model responses.

Handles the reality that smaller models often:
- Don't follow JSON format
- Output confidence outside 0-100 range
- Give invalid answer letters
- Include extra text around structured output
"""

import json
import logging
import re
from typing import Optional

_REASONING_TAG_PATTERNS = [
    r"<think>.*?</think>",
    r"<analysis>.*?</analysis>",
    r"<reasoning>.*?</reasoning>",
    r"<thought>.*?</thought>",
    r"<scratchpad>.*?</scratchpad>",
]


_ANSWER_CANDIDATE_PATTERNS = [
    r"\bfinal answer\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bthe final answer\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bthe answer is\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\banswer is\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bmy answer\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\btherefore[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bthus[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bso[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bhence[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
    r"\bwe conclude that\b\s*(.+?)(?:\n|$)",
    r"\bthe result is\b\s*[:=\-]?\s*(.+?)(?:\n|$)",
]

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _strip_reasoning_artifacts(text: str) -> str:
    cleaned = text or ""
    for pattern in _REASONING_TAG_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(
        r"</?(?:analysis|think|reasoning|thought|scratchpad)>", "", cleaned, flags=re.IGNORECASE
    )
    return cleaned.strip()


def _strip_box_wrappers(text: str) -> str:
    cleaned = text.strip()
    patterns = [
        r"^\$?\\boxed\{(.+)\}\$?$",
        r"^\$?\\fbox\{(.+)\}\$?$",
        r"^\$?boxed\{(.+)\}\$?$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = re.match(pattern, cleaned, flags=re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                changed = True
    return cleaned


def extract_answer_candidate(text: str) -> str:
    cleaned = _strip_reasoning_artifacts(text)
    if not cleaned:
        return ""

    for pattern in _ANSWER_CANDIDATE_PATTERNS:
        match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip().strip('"').strip("'")
            answer = re.sub(r"[.,;:]\s*$", "", answer).strip()
            answer = _strip_box_wrappers(answer)
            if answer:
                return answer

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    tail = lines[-1]
    tail = re.sub(r"^(?:therefore|thus|hence|so)[,:]?\s*", "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"^(?:the )?answer(?: is)?[:\-]?\s*", "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"^(?:the )?final answer[:\-]?\s*", "", tail, flags=re.IGNORECASE)
    tail = tail.strip().strip('"').strip("'")
    tail = re.sub(r"[.,;:]\s*$", "", tail).strip()
    return _strip_box_wrappers(tail)


class MCResponse(BaseModel):
    """Parsed response from a model answering an MC question."""

    answer: str = Field(description="Answer letter: A to J")
    confidence: float = Field(description="Confidence in 0-1 scale (normalized from 0-100)")
    strategy: str = Field(description="High-level solution plan", default="")
    reasoning: str = Field(description="Step-by-step reasoning", default="")


class StaticResponse(BaseModel):
    """Parsed response from a static benchmark prompt."""

    answer: str = Field(description="Final answer string")
    confidence: float = Field(description="Confidence in 0-1 scale")
    strategy: str = Field(description="High-level solution plan", default="")
    reasoning: str = Field(description="Step-by-step reasoning", default="")


class ExtractResponse(BaseModel):
    """Minimal parser schema for LLM extraction."""

    answer: str = Field(description="Final extracted answer")
    confidence: float = Field(description="Confidence in 0-1 scale")


# ---- Parsing stats tracker ----

_parse_stats: dict[str, dict[str, int]] = {}


def get_parse_stats() -> dict[str, dict[str, int]]:
    """Return parse failure stats per model."""
    return _parse_stats.copy()


def reset_parse_stats() -> None:
    """Reset parse failure stats."""
    global _parse_stats
    _parse_stats = {}


def _track_parse(model_name: str, success: bool) -> None:
    """Track parse success/failure for a model."""
    if model_name not in _parse_stats:
        _parse_stats[model_name] = {"success": 0, "failure": 0}
    if success:
        _parse_stats[model_name]["success"] += 1
    else:
        _parse_stats[model_name]["failure"] += 1


# ---- Confidence normalization ----


def normalize_confidence(value: int | float) -> float:
    """Clamp confidence to [0, 100], then convert to 0-1 scale.

    Handles common model quirks:
    - Values > 100 (e.g. "150" confidence)
    - Values < 0
    - Values already in 0-1 range (e.g. "0.85")
    """
    # If value is already in 0-1 range (likely the model output 0-1 instead of 0-100)
    if 0 <= value <= 1:
        return float(value)

    # Clamp to 0-100, then normalize
    clamped = max(0.0, min(100.0, float(value)))
    return clamped / 100.0


# ---- Core parsing ----


def parse_mc_response(
    raw_text: str,
    model_name: str = "unknown",
    valid_choices: list[str] | None = None,
) -> Optional[MCResponse]:
    """Robustly parse a model's MC response with multiple fallback strategies.

    Strategy chain:
    1. Try JSON parse (ideal case)
    2. Try regex extraction for answer + confidence
    3. Try line-by-line extraction

    Args:
        raw_text: Raw model output text.
        model_name: For tracking parse stats per model.
        valid_choices: Valid answer letters (default: A, B, C, D).

    Returns:
        Parsed MCResponse or None if all strategies fail.
    """
    raw_text = _strip_reasoning_artifacts(raw_text)
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # Strategy 1: JSON parse
    result = _try_json_parse(raw_text, valid_choices)
    if result:
        _track_parse(model_name, True)
        return result

    # Strategy 2: Regex extraction
    result = _try_regex_parse(raw_text, valid_choices)
    if result:
        _track_parse(model_name, True)
        return result

    # Strategy 3: Line-by-line extraction
    result = _try_line_parse(raw_text, valid_choices)
    if result:
        _track_parse(model_name, True)
        return result

    # All strategies failed
    _track_parse(model_name, False)
    logger.warning(
        f"[{model_name}] Failed to parse response. First 200 chars: "
        f"{raw_text[:200].replace(chr(10), ' ')}"
    )
    return None


def parse_static_response(
    raw_text: str,
    model_name: str = "unknown",
) -> Optional[StaticResponse]:
    """Parse a free-form static-task answer from JSON or light regex."""
    raw_text = _strip_reasoning_artifacts(raw_text)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    text = json_match.group(1) if json_match else raw_text
    if not json_match:
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = raw_text[start_idx : end_idx + 1]

    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip()
        confidence = normalize_confidence(float(data.get("confidence", 50)))
        if not answer:
            return None
        return StaticResponse(
            answer=answer,
            confidence=confidence,
            strategy=str(data.get("strategy", "")),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    final_patterns = [
        r"\bfinal answer\b\s*[:=\-]\s*(.+?)(?:\n|$)",
        r"\btherefore[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
        r"\bso[, ]+(?:the )?answer(?: is)?\s*[:=\-]?\s*(.+?)(?:\n|$)",
        r"\bmy answer\b\s*[:=\-]\s*(.+?)(?:\n|$)",
    ]
    for pat in final_patterns:
        match = re.search(pat, raw_text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = extract_answer_candidate(match.group(1)) or extract_answer_candidate(raw_text)
            if answer:
                conf_match = re.search(
                    r'["\']?[Cc]onfidence["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?',
                    raw_text,
                    re.IGNORECASE,
                )
                confidence = normalize_confidence(float(conf_match.group(1))) if conf_match else 0.5
                return StaticResponse(
                    answer=answer,
                    confidence=confidence,
                    strategy="",
                    reasoning=raw_text.strip(),
                )

    answer_match = re.search(
        r'["\']?[Aa]nswer["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:,|$)', raw_text, re.DOTALL
    )
    conf_match = re.search(
        r'["\']?[Cc]onfidence["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?',
        raw_text,
        re.IGNORECASE,
    )
    if answer_match:
        answer = extract_answer_candidate(answer_match.group(1)) or extract_answer_candidate(
            raw_text
        )
        confidence = normalize_confidence(float(conf_match.group(1))) if conf_match else 0.5
        return StaticResponse(
            answer=answer,
            confidence=confidence,
            strategy="",
            reasoning=raw_text.strip(),
        )

    _track_parse(model_name, False)
    return None


def parse_extract_response(
    raw_text: str,
    model_name: str = "unknown",
) -> Optional[ExtractResponse]:
    """Parse a minimal extractor response that only contains answer/confidence."""
    raw_text = _strip_reasoning_artifacts(raw_text)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    text = json_match.group(1) if json_match else raw_text
    if not json_match:
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = raw_text[start_idx : end_idx + 1]

    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip()
        confidence = normalize_confidence(float(data.get("confidence", 50)))
        if not answer:
            return None
        return ExtractResponse(answer=answer, confidence=confidence)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    answer_match = re.search(
        r'["\']?[Aa]nswer["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:,|\n|$)',
        raw_text,
        re.DOTALL,
    )
    conf_match = re.search(
        r'["\']?[Cc]onfidence["\']?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?',
        raw_text,
        re.IGNORECASE,
    )
    if answer_match:
        answer = extract_answer_candidate(answer_match.group(1)) or extract_answer_candidate(
            raw_text
        )
        if answer:
            confidence = normalize_confidence(float(conf_match.group(1))) if conf_match else 0.5
            return ExtractResponse(answer=answer, confidence=confidence)

    _track_parse(model_name, False)
    return None


def _try_json_parse(raw: str, valid: list[str]) -> Optional[MCResponse]:
    """Attempt to parse as JSON, handling markdown code blocks."""
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    text = json_match.group(1) if json_match else raw

    # Also try finding raw JSON object if no code block
    if not json_match:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = raw[start_idx : end_idx + 1]
            json_match = True

    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip().upper()
        confidence_raw = data.get("confidence", 50)
        strategy = str(data.get("strategy", ""))
        reasoning = str(data.get("reasoning", ""))

        # Validate answer
        if answer and answer[0] in valid:
            answer = answer[0]
        else:
            return None

        confidence = normalize_confidence(float(confidence_raw))
        return MCResponse(
            answer=answer,
            confidence=confidence,
            strategy=strategy,
            reasoning=reasoning,
        )
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def _try_regex_parse(raw: str, valid: list[str]) -> Optional[MCResponse]:
    """Extract answer and confidence using regex patterns."""
    answer = None
    confidence = None
    reasoning = ""

    # Answer patterns (accommodating quotes for invalid JSON)
    answer_patterns = [
        r'["\']?[Aa]nswer["\']?[:\s]*["\']?\(?([A-Ja-j])\)?',
        r'["\']?[Cc]hoice["\']?[:\s]*["\']?\(?([A-Ja-j])\)?',
        r'["\']?[Ss]elect(?:ed|ion)?["\']?[:\s]*["\']?\(?([A-Ja-j])\)?',
        r"^([A-Ja-j])\)",  # Just "A)" at start of line
        r"\b(?:The answer is|I choose|My answer is|answer is)\s*\(?([A-Ja-j])\)?",
        r"\b(?:option|choice)\s*(?:is|:)?\s*\(?([A-Ja-j])\)?",
        r"\b(?:correct\s+(?:answer|option|choice))\s*(?:is|:)?\s*\(?([A-Ja-j])\)?",
        r"\(([A-Ja-j])\)\s*(?:is\s+)?(?:the\s+)?(?:correct|best)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, raw, re.MULTILINE | re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                answer = letter
                break

    # Confidence patterns (accommodating quotes for invalid JSON)
    conf_patterns = [
        r'["\']?[Cc]onfidence["\']?[:\s]*["\']?(\d+(?:\.\d+)?)',
        r'["\']?[Cc]onfident["\']?[:\s]*["\']?(\d+(?:\.\d+)?)',
        r"(\d+(?:\.\d+)?)\s*%?\s*confident",
        r"(\d+(?:\.\d+)?)\s*%\s*confiden",
        r"[Ii]\s+am\s+(\d+(?:\.\d+)?)\s*%",
        r"confidence.*?(\d+(?:\.\d+)?)",
        r"\bconfidence(?:\s*level)?\s*(?:is|:)?\s*(\d+(?:\.\d+)?)\s*%?",
        r"\b(\d+(?:\.\d+)?)\s*/\s*100\b",
    ]

    for pattern in conf_patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                confidence = normalize_confidence(float(match.group(1)))
                break
            except ValueError:
                continue

    # Reasoning: take everything that's not answer/confidence
    reasoning_match = re.search(
        r"[Rr]eason(?:ing)?[:\s]*(.*?)(?=[Aa]nswer|[Cc]onfidence|$)",
        raw,
        re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    if answer and confidence is not None:
        return MCResponse(answer=answer, confidence=confidence, reasoning=reasoning)

    # Qualitative fallback for models that refuse numeric confidence
    if answer and confidence is None:
        raw_lower = raw.lower()
        qualitative_conf = [
            (0.95, ["extremely confident", "very certain", "almost certain"]),
            (0.85, ["very confident", "high confidence", "highly confident"]),
            (0.70, ["confident", "fairly confident", "reasonably confident"]),
            (0.55, ["somewhat confident", "moderately confident"]),
            (0.35, ["uncertain", "not sure", "low confidence", "not very confident"]),
            (0.15, ["guess", "pure guess", "wild guess"]),
        ]
        for val, phrases in qualitative_conf:
            if any(p in raw_lower for p in phrases):
                return MCResponse(answer=answer, confidence=val, reasoning=reasoning)

    return None


def _try_line_parse(raw: str, valid: list[str]) -> Optional[MCResponse]:
    """Last resort: scan lines for answer letter and numbers."""
    lines = raw.strip().split("\n")
    answer = None
    confidence = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for a lone answer letter
        if not answer:
            match = re.match(r"^([A-Ja-j])[\s\.\)]*$", line)
            if match and match.group(1).upper() in valid:
                answer = match.group(1).upper()

        # Look for a lone number (could be confidence)
        if confidence is None:
            match = re.match(r"^(\d{1,3})(?:\.\d+)?$", line)
            if match:
                val = float(match.group(1))
                if 0 <= val <= 100:
                    confidence = normalize_confidence(val)

    if answer and confidence is not None:
        return MCResponse(answer=answer, confidence=confidence, reasoning="")

    return None
