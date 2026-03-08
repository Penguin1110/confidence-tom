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

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCResponse(BaseModel):
    """Parsed response from a model answering an MC question."""

    answer: str = Field(description="Answer letter: A, B, C, or D")
    confidence: float = Field(description="Confidence in 0-1 scale (normalized from 0-100)")
    reasoning: str = Field(description="Step-by-step reasoning", default="")


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
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D"]

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


def _try_json_parse(raw: str, valid: list[str]) -> Optional[MCResponse]:
    """Attempt to parse as JSON, handling markdown code blocks."""
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    text = json_match.group(1) if json_match else raw

    # Also try finding raw JSON object
    if not json_match:
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip().upper()
        confidence_raw = data.get("confidence", 50)
        reasoning = str(data.get("reasoning", ""))

        # Validate answer
        if answer and answer[0] in valid:
            answer = answer[0]
        else:
            return None

        confidence = normalize_confidence(float(confidence_raw))
        return MCResponse(answer=answer, confidence=confidence, reasoning=reasoning)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def _try_regex_parse(raw: str, valid: list[str]) -> Optional[MCResponse]:
    """Extract answer and confidence using regex patterns."""
    answer = None
    confidence = None
    reasoning = ""

    # Answer patterns (accommodating quotes for invalid JSON)
    answer_patterns = [
        r'["\']?[Aa]nswer["\']?[:\s]*["\']?\(?([A-Da-d])\)?',
        r'["\']?[Cc]hoice["\']?[:\s]*["\']?\(?([A-Da-d])\)?',
        r'["\']?[Ss]elect(?:ed|ion)?["\']?[:\s]*["\']?\(?([A-Da-d])\)?',
        r"^([A-Da-d])\)",  # Just "A)" at start of line
        r"\b(?:The answer is|I choose|My answer is|answer is)\s*\(?([A-Da-d])\)?",
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
            match = re.match(r"^([A-Da-d])[\s\.\)]*$", line)
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
