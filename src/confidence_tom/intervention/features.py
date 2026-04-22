from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

from confidence_tom.intervention.models import (
    InterventionFeatureVector,
    InterventionState,
    StepRecord,
)

_HEDGE_PATTERNS = [
    "i think",
    "maybe",
    "perhaps",
    "it seems",
    "likely",
    "possibly",
    "probably",
    "not sure",
    "unclear",
]

_BACKTRACK_PATTERNS = [
    "go back",
    "back to",
    "revisit",
    "earlier",
    "previous step",
    "step 1",
    "step 2",
    "step 3",
]

_VERIFICATION_CODE = {"none": 0, "partial": 1, "verified": 2, "failed": 3}


def build_state(
    task_id: str, question: str, steps: list[StepRecord], step_index: int
) -> InterventionState:
    prefix = steps[:step_index]
    current = prefix[-1]
    return InterventionState(
        task_id=task_id,
        step_index=step_index,
        total_steps_available=len(steps),
        question=question,
        steps_so_far=prefix,
        current_partial_answer=current.partial_answer,
        current_step_confidence=current.step_confidence / 100.0,
    )


def extract_features(
    state: InterventionState, embedding_window: list[list[float]] | None = None
) -> InterventionFeatureVector:
    steps = state.steps_so_far
    current = steps[-1]
    confidences = [s.step_confidence / 100.0 for s in steps]
    drops = [max(0.0, confidences[i - 1] - confidences[i]) for i in range(1, len(confidences))]
    prev_answers = [s.partial_answer.strip() for s in steps[:-1] if s.partial_answer.strip()]
    current_answer = current.partial_answer.strip()
    answers = [s.partial_answer.strip() for s in steps if s.partial_answer.strip()]

    current_tokens = _tokenize(current.reasoning)
    avg_prev_len = sum(len(_tokenize(s.reasoning)) for s in steps[:-1]) / max(1, len(steps[:-1]))
    token_density_ratio = len(current_tokens) / max(1.0, avg_prev_len)

    semantic_drift = 0.0
    semantic_drift_velocity = 0.0
    if embedding_window and len(embedding_window) >= 2:
        semantic_drift = _dense_cosine_distance(embedding_window[-2], embedding_window[-1])
        semantic_drift_velocity = semantic_drift / max(1, len(current_tokens))
    elif len(steps) >= 2:
        semantic_drift = _cosine_distance(_bow(steps[-2].reasoning), _bow(current.reasoning))
        semantic_drift_velocity = semantic_drift / max(1, len(current_tokens))

    if embedding_window and len(embedding_window) >= 2:
        window_variance = _dense_window_variance(embedding_window[-3:])
    else:
        window_texts = [s.reasoning for s in steps[-3:]]
        window_variance = _window_variance(window_texts)

    return InterventionFeatureVector(
        task_id=state.task_id,
        step_index=state.step_index,
        current_step_confidence=state.current_step_confidence,
        confidence_delta=(confidences[-1] - confidences[-2]) if len(confidences) >= 2 else 0.0,
        max_confidence_drop_so_far=max(drops, default=0.0),
        mean_confidence_drop_so_far=(sum(drops) / len(drops)) if drops else 0.0,
        num_confidence_drops=sum(1 for d in drops if d > 0),
        partial_answer_changed=int(
            bool(prev_answers and current_answer and current_answer != prev_answers[-1])
        ),
        num_unique_partial_answers=len(set(answers)),
        self_correction_depth=_self_correction_depth(prev_answers[-1], current_answer)
        if prev_answers and current_answer
        else 0.0,
        backtracking_flag=int(_has_backtracking(current.reasoning)),
        reasoning_length=len(current_tokens),
        token_density_ratio=token_density_ratio,
        hedge_density=_hedge_density(current.reasoning),
        uncertainty_flag=int(
            bool(current.uncertainty_note.strip()) or _hedge_density(current.reasoning) > 0.02
        ),
        assumptions_count=len(current.assumptions),
        verification_status_code=_VERIFICATION_CODE.get(current.verification_status, 0),
        semantic_drift=semantic_drift,
        semantic_drift_velocity=semantic_drift_velocity,
        window_variance=window_variance,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _bow(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_distance(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine = dot / (norm_a * norm_b)
    return 1.0 - cosine


def _window_variance(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    bows = [_bow(t) for t in texts]
    dists: list[float] = []
    for i in range(1, len(bows)):
        dists.append(_cosine_distance(bows[i - 1], bows[i]))
    mean = sum(dists) / len(dists)
    return sum((d - mean) ** 2 for d in dists) / len(dists)


def _hedge_density(text: str) -> float:
    lowered = text.lower()
    hits = sum(lowered.count(pattern) for pattern in _HEDGE_PATTERNS)
    token_count = max(1, len(_tokenize(text)))
    return hits / token_count


def _has_backtracking(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _BACKTRACK_PATTERNS)


def _self_correction_depth(prev_answer: str, current_answer: str) -> float:
    if not prev_answer or not current_answer:
        return 0.0
    return 1.0 - SequenceMatcher(a=prev_answer, b=current_answer).ratio()


def _dense_cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def _dense_window_variance(vectors: list[list[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    dists = [_dense_cosine_distance(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]
    mean = sum(dists) / len(dists)
    return sum((d - mean) ** 2 for d in dists) / len(dists)
