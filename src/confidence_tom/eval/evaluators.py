"""Shared benchmark evaluators for dynamic tasks.

These evaluators are heuristic because the current generator does not execute
inside the original benchmark environments. The goal is to reduce both obvious
false positives and false negatives while keeping the logic deterministic.
"""

from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path
from typing import Callable

from confidence_tom.data.task_models import DynamicTask

BenchmarkEvaluator = Callable[[str, DynamicTask, str], bool]

_NEGATION_PATTERNS = (
    "cannot craft",
    "can't craft",
    "could not craft",
    "unable to craft",
    "insufficient",
    "missing",
    "not enough",
    "task incomplete",
    "failed to craft",
)

_OPENING_FENCE_RE = re.compile(r"^\s*```(?:sql|sqlite)?\s*", re.IGNORECASE)
_CLOSING_FENCE_RE = re.compile(r"\s*```\s*$")


def build_evaluator(benchmark: str) -> BenchmarkEvaluator:
    """Return the evaluator for a benchmark."""
    if benchmark == "tau_bench":
        return evaluate_tau_bench
    if benchmark == "bird_sql":
        return evaluate_bird_sql
    if benchmark == "plancraft":
        return evaluate_plancraft
    if benchmark == "intercode":
        return evaluate_intercode
    raise ValueError(f"No evaluator for benchmark '{benchmark}'")


def evaluate_intercode(final_answer: str, task: DynamicTask, evidence_text: str = "") -> bool:
    """Basic string containment evaluator for InterCode."""
    gold = str(task.ground_truth).strip().lower()
    haystack = _normalize_text(f"{final_answer}\n{evidence_text}")
    return bool(gold) and gold in haystack


def evaluate_bird_sql(final_answer: str, task: DynamicTask, evidence_text: str = "") -> bool:
    """Execution-match evaluator with light SQL extraction."""
    from confidence_tom.benchmarks.bird_sql import evaluate_sql

    db_path = str(task.metadata.get("db_path", "")).strip()
    if not db_path or not Path(db_path).exists():
        return False

    candidate_sql = extract_sql(final_answer)
    if not candidate_sql:
        candidate_sql = extract_sql(evidence_text)
    if not candidate_sql:
        return False

    return evaluate_sql(candidate_sql, str(task.ground_truth), db_path)


def evaluate_tau_bench(final_answer: str, task: DynamicTask, evidence_text: str = "") -> bool:
    """Heuristic tau-bench evaluator over the full generated evidence."""
    haystack = _normalize_text(f"{final_answer}\n{evidence_text}")
    actions = task.ground_truth.get("actions", []) if isinstance(task.ground_truth, dict) else []
    outputs = task.ground_truth.get("outputs", []) if isinstance(task.ground_truth, dict) else []

    if outputs:
        normalized_outputs = [_normalize_output_token(str(output)) for output in outputs]
        if not all(token and token in haystack for token in normalized_outputs):
            return False

    if not actions:
        return bool(outputs)

    scored_actions = 0
    for action in actions:
        if _tau_action_supported(action, haystack):
            scored_actions += 1

    threshold = max(1, math.ceil(len(actions) * 0.6))
    if outputs:
        threshold = max(1, math.ceil(len(actions) * 0.4))
    return scored_actions >= threshold


def evaluate_plancraft(final_answer: str, task: DynamicTask, evidence_text: str = "") -> bool:
    """Check target match, contradictions, and craftability from inventory."""
    target = str(task.metadata.get("target_item", "")).strip().lower()
    if not target:
        target = _parse_plancraft_target(task.instruction)
    if not target:
        return False

    final_answer_low = final_answer.strip().lower()
    if final_answer_low != target:
        return False

    haystack = _normalize_text(f"{final_answer}\n{evidence_text}")
    if any(pattern in haystack for pattern in _NEGATION_PATTERNS):
        return False

    if any(term in haystack for term in ("gather ", "harvest ", "mine ", "loot ", "buy ")):
        return False

    inventory = task.metadata.get("initial_inventory")
    if not isinstance(inventory, dict):
        inventory = _parse_plancraft_inventory(task.instruction)
    if not inventory:
        return False

    return _can_reach_item(target, {str(k): int(v) for k, v in inventory.items() if int(v) > 0})


def extract_sql(text: str) -> str:
    """Extract a single SQL statement from model output."""
    if not text:
        return ""

    cleaned = text.strip()
    opening = _OPENING_FENCE_RE.match(cleaned)
    if opening:
        cleaned = cleaned[opening.end() :]
        closing = cleaned.find("```")
        if closing >= 0:
            cleaned = cleaned[:closing]
    cleaned = _CLOSING_FENCE_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    lowered = cleaned.lower()
    prefixes = ("sql query:", "sql:", "query:")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break

    match = re.search(
        r"(?is)\b(select|with|pragma|insert|update|delete)\b.*?(?:;|$)",
        cleaned,
    )
    if not match:
        return ""
    return match.group(0).strip().rstrip(";") + (";" if ";" in match.group(0) else "")


def _tau_action_supported(action: dict[str, object], haystack: str) -> bool:
    name = _normalize_text(str(action.get("name", "")))
    if not name:
        return False

    alias = name.replace("_", " ")
    if alias in haystack or name in haystack:
        return True

    kwargs = action.get("kwargs", {})
    if not isinstance(kwargs, dict):
        return False

    salient_values = []
    for value in kwargs.values():
        if isinstance(value, list):
            salient_values.extend(_flatten_scalars(value))
        else:
            salient_values.extend(_flatten_scalars([value]))

    salient_values = [token for token in salient_values if _is_salient_tau_token(token)]
    if not salient_values:
        return False

    matched_values = sum(
        1 for token in salient_values if _normalize_output_token(token) in haystack
    )
    return matched_values >= max(1, math.ceil(len(salient_values) * 0.5))


def _flatten_scalars(values: list[object]) -> list[str]:
    flattened: list[str] = []
    for value in values:
        if isinstance(value, (str, int, float)):
            flattened.append(str(value))
        elif isinstance(value, dict):
            flattened.extend(_flatten_scalars(list(value.values())))
    return flattened


def _is_salient_tau_token(token: str) -> bool:
    token = token.strip()
    if len(token) < 2:
        return False
    if token.lower() in {"true", "false", "none"}:
        return False
    return any(ch.isdigit() for ch in token) or "_" in token or "-" in token


def _normalize_output_token(text: str) -> str:
    return _normalize_text(text).replace(",", "")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_.#\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_plancraft_target(instruction: str) -> str:
    match = re.search(r"Craft the following item:\s*([a-z0-9_]+)", instruction, re.IGNORECASE)
    return match.group(1).strip().lower() if match else ""


def _parse_plancraft_inventory(instruction: str) -> dict[str, int]:
    match = re.search(
        r"Your current inventory:\s*(.+?)\.\s*List the crafting steps", instruction, re.DOTALL
    )
    if not match:
        return {}
    inventory_text = match.group(1).strip()
    if inventory_text.lower() == "empty":
        return {}

    inventory: dict[str, int] = {}
    for part in inventory_text.split(","):
        item_match = re.match(r"\s*([a-z0-9_]+)\s+x(\d+)\s*$", part.strip(), re.IGNORECASE)
        if not item_match:
            continue
        inventory[item_match.group(1).lower()] = int(item_match.group(2))
    return inventory


def _can_reach_item(target: str, inventory: dict[str, int], max_states: int = 5000) -> bool:
    """Breadth-first search over reachable inventories using Plancraft recipes."""
    if inventory.get(target, 0) > 0:
        return True

    try:
        from plancraft.environment.recipes import RECIPES
    except Exception:
        return False

    queue: deque[dict[str, int]] = deque([dict(inventory)])
    seen = {_inventory_key(inventory)}
    explored = 0

    while queue and explored < max_states:
        state = queue.popleft()
        explored += 1
        if state.get(target, 0) > 0:
            return True

        for recipes in RECIPES.values():
            for recipe in recipes:
                try:
                    if not recipe.can_craft_from_inventory(state):
                        continue
                    next_state = recipe.craft_from_inventory(state)
                except Exception:
                    continue
                if not next_state:
                    continue
                key = _inventory_key(next_state)
                if key in seen:
                    continue
                seen.add(key)
                queue.append(next_state)

    return False


def _inventory_key(inventory: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((item, qty) for item, qty in inventory.items() if qty > 0))
