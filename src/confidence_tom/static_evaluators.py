"""Evaluator interface for single-turn static benchmarks."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from confidence_tom.dataset_models import StaticTask


@dataclass
class EvaluationResult:
    is_correct: bool
    score: float | None = None
    extracted_answer: str = ""
    evaluator_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StaticEvaluator(Protocol):
    def __call__(self, prediction: str, task: StaticTask) -> EvaluationResult: ...


def build_static_evaluator(task: StaticTask) -> StaticEvaluator:
    name = task.evaluator_name
    if name in {"mc_letter", "musr"}:
        return evaluate_multiple_choice
    if name == "olympiadbench":
        return evaluate_olympiadbench
    if name == "livebench_reasoning":
        return evaluate_livebench_reasoning
    if name == "exact_match":
        return evaluate_exact_match
    raise ValueError(f"No static evaluator for '{name}'")


def evaluate_multiple_choice(prediction: str, task: StaticTask) -> EvaluationResult:
    normalized = _normalize_mc_letter(prediction)
    correct = _normalize_mc_letter(task.correct_answer)
    ref = task.reference_answer or task.correct_answer
    if not normalized and ref:
        normalized_pred_text = _normalize_text_answer(prediction)
        normalized_ref_text = _normalize_text_answer(ref)
        is_correct = bool(normalized_pred_text) and normalized_pred_text == normalized_ref_text
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            extracted_answer=prediction.strip(),
            evaluator_name=task.evaluator_name,
            metadata={"reference_answer": ref},
        )
    return EvaluationResult(
        is_correct=bool(normalized and normalized == correct),
        score=1.0 if normalized and normalized == correct else 0.0,
        extracted_answer=normalized or prediction.strip(),
        evaluator_name=task.evaluator_name,
        metadata={"reference_answer": ref},
    )


def evaluate_exact_match(prediction: str, task: StaticTask) -> EvaluationResult:
    normalized_pred = _normalize_text_answer(prediction)
    normalized_ref = _normalize_text_answer(task.reference_answer or task.correct_answer)
    is_correct = bool(normalized_pred) and normalized_pred == normalized_ref
    return EvaluationResult(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        extracted_answer=prediction.strip(),
        evaluator_name=task.evaluator_name,
        metadata={"reference_answer": task.reference_answer or task.correct_answer},
    )


def evaluate_olympiadbench(prediction: str, task: StaticTask) -> EvaluationResult:
    official = _try_official_olympiadbench(prediction, task)
    if official is not None:
        return official

    normalized_pred = _normalize_text_answer(prediction)
    normalized_ref = _normalize_text_answer(task.reference_answer)
    is_correct = bool(normalized_pred) and normalized_pred == normalized_ref
    return EvaluationResult(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        extracted_answer=prediction.strip(),
        evaluator_name="olympiadbench_fallback",
        metadata={
            "reference_answer": task.reference_answer,
            "answer_type": task.metadata.get("answer_type", ""),
            "used_official": False,
        },
    )


def evaluate_livebench_reasoning(prediction: str, task: StaticTask) -> EvaluationResult:
    official = _try_official_livebench(prediction, task)
    if official is not None:
        return official

    normalized_pred = _normalize_text_answer(prediction)
    normalized_ref = _normalize_text_answer(task.reference_answer)
    is_correct = bool(normalized_pred) and normalized_pred == normalized_ref
    return EvaluationResult(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        extracted_answer=prediction.strip(),
        evaluator_name="livebench_reasoning_fallback",
        metadata={
            "reference_answer": task.reference_answer,
            "task": task.metadata.get("task", ""),
            "used_official": False,
        },
    )


def _try_official_olympiadbench(prediction: str, task: StaticTask) -> EvaluationResult | None:
    """Call official scorer when the package/repo is installed locally."""
    isolated = _try_isolated_olympiadbench(prediction, task)
    if isolated is not None:
        return isolated

    judge_cls = _load_olympiadbench_judger()
    if judge_cls is None:
        return None

    try:
        judger = judge_cls()
        precision = task.metadata.get("precision", 1e-8)
        score = judger.judge(task.reference_answer, prediction, precision)
    except Exception:
        return None

    return EvaluationResult(
        is_correct=bool(score),
        score=float(score),
        extracted_answer=prediction.strip(),
        evaluator_name="olympiadbench_official",
        metadata={"used_official": True},
    )


def _try_official_livebench(prediction: str, task: StaticTask) -> EvaluationResult | None:
    """Call official LiveBench scorer if the package is installed."""
    scorer = _load_livebench_scorer(
        str(task.metadata.get("task", "")),
        str(task.metadata.get("livebench_release_date", "")),
    )
    if scorer is None:
        return None

    try:
        result = scorer(task.reference_answer, prediction, False)
    except Exception:
        return None

    score = float(result.get("score", 0.0)) if isinstance(result, dict) else float(result)
    return EvaluationResult(
        is_correct=score >= 1.0,
        score=score,
        extracted_answer=prediction.strip(),
        evaluator_name="livebench_official",
        metadata={"used_official": True},
    )


def _normalize_mc_letter(text: str) -> str:
    match = re.search(r"\b([A-J])\b", text.strip().upper())
    return match.group(1) if match else ""


def _normalize_text_answer(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", normalized)
    normalized = re.sub(r"[^a-z0-9,.\-_/=\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _load_olympiadbench_judger() -> type[Any] | None:
    """Load OlympiadBench official MathJudger from installed package or local clone."""
    try:
        module = importlib.import_module("math_judger")
        return getattr(module, "MathJudger", None)
    except Exception:
        pass

    for path in _candidate_repo_paths("OlympiadBench", "inference/code"):
        module = _import_from_path("math_judger", path)
        if module is not None:
            return getattr(module, "MathJudger", None)
    return None


def _try_isolated_olympiadbench(prediction: str, task: StaticTask) -> EvaluationResult | None:
    root = Path.cwd()
    scorer_script = root / "tools" / "score_olympiadbench.py"
    env_python = root / ".venvs" / "olympiadbench-eval" / "bin" / "python"
    if not scorer_script.exists() or not env_python.exists():
        return None

    precision = str(task.metadata.get("precision", "1e-8"))
    try:
        completed = subprocess.run(
            [
                str(env_python),
                str(scorer_script),
                "--reference",
                task.reference_answer,
                "--prediction",
                prediction,
                "--precision",
                precision,
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(root),
        )
        payload = completed.stdout.strip()
        if not payload:
            return None
        data = importlib.import_module("json").loads(payload)
        score = float(data.get("score", 0.0))
        return EvaluationResult(
            is_correct=bool(data.get("is_correct", False)),
            score=score,
            extracted_answer=prediction.strip(),
            evaluator_name="olympiadbench_official_isolated",
            metadata={"used_official": True, "isolated_env": True},
        )
    except Exception:
        return None


def _load_livebench_scorer(task_name: str, release_date: str = "") -> Callable[..., Any] | None:
    """Load official LiveBench scorer for supported reasoning tasks."""
    module_map = {
        "web_of_lies_v2": (
            "livebench.process_results.reasoning.web_of_lies_v2.utils",
            "web_of_lies_process_results",
        ),
        "spatial": (
            "livebench.process_results.reasoning.spatial.utils",
            "spatial_process_results",
        ),
        "zebra_puzzle": (
            "livebench.process_results.reasoning.zebra_puzzle.utils",
            "get_zebra_puzzle_evaluator",
        ),
    }
    if task_name not in module_map:
        return None

    module_name, attr = module_map[task_name]
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None

    if module is None:
        for path in _candidate_repo_paths("LiveBench", ""):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            try:
                module = importlib.import_module(module_name)
                break
            except Exception:
                module = None
        if module is None:
            return None

    scorer = getattr(module, attr, None)
    if scorer is None:
        return None
    if attr.startswith("get_"):
        return scorer(release_date or "9999-12-31")
    return scorer


def _candidate_repo_paths(repo_name: str, suffix: str) -> list[Path]:
    base_candidates = [
        Path.cwd() / "external" / repo_name,
        Path.cwd() / repo_name,
        Path("/tmp/bench_inspect") / repo_name,
    ]
    result: list[Path] = []
    for base in base_candidates:
        path = base / suffix if suffix else base
        if path.exists():
            result.append(path)
    return result


def _import_from_path(module_name: str, path: Path) -> Any | None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
