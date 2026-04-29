from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from confidence_tom.data import scale_dataset
from confidence_tom.data.dataset_models import StaticTask

ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "experiments" / "mainline" / "run" / "core" / "common.py"
ANALYZE_PATH = (
    ROOT / "experiments" / "mainline" / "analysis" / "prefix" / "analyze_prefix_oracle_gain.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_aime_2024_maps_to_math_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"problem": "What is 1+1?", "answer": "2"}]
    monkeypatch.setattr(scale_dataset, "load_dataset", lambda *args, **kwargs: _FakeDataset(rows))
    tasks = scale_dataset.load_aime_2024(num_samples=1)
    assert len(tasks) == 1
    assert tasks[0].evaluator_name == "math_exact"
    assert tasks[0].reference_answer == "2"


def test_load_math500_maps_subject_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "problem": "Find x",
            "answer": "7",
            "subject": "Algebra",
            "level": 5,
            "unique_id": "abc",
        }
    ]
    monkeypatch.setattr(scale_dataset, "load_dataset", lambda *args, **kwargs: _FakeDataset(rows))
    tasks = scale_dataset.load_math500(num_samples=1)
    assert len(tasks) == 1
    assert tasks[0].metadata["subject"] == "Algebra"
    assert tasks[0].metadata["level"] == 5


def test_load_gpqa_diamond_maps_to_multiple_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "Question": "Which is correct?",
            "Correct Answer": "alpha",
            "Incorrect Answer 1": "beta",
            "Incorrect Answer 2": "gamma",
            "Incorrect Answer 3": "delta",
            "Record ID": "r1",
            "Subdomain": "physics",
        }
    ]
    monkeypatch.setattr(scale_dataset, "load_dataset", lambda *args, **kwargs: _FakeDataset(rows))
    tasks = scale_dataset.load_gpqa_diamond(num_samples=1)
    assert len(tasks) == 1
    assert tasks[0].evaluator_name == "mc_letter"
    assert len(tasks[0].choices) == 4
    assert tasks[0].metadata["subcategory"] == "physics"


def test_common_load_static_questions_supports_new_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_common = _load_module(COMMON_PATH, "core_common_for_test")
    fake_task = StaticTask(
        id="x",
        question="q",
        reference_answer="a",
        category="math",
        source="aime_2024",
        answer_format="open_ended",
        evaluator_name="exact_match",
    )
    monkeypatch.setattr(core_common, "load_aime_2024", lambda num_samples: [fake_task])
    cfg = _DictLike({"aime_2024": 1, "limit": 1})
    tasks = core_common.load_static_questions("aime_2024", cfg)
    assert tasks == [fake_task]


def test_analysis_load_tasks_supports_gpqa(monkeypatch: pytest.MonkeyPatch) -> None:
    analyze_prefix = _load_module(ANALYZE_PATH, "analyze_prefix_for_test")
    fake_task = StaticTask(
        id="g",
        question="q",
        choices=["A) x", "B) y"],
        correct_answer="A",
        reference_answer="x",
        category="science",
        source="gpqa_diamond",
        evaluator_name="mc_letter",
    )
    monkeypatch.setattr(analyze_prefix, "load_gpqa_diamond", lambda num_samples: [fake_task])
    dataset_cfg = _DictLike({"benchmark": "gpqa_diamond", "gpqa_diamond": 1, "limit": 1})
    cfg = _Holder(dataset=dataset_cfg)
    tasks = analyze_prefix._load_tasks(cfg)
    assert tasks == {"g": fake_task}


class _FakeDataset(list[dict[str, Any]]):
    def shuffle(self, seed: int) -> "_FakeDataset":
        return self


class _DictLike(dict[str, Any]):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _Holder:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
