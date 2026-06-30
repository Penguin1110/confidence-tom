from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confidence_tom.data.logic_bench import (
    _ensure_minimum_logic_trace_length,
    _iter_prontoqa_examples,
    _normalize_prontoqa_query,
    _render_prontoqa_question,
    load_prontoqa_tasks,
    load_proofwriter_tasks,
)
from confidence_tom.eval.static_evaluators import build_static_evaluator


def test_normalize_prontoqa_query_strips_prove_prefix() -> None:
    assert _normalize_prontoqa_query("Prove: Polly is a yumpus.") == "Polly is a yumpus"


def test_render_prontoqa_question_wraps_context_and_goal() -> None:
    rendered = _render_prontoqa_question("A. B.", "Prove: Polly is a yumpus.")
    assert "Facts and rules:" in rendered
    assert "Goal:" in rendered
    assert rendered.endswith("Prove that Polly is a yumpus.")


def test_iter_prontoqa_examples_flattens_nested_examples() -> None:
    record = {
        "example2": {
            "question": "Ctx 2",
            "query": "Prove: B.",
            "chain_of_thought": ["b1", "b2"],
        },
        "example1": {
            "question": "Ctx 1",
            "query": "Prove: A.",
            "chain_of_thought": ["a1", "a2"],
        },
    }
    flattened = list(_iter_prontoqa_examples([record]))
    assert [item["example_key"] for item in flattened] == ["example1", "example2"]
    assert flattened[0]["chain_of_thought"] == ["a1", "a2"]


def test_ensure_minimum_logic_trace_length_appends_goal_step() -> None:
    steps = _ensure_minimum_logic_trace_length(["a1", "a2"], "Polly is a yumpus")
    assert len(steps) == 3
    assert steps[-1] == "Therefore, the goal is established: Polly is a yumpus."


def test_load_prontoqa_tasks_builds_static_tasks(monkeypatch) -> None:
    fake_records = [
        {
            "example1": {
                "question": "Rule 1. Fact 1.",
                "query": "Prove: Polly is a yumpus.",
                "chain_of_thought": ["Polly is a yumpus and a shumpus.", "Polly is a yumpus."],
            },
            "example2": {
                "question": "Rule 2. Fact 2.",
                "query": "Prove: Rex is a zumpus.",
                "chain_of_thought": ["Rex is a zumpus and a jompus.", "Rex is a zumpus."],
            },
        }
    ]

    monkeypatch.setattr(
        "confidence_tom.data.logic_bench.load_dataset",
        lambda *args, **kwargs: iter(fake_records),
    )

    tasks = load_prontoqa_tasks(num_samples=2, seed=0, max_records=1)
    assert len(tasks) == 2
    assert tasks[0].source == "prontoqa"
    assert tasks[0].metadata["query"].startswith("Prove:")
    assert isinstance(tasks[0].metadata["chain_of_thought"], list)


def test_prontoqa_tasks_use_logic_goal_evaluator(monkeypatch) -> None:
    fake_records = [
        {
            "example1": {
                "question": "Rule 1. Fact 1.",
                "query": "Prove: Polly is a yumpus.",
                "chain_of_thought": ["Polly is a yumpus and a shumpus.", "Polly is a yumpus."],
            }
        }
    ]
    monkeypatch.setattr(
        "confidence_tom.data.logic_bench.load_dataset",
        lambda *args, **kwargs: iter(fake_records),
    )
    task = load_prontoqa_tasks(num_samples=1, seed=0, max_records=1)[0]
    evaluator = build_static_evaluator(task)
    result = evaluator("Review done.\nFinal Answer: Polly is a yumpus", task)
    assert result.is_correct is True


def test_load_proofwriter_tasks_filters_depth_and_unknown(monkeypatch) -> None:
    fake_records = [
        {
            "id": "d1",
            "maxD": 1,
            "NFact": 1,
            "NRule": 1,
            "theory": "Bob is kind. If Bob is kind then Bob is blue.",
            "question": "Bob is blue.",
            "answer": "True",
            "QDep": 1,
            "QLen": 1.0,
            "allProofs": "",
            "config": "x",
        },
        {
            "id": "d2",
            "maxD": 3,
            "NFact": 1,
            "NRule": 2,
            "theory": "Bob is kind. If Bob is kind then Bob is blue. If Bob is blue then Bob is nice.",
            "question": "Bob is nice.",
            "answer": "True",
            "QDep": 2,
            "QLen": 1.0,
            "allProofs": "",
            "config": "x",
        },
        {
            "id": "d3",
            "maxD": 3,
            "NFact": 1,
            "NRule": 2,
            "theory": "Bob is kind.",
            "question": "Bob is quiet.",
            "answer": "Unknown",
            "QDep": 2,
            "QLen": 1.0,
            "allProofs": "",
            "config": "x",
        },
    ]
    monkeypatch.setattr(
        "confidence_tom.data.logic_bench.load_dataset",
        lambda *args, **kwargs: iter(fake_records),
    )
    tasks = load_proofwriter_tasks(num_samples=10, min_depth=2, max_records=10)
    assert len(tasks) == 1
    assert tasks[0].source == "proofwriter"
    assert tasks[0].metadata["raw_id"] == "d2"
    assert tasks[0].reference_answer == "True"
