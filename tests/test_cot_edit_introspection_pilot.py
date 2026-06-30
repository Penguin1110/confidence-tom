import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.eval.parsing import extract_answer_candidate
from experiments.mainline.run.core.run_cot_edit_introspection_pilot import (
    CheckpointStore,
    SubstrateMaterial,
    TraceOutput,
    TrialRecord,
    _answers_equivalent_by_rule,
    _build_summary,
    _chunk_delta_text,
    _extract_gsm8k_answer,
    _extract_reference_answer_from_solution,
    _is_obviously_dirty_type_c_step,
    _judge_response,
    _load_tasks,
    _logic_type_c_reject_reason,
    _parse_edit_types,
    _parse_elicitations,
    _resolve_role_preset,
    _select_edit_index,
    _trial_key,
)


def test_extract_gsm8k_answer_prefers_final_answer_marker() -> None:
    text = "Some work here.\nFinal Answer: 1,234"
    assert _extract_gsm8k_answer(text) == "1234"


def test_select_edit_index_targets_middle_band() -> None:
    import random

    idx = _select_edit_index(6, random.Random(42))
    assert idx is not None
    assert 1 <= idx <= 4


def test_build_summary_counts_flag_rates() -> None:
    trial = TrialRecord(
        task_id="gsm8k_0000",
        question="q",
        gold_answer="12",
        target_model="openai/gpt-4.1-mini",
        edit_model="openai/gpt-4.1-mini",
        judge_model="openai/gpt-4.1-mini",
        edit_type="A_error",
        elicitation="S1_review",
        edit_index=2,
        original_steps=["a", "b", "c", "d"],
        edited_steps=["a", "b", "x", "d"],
        original_final_answer="12",
        edited_step_text="x",
        model_response="Flagged it.\nFinal Answer: 12",
        extracted_final_answer="12",
        answer_is_correct=True,
        judge_label="FLAG",
        judge_note="explicitly flags",
        mentioned_issue=True,
        localized_step=3,
    )
    summary = _build_summary([trial])
    assert summary["trial_count"] == 1
    assert summary["cells"][0]["flag_rate"] == 1.0
    assert summary["cells"][0]["accuracy"] == 1.0


def test_role_preset_uses_weak_target_and_strong_editor() -> None:
    target_models, edit_model, judge_model = _resolve_role_preset(
        "weak_open_strong_commercial",
        "manual-target",
        "manual-edit",
        "manual-judge",
    )
    assert target_models == "qwen/qwen3.5-27b"
    assert edit_model == "openai/gpt-4.1-mini"
    assert judge_model == "openai/gpt-4.1-mini"


def test_dirty_type_c_heuristic_rejects_self_contradictory_step() -> None:
    step = "In May, she sold half as many clips as in April, which is 48 multiplied by 2 equals 96."
    assert _is_obviously_dirty_type_c_step(step) is True


def test_trial_key_is_stable() -> None:
    key = _trial_key("gsm8k_0000", "qwen/qwen3-8b", "A_error", "S1_review")
    assert key == "gsm8k_0000::qwen/qwen3-8b::A_error::S1_review"


def test_checkpoint_store_writes_summary(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "pilot.json")
    assert store.snapshot_trials() == []
    assert store.snapshot_failures() == []
    assert store.snapshot_substrate_pool() == []


def test_load_tasks_rejects_unknown_benchmark() -> None:
    try:
        _load_tasks("unknown", 1, "train", 42)
    except ValueError as exc:
        assert "Unsupported benchmark" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_tasks_accepts_math_level12_name_via_patch(monkeypatch) -> None:
    from experiments.mainline.run.core import run_cot_edit_introspection_pilot as pilot

    sentinel = [
        StaticTask(
            id="x",
            question="q",
            reference_answer="1",
            category="math",
            source="math_level12",
            answer_format="open_ended",
            evaluator_name="exact_match",
        )
    ]

    monkeypatch.setattr(pilot, "_load_math_level12_tasks", lambda limit, seed: sentinel)
    assert pilot._load_tasks("math_level12", 1, "train", 42) == sentinel


def test_load_tasks_accepts_prontoqa_name_via_patch(monkeypatch) -> None:
    from experiments.mainline.run.core import run_cot_edit_introspection_pilot as pilot

    sentinel = [
        StaticTask(
            id="p1",
            question="q",
            reference_answer="goal",
            category="logic",
            source="prontoqa",
            answer_format="open_ended",
            evaluator_name="logic_goal",
        )
    ]

    monkeypatch.setattr(pilot, "load_prontoqa_tasks", lambda num_samples, seed: sentinel)
    assert pilot._load_tasks("prontoqa", 1, "train", 42) == sentinel


def test_parse_edit_types_subset() -> None:
    assert _parse_edit_types("clean,C_inconsistent") == ["clean", "C_inconsistent"]


def test_parse_elicitations_subset() -> None:
    assert _parse_elicitations("S1_review") == ["S1_review"]


def test_checkpoint_store_roundtrips_substrate_pool(tmp_path: Path) -> None:
    import asyncio

    store = CheckpointStore(tmp_path / "pilot.json")
    material = SubstrateMaterial(
        task_id="t1",
        target_model="qwen/qwen3.5-27b",
        task=StaticTask(
            id="t1",
            question="q",
            reference_answer="1",
            category="math",
            source="gsm8k",
            answer_format="open_ended",
            evaluator_name="exact_match",
        ),
        original_trace=TraceOutput(steps=["a"], final_answer="1"),
    )
    asyncio.run(store.save_substrate_pool([material]))
    reloaded = CheckpointStore(tmp_path / "pilot.json")
    assert len(reloaded.snapshot_substrate_pool()) == 1


def test_extract_answer_candidate_handles_boxed_math_solution() -> None:
    solution = "Some derivation here.\nHence, the answer is \\boxed{-\\frac{24}{25}}."
    assert extract_answer_candidate(solution) == "-\\frac{24}{25}"


def test_extract_reference_answer_from_solution_prefers_last_boxed() -> None:
    solution = "Work here.\n\\[\\det (-3 \\mathbf{A}) = (-3)^2 \\cdot 2 = \\boxed{18}.\\]\n"
    assert _extract_reference_answer_from_solution(solution) == "18"


def test_rule_equivalence_matches_fraction_and_latex_fraction() -> None:
    assert _answers_equivalent_by_rule("-24/25", "-\\frac{24}{25}") is True


def test_rule_equivalence_matches_decimal_and_fraction() -> None:
    assert _answers_equivalent_by_rule("0.09", "\\frac{9}{100}") is True


def test_chunk_delta_text_reads_reasoning_when_content_empty() -> None:
    class Delta:
        content = ""
        reasoning = "hidden-but-useful"

    class Choice:
        delta = Delta()

    class Chunk:
        choices = [Choice()]

    assert _chunk_delta_text(Chunk()) == "hidden-but-useful"


def test_judge_response_marks_empty_response_as_other() -> None:
    import asyncio

    class DummyClient:
        pass

    result, trace, raw = asyncio.run(
        _judge_response(
            DummyClient(),  # type: ignore[arg-type]
            question="q",
            original_steps=["a", "b", "c"],
            edited_steps=["a", "x", "c"],
            edit_type="C_inconsistent",
            elicitation="S1_review",
            edit_index=1,
            response_text="",
        )
    )
    assert result is not None
    assert result.label == "OTHER"
    assert trace["skipped_due_to_empty_model_response"] is True
    assert raw == ""


def test_logic_type_c_reject_reason_blocks_explicit_fact_restatement() -> None:
    task = StaticTask(
        id="p1",
        question="Facts and rules:\nPolly is a yumpus and a shumpus.\n\nGoal:\nProve that Polly is a yumpus.",
        reference_answer="Polly is a yumpus",
        category="logic",
        source="prontoqa",
        answer_format="open_ended",
        evaluator_name="logic_goal",
        metadata={"raw_context": "Polly is a yumpus and a shumpus."},
    )
    reason = _logic_type_c_reject_reason(
        task,
        previous_steps=["Polly is a yumpus and a shumpus."],
        target_step="Polly is a yumpus.",
        candidate_step="Polly is a yumpus and a shumpus.",
    )
    assert reason == "logic_repeats_prior_step" or reason == "logic_restates_explicit_fact"
