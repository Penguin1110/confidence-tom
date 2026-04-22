from confidence_tom.eval.parsing import extract_answer_candidate, parse_static_response
from confidence_tom.eval.static_evaluators import (
    _normalize_olympiadbench_prediction,
    _normalize_text_answer,
)


def test_extract_answer_candidate_handles_natural_language_tail() -> None:
    text = "Let me think this through. Therefore, the answer is 4\\sqrt{5}."
    assert extract_answer_candidate(text) == "4\\sqrt{5}"


def test_extract_answer_candidate_strips_reasoning_tags_and_boxes() -> None:
    text = "<reasoning>scratch</reasoning> The final answer: \\boxed{3/2}"
    assert extract_answer_candidate(text) == "3/2"


def test_parse_static_response_handles_natural_language_answer() -> None:
    parsed = parse_static_response("We compute carefully. Therefore, the answer is 42.")
    assert parsed is not None
    assert parsed.answer == "42"


def test_normalize_text_answer_handles_natural_language_answer() -> None:
    assert _normalize_text_answer("Hence, the answer is B.") == "b"


def test_normalize_olympiadbench_prediction_handles_natural_language_answer() -> None:
    text = "We compute carefully. Therefore, the answer is 4\\sqrt{5}."
    assert _normalize_olympiadbench_prediction(text, "4\\sqrt{5}") == "4\\sqrt{5}"
