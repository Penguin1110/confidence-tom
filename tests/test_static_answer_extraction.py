from confidence_tom.eval.parsing import extract_answer_candidate, parse_static_response
from confidence_tom.eval.static_evaluators import (
    evaluate_math_exact,
    _normalize_olympiadbench_prediction,
    _normalize_math_prediction,
    _normalize_text_answer,
)
from confidence_tom.data.dataset_models import StaticTask


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


def test_normalize_math_prediction_strips_surface_variants() -> None:
    assert _normalize_math_prediction(r"\boxed{42.0}", "42") == "42"
    assert _normalize_math_prediction(r"\left( 3, \frac{1}{2} \right)", r"(3,\frac{1}{2})") == "(3,1/2)"


def test_evaluate_math_exact_accepts_boxed_integer_variant() -> None:
    task = StaticTask(
        id="aime_1",
        question="q",
        reference_answer="42",
        category="math",
        source="aime_2024",
        answer_format="open_ended",
        evaluator_name="math_exact",
    )
    result = evaluate_math_exact(r"Final Answer: \boxed{42.0}", task)
    assert result.is_correct


def test_evaluate_math_exact_accepts_fraction_decimal_equivalence() -> None:
    task = StaticTask(
        id="math500_1",
        question="q",
        reference_answer="1/2",
        category="math",
        source="math500",
        answer_format="open_ended",
        evaluator_name="math_exact",
    )
    result = evaluate_math_exact("0.5", task)
    assert result.is_correct


def test_evaluate_math_exact_accepts_latex_fraction_equivalence() -> None:
    task = StaticTask(
        id="math500_2",
        question="q",
        reference_answer=r"\frac{3}{4}",
        category="math",
        source="math500",
        answer_format="open_ended",
        evaluator_name="math_exact",
    )
    result = evaluate_math_exact(r"Final Answer: \boxed{0.75}", task)
    assert result.is_correct
