"""Observer (Manager) data models for dynamic benchmark tasks.

The Manager Trace schema is benchmark-agnostic: it accepts any Worker output
(static MCQ, dynamic trajectory, tool-use, etc.) and always emits the same
structured judgment so results across benchmarks can be directly compared.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Taxonomy of failure modes the Manager attributes to the Worker."""

    Logic_Error = "Logic_Error"
    """The Worker's reasoning chain contains a logical inconsistency or wrong inference."""

    Hallucination = "Hallucination"
    """The Worker stated a fact, tool result, or constraint that does not exist."""

    Tool_Argument_Error = "Tool_Argument_Error"
    """The Worker called a tool / API with wrong arguments or misread its output."""

    Observation_Ignored = "Observation_Ignored"
    """The environment returned feedback that the Worker failed to incorporate."""

    None_ = "None"
    """No identifiable error — the Worker's answer appears correct."""


class JudgmentOutput(BaseModel):
    """Unified Manager Trace — the observer's structured judgment on a Worker run.

    Benchmark-agnostic: works for static MCQ, dynamic trajectory, tool-use tasks.
    All float fields are in [0.0, 1.0].
    """

    judge_reasoning: str = Field(
        description=(
            "Step-by-step CoT analysis of the Worker's reasoning, answer quality, "
            "and confidence signals. Be specific: quote relevant parts of the trajectory."
        )
    )
    predicted_correctness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Manager's estimate of the probability that the Worker's final answer is "
            "correct (0.0 = certainly wrong, 1.0 = certainly correct)."
        ),
    )
    predicted_worker_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Manager's assessment of the Worker's *true* confidence level as expressed "
            "through its language and reasoning — not just the self-reported number "
            "(0.0 = highly uncertain, 1.0 = completely certain)."
        ),
    )
    predicted_error_type: ErrorType = Field(
        description=(
            "Precise error attribution. Choose 'None' if the answer looks correct. "
            "Options: Logic_Error | Hallucination | Tool_Argument_Error | "
            "Observation_Ignored | None"
        ),
    )
    manager_self_confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Manager's own confidence in this evaluation (0.0 = very uncertain about "
            "the judgment, 1.0 = highly certain). Low when the Worker trajectory is "
            "ambiguous or truncated."
        ),
    )
