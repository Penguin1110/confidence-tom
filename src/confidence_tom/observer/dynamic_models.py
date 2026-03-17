"""Observer data models for dynamic benchmark judgments."""

from enum import Enum

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    Logic_Error = "Logic_Error"
    Hallucination = "Hallucination"
    Tool_Argument_Error = "Tool_Argument_Error"
    Observation_Ignored = "Observation_Ignored"
    None_ = "None"


class JudgmentOutput(BaseModel):
    """Structured manager judgment for dynamic benchmark runs."""

    judge_reasoning: str = Field(
        description=(
            "Step-by-step analysis of the worker's reasoning quality, factual accuracy, "
            "and confidence signals."
        )
    )
    predicted_correctness: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that the worker's final answer is correct.",
    )
    predicted_worker_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated true confidence level expressed by the worker.",
    )
    predicted_error_type: ErrorType = Field(
        description=(
            "Most likely failure mode: Logic_Error, Hallucination, "
            "Tool_Argument_Error, Observation_Ignored, or None."
        )
    )
    manager_self_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Manager's own confidence in this evaluation.",
    )
