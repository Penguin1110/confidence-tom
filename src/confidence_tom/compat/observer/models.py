"""Observer data models for both the legacy ToM and the dynamic benchmark tracks."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    Logic_Error = "Logic_Error"
    Hallucination = "Hallucination"
    Tool_Argument_Error = "Tool_Argument_Error"
    Observation_Ignored = "Observation_Ignored"
    None_ = "None"


class JudgmentOutput(BaseModel):
    """Structured judgment output for the modern dynamic benchmark observer."""

    judge_reasoning: str = Field(
        description=(
            "Step-by-step CoT analysis of the Worker's reasoning, answer quality, "
            "and confidence signals. Be specific: quote relevant parts of the trajectory."
        )
    )
    predicted_correctness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Manager's estimate of the probability that the Worker's final answer is "
            "correct (0.0 = certainly wrong, 1.0 = certainly correct)."
        ),
    )
    predicted_worker_confidence: float = Field(
        ge=0.0,
        le=1.0,
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
        ge=0.0,
        le=1.0,
        description=(
            "Manager's own confidence in this evaluation (0.0 = very uncertain about "
            "the judgment, 1.0 = highly certain). Low when the Worker trajectory is "
            "ambiguous or truncated."
        ),
    )


class ObserverSelfSolve(BaseModel):
    """The Observer's own attempt at solving the question, used for legacy Protocol P2."""

    reasoning: str = Field(description="Your step-by-step reasoning.")
    answer: str = Field(description="Your final answer.")
    confidence: int = Field(description="Your confidence in this answer (0-100).")


class CanonicalizedSubjectOutput(BaseModel):
    """Canonicalized version of the subject's output, stripped of stylistic bias/tone."""

    canonical_reasoning: str = Field(
        description="The logical reasoning steps, stripped of tone and confidence words."
    )
    canonical_answer: str = Field(description="The raw final answer without extra fluff.")


class ObserverFrameCheckSelfSolve(BaseModel):
    """Observer's Frame-Check then Self-Solve attempt used for Protocol P2+."""

    epistemic_frame: str = Field(
        description="Identify the frame of the question: 'real-world' or 'in-universe'."
    )
    frame_analysis: str = Field(
        description="Why you classified the frame this way, and any tricky premises found."
    )
    reasoning: str = Field(description="Your step-by-step reasoning under this frame.")
    answer: str = Field(description="Your final answer.")
    confidence: int = Field(description="Your confidence in this answer (0-100).")


class TrapDeclaration(BaseModel):
    """Observer's pre-analysis of potential logical traps BEFORE seeing the subject's answer."""

    question_summary: str = Field(description="Brief summary of what the question is asking.")
    difficulty_assessment: str = Field(
        description="How difficult is this question for a 20B-parameter model? (Easy/Medium/Hard)"
    )
    potential_traps: list[str] = Field(
        description=(
            "List 2-4 specific logical traps or common mistakes a 20B model might fall into."
        )
    )
    success_indicators: list[str] = Field(
        description="What would good reasoning look like? Key markers of correct approach."
    )
    failure_indicators: list[str] = Field(
        description="What would flawed reasoning look like? Red flags to watch for."
    )


class CoTDiagnosis(BaseModel):
    """Detailed diagnosis of the subject's Chain of Thought."""

    fell_into_trap: bool = Field(
        description="Did the subject fall into any of the identified logical traps?"
    )
    trap_details: Optional[str] = Field(
        default=None, description="If fell_into_trap is True, describe which trap and how."
    )
    reasoning_quality: str = Field(
        description=(
            "Overall quality of reasoning: 'Sound', 'Flawed but functional', 'Fundamentally broken'"
        )
    )
    luck_factor: bool = Field(
        description="Is there evidence of 'lucky guess' - correct answer despite flawed reasoning?"
    )
    luck_explanation: Optional[str] = Field(
        default=None, description="If luck_factor is True, explain how the subject got lucky."
    )
    key_errors: list[str] = Field(
        default_factory=list, description="List of specific logical errors found in the CoT."
    )
    key_strengths: list[str] = Field(
        default_factory=list, description="List of specific strengths in the CoT."
    )


class EnhancedJudgmentOutput(BaseModel):
    """Enhanced judgment output with structured diagnosis for Frame-Aware Observer."""

    diagnosis: CoTDiagnosis = Field(
        description="Detailed diagnosis of the subject's reasoning process."
    )
    predicted_confidence: int = Field(
        description=(
            "Predicted probability (0-100) that the subject would answer correctly "
            "if asked this question 10 times independently."
        )
    )
    confidence_reasoning: str = Field(
        description=(
            "Explain how you arrived at this confidence prediction, accounting for "
            "reasoning quality, luck factors, and trap susceptibility."
        )
    )
    is_overconfident: bool = Field(
        description=(
            "True if subject's self-reported confidence exceeds predicted confidence significantly."
        )
    )
    is_underconfident: bool = Field(
        description=(
            "True if subject's self-reported confidence is much lower than their actual capability."
        )
    )


class LegacyJudgmentOutput(BaseModel):
    """Legacy observer judgment used by the original recursive ToM experiments."""

    predicted_confidence: int = Field(
        description=(
            "Predicted probability (0-100) that the subject would answer correctly "
            "if asked this question 10 times independently."
        )
    )
    is_overconfident: bool = Field(
        description=(
            "True if the subject's self-reported confidence seems higher than "
            "warranted by their reasoning quality."
        )
    )
    reasoning: str = Field(
        description="Detailed explanation of why you assigned this confidence score."
    )


class RecursiveLevelResult(BaseModel):
    """A record of a single level in the recursive observer process."""

    level: int
    observer_model: str
    protocol: str
    observer_group: str = Field(default="unknown")
    judgment: JudgmentOutput | LegacyJudgmentOutput | EnhancedJudgmentOutput | dict[str, Any]
    trap_declaration: Optional[TrapDeclaration] = Field(
        default=None, description="Only populated for Frame-Aware (Group C) observers."
    )
