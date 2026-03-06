from typing import List, Optional

from pydantic import BaseModel, Field


class ConfidenceJustification(BaseModel):
    """Structured justification for the subject's self-assessed confidence."""

    key_reasoning_steps: List[str] = Field(
        description="The 2-3 most critical reasoning steps that led to the answer."
    )
    uncertainty_factors: List[str] = Field(
        description="Factors that reduce confidence (e.g., ambiguous wording, multiple interpretations)."
    )
    confidence_anchors: List[str] = Field(
        description="Factors that increase confidence (e.g., clear logical chain, verified calculation)."
    )
    potential_errors: List[str] = Field(
        description="Potential mistakes or weak points in the reasoning."
    )


class SolvedInstance(BaseModel):
    """A single generation sample from the subject model."""

    cot: str = Field(description="The Chain of Thought reasoning process.")
    answer: str = Field(
        description="The final short answer derived from the CoT. Must be a clear string/category."
    )
    reported_confidence: int = Field(description="Self-assessed verbal confidence (0-100%).")
    confidence_justification: Optional[ConfidenceJustification] = Field(
        default=None, description="Structured justification for the confidence score."
    )


class SubjectOutputV2(BaseModel):
    """The aggregate output containing behavioral confidence and multi-samples."""

    question_id: str
    question: str
    ambiguity_level: str = Field(description="L1, L3a, L3b, L4...")
    framing: str = Field(description="e.g., 'standard', 'real-world', 'in-universe'")
    ground_truth: str = Field(description="The actual correct answer (from Verifier/Dataset)")
    task_type: str = Field(default="unknown", description="Task type (e.g., 'navigate', 'formal_fallacies')")

    samples: List[SolvedInstance] = Field(description="K samples generated at T>0")
    k_samples: int = Field(default=10, description="Number of independent samples taken")

    majority_answer: str = Field(description="The most frequent answer generated.")
    correct_count: int = Field(default=0, description="Number of samples with correct answer.")
    behavioral_confidence: float = Field(
        description="c_beh: fraction of samples with CORRECT answer (0.0 - 1.0). This is the Ground Truth."
    )
    consistency_rate: float = Field(
        default=0.0, description="Fraction of samples matching majority_answer (self-consistency metric)."
    )
    avg_reported_confidence: float = Field(
        description="c_rep: average of verbal confidences (0-100)."
    )

    primary_cot: str = Field(description="The CoT from a sample that yielded the majority answer.")
    is_correct: bool = Field(description="Did the majority answer match the ground truth?")
