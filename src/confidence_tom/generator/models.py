from typing import List

from pydantic import BaseModel, Field


class SolvedInstance(BaseModel):
    """A single generation sample from the subject model."""

    cot: str = Field(description="The Chain of Thought reasoning process.")
    answer: str = Field(
        description="The final short answer derived from the CoT. Must be a clear string/category."
    )
    reported_confidence: int = Field(description="Self-assessed verbal confidence (0-100%).")


class SubjectOutputV2(BaseModel):
    """The aggregate output containing behavioral confidence and multi-samples."""

    question_id: str
    question: str
    ambiguity_level: str = Field(description="L0, L1, L2, L3...")
    ground_truth: str = Field(description="The actual correct answer (from Verifier/Dataset)")

    samples: List[SolvedInstance] = Field(description="K samples generated at T>0")

    majority_answer: str = Field(description="The most frequent answer generated.")
    behavioral_confidence: float = Field(
        description="c_beh: fraction of samples matching majority_answer (0.0 - 1.0)."
    )
    avg_reported_confidence: float = Field(
        description="c_rep: average of verbal confidences (0-100)."
    )

    # We select one representative sample of the majority answer as the 'primary' CoT
    # to show observers.
    primary_cot: str = Field(description="The CoT from a sample that yielded the majority answer.")

    # Simple rigorous correctness check (Does majority_answer match ground_truth?)
    is_correct: bool = Field(description="Did the majority answer match the ground truth?")
