from pydantic import BaseModel, Field


class ScaleObserverJudgment(BaseModel):
    """Observer's second-order judgment on a subject's answer."""

    predicted_correctness: int = Field(
        description="Probability that the subject's answer is correct (0-100 percent)."
    )
    predicted_subject_confidence: int = Field(
        description="The confidence score you think the subject reported about its own answer (0-100)."
    )
    reasoning: str = Field(description="Step-by-step reasoning for BOTH estimates.")
