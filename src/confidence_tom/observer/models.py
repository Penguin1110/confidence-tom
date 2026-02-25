from pydantic import BaseModel, Field


class JudgmentOutput(BaseModel):
    """The structured judgment output from the Observer model."""

    predicted_confidence: int = Field(
        description="Your predicted confidence score of the subject (0-100). "
        "How confident do they seem?"
    )
    is_overconfident: bool = Field(
        description="True if you believe the subject is overconfident given its "
        "reasoning and output, False otherwise."
    )
    reasoning: str = Field(
        description="Why you assigned this confidence score and overconfidence judgment."
    )


class RecursiveLevelResult(BaseModel):
    """A record of a single level in the recursive observer process."""

    level: int
    observer_model: str
    protocol: str
    judgment: JudgmentOutput
