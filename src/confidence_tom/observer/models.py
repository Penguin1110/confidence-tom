from pydantic import BaseModel, Field


class ObserverSelfSolve(BaseModel):
    """The Observer's own attempt at solving the question, used for Protocol P2."""

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
