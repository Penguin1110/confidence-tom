from pydantic import BaseModel, Field


class SolvedItem(BaseModel):
    """The original raw output from the Subject Model (Level-0 generator)."""

    question_id: str
    question: str
    cot: str = Field(description="The Chain of Thought reasoning process.")
    answer: str = Field(description="The final answer derived from the CoT.")
    true_confidence: int = Field(description="Self-assessed confidence (0-100%).")


class StyledItem(BaseModel):
    """A restyled version of the SolvedItem for testing observer drift."""

    question_id: str
    style_name: str
    styled_cot: str = Field(description="The reasoning restyled to a specific tone.")
    answer: str = Field(description="The final answer (must match the original).")
    true_confidence: int = Field(description="The original true confidence.")
