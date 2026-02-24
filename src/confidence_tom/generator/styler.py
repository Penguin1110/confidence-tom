from pydantic import BaseModel, Field

from confidence_tom.client import LLMClient
from confidence_tom.generator.models import SolvedItem, StyledItem


class StyledOutput(BaseModel):
    confident_cot: str = Field(
        description="The CoT rewritten to sound highly confident and authoritative."
    )
    hesitant_cot: str = Field(
        description="The CoT rewritten to sound doubtful, uncertain, and hesitant."
    )
    neutral_cot: str = Field(description="The CoT rewritten to be purely objective and dry.")


class StyleTransferer:
    """Takes a Subject's reasoning and restyles it to test Observer drift (Protocol Invariance)."""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3) -> None:
        # We enforce a low temperature because we want consistent semantic preservation
        self.client = LLMClient(model=model_name, temperature=temperature, max_tokens=2048)
        self.system_prompt = (
            "You are an expert editor participating in a psychological experiment.\n"
            "Your task is to rewrite a given Chain of Thought (CoT) reasoning process into "
            "THREE different distinct stylistic tones (Confident, Hesitant, Neutral).\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. DO NOT change the actual logical steps, facts, or the final answer.\n"
            "2. ONLY change the linguistic style, tone, filler words, and subjective markers."
        )

    def restyle(self, solved_item: SolvedItem) -> list[StyledItem] | None:
        """Restyles a given solved item into three distinct versions."""

        user_prompt = (
            f"Question: {solved_item.question}\n"
            f"Original Reasoning: {solved_item.cot}\n"
            f"Final Answer: {solved_item.answer}\n\n"
            "Please rewrite the reasoning into the three requested tones."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        parsed = self.client.generate_parsed(messages, StyledOutput)
        if parsed is None:
            return None

        return [
            StyledItem(
                question_id=solved_item.question_id,
                style_name="confident",
                styled_cot=parsed.confident_cot,
                answer=solved_item.answer,
                true_confidence=solved_item.true_confidence,
            ),
            StyledItem(
                question_id=solved_item.question_id,
                style_name="hesitant",
                styled_cot=parsed.hesitant_cot,
                answer=solved_item.answer,
                true_confidence=solved_item.true_confidence,
            ),
            StyledItem(
                question_id=solved_item.question_id,
                style_name="neutral",
                styled_cot=parsed.neutral_cot,
                answer=solved_item.answer,
                true_confidence=solved_item.true_confidence,
            ),
        ]
