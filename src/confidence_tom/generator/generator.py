from pydantic import BaseModel, Field

from confidence_tom.client import LLMClient
from confidence_tom.generator.models import SolvedItem


class SubjectOutput(BaseModel):
    """Structured output expected from the subject model."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(description="The short final answer to the question.")
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")


class SubjectGenerator:
    """Simulates the 'Subject' model (Level-0) that solves questions."""

    def __init__(self, model_name: str, temperature: float = 0.7) -> None:
        self.client = LLMClient(model=model_name, temperature=temperature, max_tokens=2048)
        self.system_prompt = (
            "You are an AI assistant solving problems. For each question:\n"
            "1. Think step-by-step and write down your reasoning.\n"
            "2. Provide your final short answer.\n"
            "3. Assess your own confidence (0-100) that this answer is correct."
        )

    def solve(self, question_id: str, question: str) -> SolvedItem | None:
        """Solves a single question and captures the confidence and reasoning."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        parsed_result = self.client.generate_parsed(messages, SubjectOutput)

        if parsed_result is None:
            return None

        return SolvedItem(
            question_id=question_id,
            question=question,
            cot=parsed_result.reasoning,
            answer=parsed_result.final_answer,
            true_confidence=parsed_result.confidence,
        )
