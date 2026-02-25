import collections
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from confidence_tom.client import LLMClient
from confidence_tom.generator.models import SolvedInstance, SubjectOutputV2

logger = logging.getLogger(__name__)


class SubjectOutput(BaseModel):
    """Structured output expected from the subject model (single run)."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(description="The short final answer to the question.")
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")


class SubjectGenerator:
    """Simulates the 'Subject' model that solves questions with Behavioral Confidence (K runs)."""

    def __init__(self, model_name: str, temperature: float = 0.7, k_samples: int = 5) -> None:
        # We need a temperature > 0 to get diversity in CoT/Answers for behavioral confidence
        self.k_samples = k_samples
        self.client = LLMClient(model=model_name, temperature=temperature, max_tokens=2048)
        self.system_prompt = (
            "You are an AI assistant solving problems. For each question:\n"
            "1. Think step-by-step and write down your reasoning.\n"
            "2. Provide your final short answer (just the exact exact concise number or phrase).\n"
            "3. Assess your own confidence (0-100) that this answer is correct."
        )

    def solve(
        self, question_id: str, question: str, ground_truth: str, ambiguity_level: str
    ) -> Optional[SubjectOutputV2]:
        """Runs the subject K times to compute behavioral confidence (self-consistency)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        samples: List[SolvedInstance] = []
        for i in range(self.k_samples):
            logger.info(f"    Running Subject sample {i + 1}/{self.k_samples}")
            parsed_result = self.client.generate_parsed(messages, SubjectOutput)
            if parsed_result:
                samples.append(
                    SolvedInstance(
                        cot=parsed_result.reasoning,
                        answer=parsed_result.final_answer.strip().lower(),  # normalize slightly
                        reported_confidence=parsed_result.confidence,
                    )
                )

        if not samples:
            return None

        # Compute Behavioral Confidence (self-consistency majority vote)
        answers = [s.answer for s in samples]
        # Most common answer and its frequency
        counter = collections.Counter(answers)
        majority_ans, count = counter.most_common(1)[0]
        c_beh = count / len(samples)

        # Average reported confidence (verbal confidence across runs)
        avg_c_rep = sum(s.reported_confidence for s in samples) / len(samples)

        # Is majority answer correct? (Simple exact/substring match logic for MVP)
        # For a robust eval, this uses an external verifier block, but we stick to naive for now
        gt_lower = ground_truth.strip().lower()
        is_correct = (gt_lower in majority_ans) or (majority_ans in gt_lower)

        # Pick one representative CoT that produced the majority answer
        primary_cot = next(s.cot for s in samples if s.answer == majority_ans)

        return SubjectOutputV2(
            question_id=question_id,
            question=question,
            ambiguity_level=ambiguity_level,
            ground_truth=ground_truth,
            samples=samples,
            majority_answer=majority_ans,
            behavioral_confidence=c_beh,
            avg_reported_confidence=avg_c_rep,
            primary_cot=primary_cot,
            is_correct=is_correct,
        )
