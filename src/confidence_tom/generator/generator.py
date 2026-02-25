import asyncio
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

    def __init__(
        self, model_name: str, temperature: float = 0.7, k_samples: int = 5, max_tokens: int = 4096
    ) -> None:
        self.k_samples = k_samples
        self.client = LLMClient(model=model_name, temperature=temperature, max_tokens=max_tokens)
        self.system_prompt = (
            "You are an AI assistant solving problems. For each question:\n"
            "1. Think step-by-step and write down your reasoning.\n"
            "2. Provide your final short answer (just the exact concise number or phrase).\n"
            "3. Assess your own confidence (0-100) that this answer is correct."
        )

    async def solve(
        self,
        question_id: str,
        question: str,
        ground_truth: str,
        ambiguity_level: str,
        framing: str = "standard",
    ) -> Optional[SubjectOutputV2]:
        """Runs the subject K times to compute behavioral confidence (self-consistency)."""

        framing_instruction = ""
        if framing == "real-world":
            framing_instruction = (
                "Answer truthfully about the real world. If the premise is fictional, "
                "say so explicitly."
            )
        elif framing == "in-universe":
            framing_instruction = (
                "Answer strictly within the fictional/folklore premise provided by the question."
            )

        system_content = self.system_prompt
        if framing_instruction:
            system_content += f"\n\n[CRITICAL DIRECTIVE]: {framing_instruction}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}"},
        ]

        async def fetch_sample(i: int) -> Optional[SubjectOutput]:
            logger.info(f"    Running Subject sample {i + 1}/{self.k_samples} [{framing}]")
            return await self.client.agenerate_parsed(messages, SubjectOutput)

        tasks = [fetch_sample(i) for i in range(self.k_samples)]
        results = await asyncio.gather(*tasks)

        samples: List[SolvedInstance] = []
        for parsed_result in results:
            if parsed_result:
                samples.append(
                    SolvedInstance(
                        cot=parsed_result.reasoning,
                        answer=parsed_result.final_answer.strip().lower(),
                        reported_confidence=parsed_result.confidence,
                    )
                )

        if not samples:
            return None

        # Compute Behavioral Confidence
        answers = [s.answer for s in samples]
        counter = collections.Counter(answers)
        majority_ans, count = counter.most_common(1)[0]
        c_beh = count / len(samples)

        avg_c_rep = sum(s.reported_confidence for s in samples) / len(samples)

        # Is majority answer correct?
        gt_lower = ground_truth.strip().lower()
        is_correct = (gt_lower in majority_ans) or (majority_ans in gt_lower)

        primary_cot = next(s.cot for s in samples if s.answer == majority_ans)

        return SubjectOutputV2(
            question_id=question_id,
            question=question,
            ambiguity_level=ambiguity_level,
            framing=framing,
            ground_truth=ground_truth,
            samples=samples,
            majority_answer=majority_ans,
            behavioral_confidence=c_beh,
            avg_reported_confidence=avg_c_rep,
            primary_cot=primary_cot,
            is_correct=is_correct,
        )
