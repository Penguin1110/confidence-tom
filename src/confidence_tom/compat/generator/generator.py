"""Legacy subject generator used by the original ToM experiments."""

from __future__ import annotations

import asyncio
import collections
import logging
import re
from typing import List, Optional, cast

from pydantic import BaseModel, Field

from confidence_tom.compat.generator.models import (
    ConfidenceJustification,
    SolvedInstance,
    SubjectOutputV2,
)
from confidence_tom.infra.client import LLMClient

logger = logging.getLogger(__name__)


def extract_choice(answer: str) -> str:
    """Extract A/B/C/D choice letter from answer string."""
    answer = answer.strip().upper()

    if answer in ["A", "B", "C", "D"]:
        return answer.lower()

    patterns = [
        r"^([ABCD])\s*[\).:]",  # A. or A) or A:
        r"^\(([ABCD])\)",  # (A)
        r"^(?:option|answer|choice)[:\s]*([ABCD])\b",  # Option A, Answer: A
        r"^([ABCD])\b",  # Just A at start
        r"\b([ABCD])\s*$",  # A at end
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    match = re.search(r"\b([ABCD])\b", answer, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return ""


class SubjectOutput(BaseModel):
    """Structured output expected from the subject model (single run)."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(
        description="The answer choice: must be exactly one of A, B, C, or D."
    )
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")


class SubjectOutputWithJustification(BaseModel):
    """Structured output with explicit confidence justification."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(
        description="The answer choice: must be exactly one of A, B, C, or D."
    )
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")
    key_reasoning_steps: List[str] = Field(
        description="The 2-3 most critical reasoning steps that led to your answer."
    )
    uncertainty_factors: List[str] = Field(
        description="Factors that reduce your confidence (e.g., ambiguous wording, guessing)."
    )
    confidence_anchors: List[str] = Field(
        description=(
            "Factors that increase your confidence (e.g., clear logic, verified calculation)."
        )
    )
    potential_errors: List[str] = Field(
        description="Potential mistakes or weak points in your reasoning."
    )


class SubjectGenerator:
    """Simulates the 'Subject' model that solves questions with Behavioral Confidence."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        k_samples: int = 10,
        max_tokens: int = 4096,
        require_justification: bool = True,
    ) -> None:
        self.k_samples = k_samples
        self.require_justification = require_justification
        self.client = LLMClient(model=model_name, temperature=temperature, max_tokens=max_tokens)

        self.system_prompt = (
            "You are an AI assistant solving multiple-choice questions. For each question:\n"
            "1. Think step-by-step and write down your reasoning.\n"
            "2. Provide your final answer as EXACTLY one letter: A, B, C, or D (nothing else).\n"
            "3. Assess your confidence (0-100): "
            "If you answered this same question 100 times independently, "
            "what percentage would you expect to get correct? "
            "(0 = pure random guess, 25 = one of four choices, "
            "50 = coin flip, 100 = absolutely certain)\n"
        )

        if require_justification:
            self.system_prompt += (
                "4. Explicitly justify your confidence by identifying:\n"
                "   - Key reasoning steps that led to your answer\n"
                "   - Factors that reduce your confidence (uncertainties)\n"
                "   - Factors that increase your confidence (anchors)\n"
                "   - Potential errors or weak points in your reasoning"
            )

    async def solve(
        self,
        question_id: str,
        question: str,
        ground_truth: str,
        ambiguity_level: str,
        framing: str = "standard",
        task_type: str = "unknown",
    ) -> Optional[SubjectOutputV2]:
        """Run the subject K times to compute behavioral confidence."""

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

        response_model = (
            SubjectOutputWithJustification if self.require_justification else SubjectOutput
        )

        RATE_LIMIT_DELAY = 4.5
        results = []
        for i in range(self.k_samples):
            logger.info(f"    Running Subject sample {i + 1}/{self.k_samples} [{framing}]")
            result = await self.client.agenerate_parsed(messages, response_model)
            results.append(result)
            if i < self.k_samples - 1:
                await asyncio.sleep(RATE_LIMIT_DELAY)

        samples: List[SolvedInstance] = []
        gt_choice = extract_choice(ground_truth)

        for parsed_result in results:
            if parsed_result:
                parsed_output = cast(SubjectOutput | SubjectOutputWithJustification, parsed_result)
                answer_choice = extract_choice(parsed_output.final_answer)

                justification = None
                if self.require_justification and isinstance(
                    parsed_output, SubjectOutputWithJustification
                ):
                    justification = ConfidenceJustification(
                        key_reasoning_steps=parsed_output.key_reasoning_steps,
                        uncertainty_factors=parsed_output.uncertainty_factors,
                        confidence_anchors=parsed_output.confidence_anchors,
                        potential_errors=parsed_output.potential_errors,
                    )

                samples.append(
                    SolvedInstance(
                        cot=parsed_output.reasoning,
                        answer=answer_choice,
                        reported_confidence=parsed_output.confidence,
                        confidence_justification=justification,
                    )
                )

        if not samples:
            return None

        correct_count = sum(1 for s in samples if s.answer and s.answer == gt_choice)
        c_beh = correct_count / len(samples)

        answers = [s.answer for s in samples]
        counter = collections.Counter(answers)
        majority_ans, majority_count = counter.most_common(1)[0]
        consistency_rate = majority_count / len(samples)
        avg_c_rep = sum(s.reported_confidence for s in samples) / len(samples)
        is_correct = majority_ans == gt_choice
        primary_cot = next(s.cot for s in samples if s.answer == majority_ans)

        return SubjectOutputV2(
            question_id=question_id,
            question=question,
            ambiguity_level=ambiguity_level,
            framing=framing,
            ground_truth=ground_truth,
            task_type=task_type,
            samples=samples,
            k_samples=len(samples),
            majority_answer=majority_ans,
            correct_count=correct_count,
            behavioral_confidence=c_beh,
            consistency_rate=consistency_rate,
            avg_reported_confidence=avg_c_rep,
            primary_cot=primary_cot,
            is_correct=is_correct,
        )
