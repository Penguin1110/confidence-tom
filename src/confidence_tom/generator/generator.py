import asyncio
import collections
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from confidence_tom.client import LLMClient
from confidence_tom.generator.models import (
    ConfidenceJustification,
    SolvedInstance,
    SubjectOutputV2,
)

logger = logging.getLogger(__name__)


def extract_choice(answer: str) -> str:
    """
    Extract A/B/C/D choice letter from answer string.
    Returns lowercase letter or empty string if not found.
    """
    answer = answer.strip().upper()
    
    # Direct single letter
    if answer in ['A', 'B', 'C', 'D']:
        return answer.lower()
    
    # Patterns like "A.", "A)", "(A)", "Option A", "Answer: A"
    patterns = [
        r'^([ABCD])\s*[\).:]',      # A. or A) or A:
        r'^\(([ABCD])\)',            # (A)
        r'^(?:option|answer|choice)[:\s]*([ABCD])\b',  # Option A, Answer: A
        r'^([ABCD])\b',              # Just A at start
        r'\b([ABCD])\s*$',           # A at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Last resort: find any standalone A/B/C/D
    match = re.search(r'\b([ABCD])\b', answer, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    return ''


class SubjectOutput(BaseModel):
    """Structured output expected from the subject model (single run)."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(description="The answer choice: must be exactly one of A, B, C, or D.")
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")


class SubjectOutputWithJustification(BaseModel):
    """Structured output with explicit confidence justification."""

    reasoning: str = Field(description="Step by step chain of thought reasoning.")
    final_answer: str = Field(description="The answer choice: must be exactly one of A, B, C, or D.")
    confidence: int = Field(description="Self-assessed confidence in the answer from 0 to 100.")
    key_reasoning_steps: List[str] = Field(
        description="The 2-3 most critical reasoning steps that led to your answer."
    )
    uncertainty_factors: List[str] = Field(
        description="Factors that reduce your confidence (e.g., ambiguous wording, guessing)."
    )
    confidence_anchors: List[str] = Field(
        description="Factors that increase your confidence (e.g., clear logic, verified calculation)."
    )
    potential_errors: List[str] = Field(
        description="Potential mistakes or weak points in your reasoning."
    )


class SubjectGenerator:
    """Simulates the 'Subject' model that solves questions with Behavioral Confidence (K runs)."""

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
        
        # Enhanced system prompt for structured confidence explanation
        self.system_prompt = (
            "You are an AI assistant solving multiple-choice questions. For each question:\n"
            "1. Think step-by-step and write down your reasoning.\n"
            "2. Provide your final answer as EXACTLY one letter: A, B, C, or D (nothing else).\n"
            "3. Assess your confidence (0-100): If you answered this same question 100 times independently, "
            "what percentage would you expect to get correct? "
            "(0 = pure random guess, 25 = one of four choices, 50 = coin flip, 100 = absolutely certain)\n"
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
        """
        Runs the subject K times to compute behavioral confidence (C_beh).
        
        C_beh = Number of CORRECT answers / K (this is the Ground Truth for ToM prediction)
        """

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

        # Choose response model based on justification requirement
        response_model = SubjectOutputWithJustification if self.require_justification else SubjectOutput

        # Sequential execution with rate limiting to avoid 429 errors
        # OpenRouter limits Gemma-3-27B to 15 requests/minute
        RATE_LIMIT_DELAY = 4.5  # seconds between requests (safe margin for 15 RPM)
        
        results = []
        for i in range(self.k_samples):
            logger.info(f"    Running Subject sample {i + 1}/{self.k_samples} [{framing}]")
            result = await self.client.agenerate_parsed(messages, response_model)
            results.append(result)
            
            # Add delay between requests to respect rate limit
            if i < self.k_samples - 1:
                await asyncio.sleep(RATE_LIMIT_DELAY)

        samples: List[SolvedInstance] = []
        gt_choice = extract_choice(ground_truth)
        
        for parsed_result in results:
            if parsed_result:
                answer_choice = extract_choice(parsed_result.final_answer)
                
                # Build confidence justification if available
                justification = None
                if self.require_justification and isinstance(parsed_result, SubjectOutputWithJustification):
                    justification = ConfidenceJustification(
                        key_reasoning_steps=parsed_result.key_reasoning_steps,
                        uncertainty_factors=parsed_result.uncertainty_factors,
                        confidence_anchors=parsed_result.confidence_anchors,
                        potential_errors=parsed_result.potential_errors,
                    )
                
                samples.append(
                    SolvedInstance(
                        cot=parsed_result.reasoning,
                        answer=answer_choice,
                        reported_confidence=parsed_result.confidence,
                        confidence_justification=justification,
                    )
                )

        if not samples:
            return None

        # Compute Behavioral Confidence: C_beh = correct_count / total_samples
        # This is the GROUND TRUTH for ToM prediction
        # Use exact match on extracted choice letter (a/b/c/d)
        correct_count = sum(
            1 for s in samples 
            if s.answer and s.answer == gt_choice
        )
        c_beh = correct_count / len(samples)

        # Compute consistency rate (for reference)
        answers = [s.answer for s in samples]
        counter = collections.Counter(answers)
        majority_ans, majority_count = counter.most_common(1)[0]
        consistency_rate = majority_count / len(samples)

        avg_c_rep = sum(s.reported_confidence for s in samples) / len(samples)

        # Is majority answer correct? (exact match on choice letter)
        is_correct = majority_ans == gt_choice

        # Primary CoT: from a sample that yielded the majority answer
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
