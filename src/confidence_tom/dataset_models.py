"""Unified data models for the scale experiment dataset.

All questions are standardized to multiple-choice format to reduce
parsing noise and enable fair confidence comparison across benchmarks.
"""

from typing import Optional

from pydantic import BaseModel, Field


class MCQuestion(BaseModel):
    """A standardized multiple-choice question for the scale experiment."""

    id: str = Field(description="Unique identifier, e.g. 'mmlu_astronomy_042'")
    question: str = Field(description="The question text")
    choices: list[str] = Field(
        description="List of choices, e.g. ['A) Mercury', 'B) Venus', 'C) Earth', 'D) Mars']"
    )
    correct_answer: str = Field(description="Correct answer letter: 'A', 'B', 'C', or 'D'")
    category: str = Field(
        description="Task category: 'math', 'science', 'knowledge', 'truthfulness'"
    )
    source: str = Field(
        description="Source benchmark: 'mmlu', 'arc_challenge', 'truthfulqa', 'gsm8k'"
    )
    external_difficulty: Optional[str] = Field(
        default=None,
        description="Benchmark-native difficulty proxy (e.g., MMLU subject, ARC split, "
        "TruthfulQA category, GSM8K reasoning steps)",
    )
