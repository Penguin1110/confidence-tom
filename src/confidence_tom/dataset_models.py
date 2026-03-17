"""Unified data models for single-run static benchmark tasks."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class StaticTask(BaseModel):
    """A static single-turn task, either MC or open-ended."""

    id: str = Field(description="Unique identifier, e.g. 'mmlu_astronomy_042'")
    question: str = Field(description="The question text")
    choices: list[str] = Field(
        default_factory=list,
        description="Optional choices for MC tasks",
    )
    correct_answer: str = Field(default="", description="Correct answer letter for MC tasks")
    reference_answer: str = Field(default="", description="Canonical answer string for evaluators")
    category: str = Field(
        description="Task category: 'math', 'science', 'knowledge', 'truthfulness'"
    )
    source: str = Field(
        description="Source benchmark: 'mmlu', 'arc_challenge', 'truthfulqa', 'gsm8k'"
    )
    answer_format: str = Field(
        default="multiple_choice",
        description="Expected answer format: multiple_choice or open_ended",
    )
    evaluator_name: str = Field(
        default="mc_letter",
        description="Static evaluator identifier",
    )
    task_type: str = Field(default="QA", description="High-level task type")
    environment_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool/API environment descriptions for the task",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific fields used by evaluators",
    )
    external_difficulty: Optional[str] = Field(
        default=None,
        description="Benchmark-native difficulty proxy (e.g., MMLU subject, ARC split, "
        "TruthfulQA category, GSM8K reasoning steps)",
    )


MCQuestion = StaticTask
