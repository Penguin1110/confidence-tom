"""Unified data models for dynamic benchmark tasks.

Replaces the old MCQuestion / SubjectOutputV2 models.
All benchmarks map their tasks into DynamicTask, and all agent runs
produce AgentRun records that aggregate into a TaskResult.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ApiTrace(BaseModel):
    """Raw API response metadata captured from each LLM call."""

    model_id: str = Field(default="", description="Actual model name returned by API")
    request_id: str = Field(default="", description="OpenRouter/API request ID for cost lookup")
    reasoning_tokens: int = Field(default=0, description="Tokens used for internal reasoning (thinking)")
    reasoning_content: str = Field(default="", description="Reasoning text if exposed by model (e.g. DeepSeek-R1)")
    response_content: str = Field(
        default="",
        description="Raw assistant message content returned by the API",
    )
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    cache_read_tokens: int = Field(default=0, description="Cache-hit tokens (cost savings)")
    cache_write_tokens: int = Field(default=0, description="Tokens written to cache this request")


class StaticTrace(BaseModel):
    """Structured trace for single-shot static reasoning tasks."""

    strategy: str = Field(
        default="",
        description="Initial high-level plan for solving the task",
    )
    reasoning: str = Field(
        default="",
        description="Core reasoning process that led to the final answer",
    )
    answer: str = Field(default="", description="Final selected answer")
    confidence: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Self-reported confidence in the final answer",
    )


class TrajectoryStep(BaseModel):
    """A single step in the agent's dynamic execution trace."""

    step: int = Field(description="Step index, starting from 1")
    thought: str = Field(description="The agent's reasoning before taking the action")
    action: str = Field(description="The action taken (tool call, query, or text action)")
    observation: str = Field(default="", description="Environment/API feedback after the action")
    step_confidence: int = Field(
        ge=0, le=100,
        description="Agent's confidence at this step (0-100)"
    )


class RunSummary(BaseModel):
    """Structured post-hoc reflection produced by the agent after completing a run."""

    plan: str = Field(description="Initial strategy before executing steps.")
    trajectory: list["TrajectoryStep"] = Field(
        description=(
            "Step-by-step reflection. For each action taken, record thought, action, "
            "observation, and step_confidence (0-100: how confident you were that "
            "this specific step was correct at the time)."
        )
    )
    summary: str = Field(description="Final synthesis of all observations.")
    final_answer: str = Field(description="Final declared answer or task result.")
    final_confidence: int = Field(
        ge=0, le=100,
        description=(
            "If you attempted this exact task 10 times independently from scratch, "
            "what percentage of those attempts would succeed? (0-100)"
        ),
    )


class DynamicTask(BaseModel):
    """A single task from any dynamic benchmark."""

    task_id: str = Field(description="Unique identifier, e.g. 'tau_retail_042'")
    benchmark: str = Field(description="Source benchmark: 'tau-bench', 'bird-sql', 'plancraft', 'intercode'")
    task_type: str = Field(default="agent", description="High-level task type, e.g. QA or agent")
    instruction: str = Field(description="Natural-language task description shown to the agent")
    ground_truth: Any = Field(description="Benchmark-specific correct answer or success criteria")
    metadata: dict = Field(default_factory=dict, description="Benchmark-specific extra data")
    environment_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool or environment descriptions exposed to the worker",
    )


class AgentRun(BaseModel):
    """A single run of the agent on a task (one of K samples)."""

    # Dynamic Trajectory Schema
    plan: str = Field(default="", description="Agent's initial strategy before executing steps")
    trajectory: list[TrajectoryStep] = Field(
        default_factory=list,
        description="Step-by-step execution trace"
    )
    summary: str = Field(default="", description="Final synthesis of all observations")
    final_answer: str = Field(description="Agent's final answer or last submitted action")
    is_correct: bool = Field(description="Whether the benchmark evaluator deemed this run successful")
    reported_confidence: float = Field(
        description="Agent's self-reported final confidence (0-1 scale)"
    )
    api_trace: Optional[ApiTrace] = Field(default=None, description="Raw API metadata for this call")

    def trajectory_as_text(self) -> str:
        """Flatten structured trajectory to a plain string for the observer prompt."""
        parts = []
        if self.plan:
            parts.append(f"Plan: {self.plan}")
        for s in self.trajectory:
            parts.append(
                f"Step {s.step} [conf={s.step_confidence}%]\n"
                f"  Thought: {s.thought}\n"
                f"  Action: {s.action}"
                + (f"\n  Observation: {s.observation}" if s.observation else "")
            )
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        return "\n\n".join(parts) if parts else self.final_answer


class TaskResult(BaseModel):
    """Aggregate result across K runs of one task — parallel to old SubjectOutputV2."""

    task_id: str
    benchmark: str
    instruction: str

    runs: list[AgentRun] = Field(description="K independent runs")

    majority_correct: bool = Field(description="True if majority of runs were correct")
    behavioral_confidence: float = Field(
        description="C_beh: fraction of runs that succeeded (0-1)"
    )
    avg_reported_confidence: float = Field(
        description="C_rep: mean of per-run self-reported confidences (0-1)"
    )

    primary_trajectory: str = Field(
        description="Flattened trajectory string from the primary run (for observer prompt)"
    )


class NativeRun(BaseModel):
    """One native benchmark execution inside a task-level K-run bundle."""

    trial: int = Field(description="0-based trial index inside the K samples")
    is_correct: bool = Field(description="Whether this native run succeeded")
    reward: Optional[float] = Field(
        default=None,
        description="Native environment reward when available",
    )
    reported_confidence: Optional[float] = Field(
        default=None,
        description="Self-reported confidence when the native benchmark exposes it",
    )
    final_output: str = Field(
        default="",
        description="Final response, SQL, action, or summary emitted by the agent",
    )
    trace_text: str = Field(
        default="",
        description="Human-readable trace summary for audits and observers",
    )
    trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw native trace payload",
    )
    api_trace: Optional[ApiTrace] = Field(
        default=None,
        description="Optional API metadata when the runner captured it",
    )
    benchmark_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific raw metadata for this run",
    )
    run_summary: Optional["RunSummary"] = Field(
        default=None,
        description="Structured post-hoc reflection elicited after the run completes",
    )


class NativeTaskResult(BaseModel):
    """Unified JSON schema for native benchmark task results."""

    schema_version: str = Field(default="native_dynamic_v1")
    native_mode: bool = Field(default=True)
    task_id: str
    benchmark: str
    instruction: str
    benchmark_metadata: dict[str, Any] = Field(default_factory=dict)
    majority_correct: bool
    c_beh: float = Field(description="Behavioral confidence over native runs")
    c_rep: Optional[float] = Field(
        default=None,
        description="Average reported confidence if the benchmark exposes it",
    )
    gap: Optional[float] = Field(
        default=None,
        description="c_rep - c_beh when c_rep is available",
    )
    k_samples: int
    primary_trace: str = Field(
        default="",
        description="Representative trace text from a successful run or the first run",
    )
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact native summary stats for downstream analysis",
    )
    runs: list[NativeRun] = Field(default_factory=list)
