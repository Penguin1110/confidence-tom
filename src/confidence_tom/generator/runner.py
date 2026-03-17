"""Agent runner — replaces the old SubjectGenerator.

For each DynamicTask the runner:
  1. Builds a prompt from the task instruction.
  2. Calls the LLM to get a final answer + self-reported confidence.
  3. Evaluates correctness via the benchmark-specific evaluator.
  4. Repeats K times (temperature > 0) to compute behavioral confidence (C_beh).

Benchmark-specific correctness evaluation is injected via the `evaluator`
callable so this file stays benchmark-agnostic.
"""

import asyncio
import logging
from typing import Callable, Optional

from pydantic import BaseModel, Field

from confidence_tom.client import LLMClient
from confidence_tom.evaluators import BenchmarkEvaluator
from confidence_tom.task_models import AgentRun, DynamicTask, TaskResult, TrajectoryStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output the LLM must return
# ---------------------------------------------------------------------------


class AgentStepOutput(BaseModel):
    """One step in the agent's execution trace."""

    step: int = Field(description="Step index, starting from 1.")
    thought: str = Field(description="Reasoning before taking the action.")
    action: str = Field(description="The action taken or tool call made.")
    observation: str = Field(default="", description="Result/feedback after the action.")
    step_confidence: int = Field(
        ge=0, le=100,
        description="Confidence at this step that the approach is correct (0-100).",
    )


class AgentOutput(BaseModel):
    """Structured output expected from the agent for any dynamic task."""

    plan: str = Field(description="Initial strategy before executing steps.")
    trajectory: list[AgentStepOutput] = Field(
        description="Step-by-step execution trace. Each step has thought, action, observation, and step_confidence."
    )
    summary: str = Field(description="Final synthesis of all observations.")
    final_answer: str = Field(
        description="The final answer or action submitted to complete the task."
    )
    final_confidence: int = Field(
        ge=0, le=100,
        description=(
            "If you attempted this exact task 10 times independently from scratch, "
            "what percentage of those attempts would succeed? (0-100)"
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI agent solving tasks. For each task you must produce a structured response:\n"
    "1. plan: Your initial strategy for solving the task.\n"
    "2. trajectory: A list of steps. Each step has:\n"
    "   - step: step number (1, 2, 3, ...)\n"
    "   - thought: your reasoning before acting\n"
    "   - action: the action you take or the tool you call\n"
    "   - observation: the result or feedback you receive\n"
    "   - step_confidence: your confidence (0-100) that this step is correct\n"
    "3. summary: Final synthesis of all observations.\n"
    "4. final_answer: Your final answer (be concise and precise).\n"
    "5. final_confidence: Your overall confidence (0-100) that the final answer is correct."
)


class AgentRunner:
    """Runs an LLM agent on DynamicTask instances with K-sample behavioral confidence."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        k_samples: int = 5,
        max_tokens: int = 4096,
        k_concurrency: int = 3,
    ) -> None:
        self.k_samples = k_samples
        self._k_sem = asyncio.Semaphore(k_concurrency)
        self.client = LLMClient(
            model=model_name, temperature=temperature, max_tokens=max_tokens
        )

    async def run(
        self,
        task: DynamicTask,
        evaluator: BenchmarkEvaluator,
    ) -> Optional[TaskResult]:
        """Run the agent K times on a task and aggregate results.

        Args:
            task: The task to solve.
            evaluator: Callable(agent_answer, task, evidence_text) -> bool that checks correctness.
                       Each benchmark provides its own implementation.

        Returns:
            TaskResult with C_rep and C_beh, or None if all K samples failed.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ]

        async def fetch_sample(i: int) -> Optional[AgentRun]:
            async with self._k_sem:
                logger.info(f"  [{task.task_id}] sample {i + 1}/{self.k_samples}")
                parsed, trace = await self.client.agenerate_with_trace(messages, AgentOutput)
            if parsed is None:
                return None
            evidence_text = _render_evidence_text(parsed)
            is_correct = evaluator(parsed.final_answer, task, evidence_text)
            steps = [
                TrajectoryStep(
                    step=s.step,
                    thought=s.thought,
                    action=s.action,
                    observation=s.observation,
                    step_confidence=s.step_confidence,
                )
                for s in parsed.trajectory
            ]
            return AgentRun(
                plan=parsed.plan,
                trajectory=steps,
                summary=parsed.summary,
                final_answer=parsed.final_answer,
                is_correct=is_correct,
                reported_confidence=max(0.0, min(1.0, parsed.final_confidence / 100.0)),
                api_trace=trace,
            )

        results = await asyncio.gather(*[fetch_sample(i) for i in range(self.k_samples)])
        runs = [r for r in results if r is not None]

        if not runs:
            logger.warning(f"[{task.task_id}] All {self.k_samples} samples failed to parse.")
            return None

        c_beh = sum(r.is_correct for r in runs) / len(runs)
        c_rep = sum(r.reported_confidence for r in runs) / len(runs)
        majority_correct = c_beh >= 0.5

        # Pick primary trajectory from a correct run if possible, else first run
        primary = next((r for r in runs if r.is_correct), runs[0])

        return TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            instruction=task.instruction,
            runs=runs,
            majority_correct=majority_correct,
            behavioral_confidence=c_beh,
            avg_reported_confidence=c_rep,
            primary_trajectory=primary.trajectory_as_text(),
        )


def _render_evidence_text(parsed: AgentOutput) -> str:
    """Flatten structured agent output for evaluator heuristics."""
    parts = []
    if parsed.plan:
        parts.append(parsed.plan)
    for step in parsed.trajectory:
        parts.extend([step.thought, step.action, step.observation])
    if parsed.summary:
        parts.append(parsed.summary)
    return "\n".join(part for part in parts if part)
