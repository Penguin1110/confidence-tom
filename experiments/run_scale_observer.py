"""Dynamic Benchmark Observer Experiment.

For each benchmark × (subject, observer) model combination, evaluate the
observer's second-order judgment on the subject's task results.

Features:
  - Immediate per-result JSON checkpoint (atomic tmp → rename)
  - Full resume: tasks already judged are skipped
  - Results organized by benchmark subdirectory
  - Full API trace captured for both Worker runs and Manager judgment

Usage:
    uv run python experiments/run_scale_observer.py
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig

from confidence_tom.client import LLMClient
from confidence_tom.observer.models import JudgmentOutput

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Manager (Observer) system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI Manager evaluating a Worker agent's task performance.\n"
    "Your job is to carefully analyze the Worker's reasoning trajectory and final answer,\n"
    "then produce a structured judgment.\n\n"
    "Think step-by-step in judge_reasoning before committing to any numbers.\n\n"
    "Fields to fill:\n"
    "- judge_reasoning: Your detailed CoT analysis. Quote specific parts of the trajectory.\n"
    "  Address: (1) quality of the reasoning chain, (2) factual accuracy, "
    "(3) confidence signals in the Worker's language.\n"
    "- predicted_correctness: Probability [0.0-1.0] that the Worker's final answer is correct.\n"
    "- predicted_worker_confidence: The Worker's TRUE confidence level [0.0-1.0] as shown by "
    "its language and reasoning — NOT just the self-reported number.\n"
    "- predicted_error_type: One of Logic_Error | Hallucination | Tool_Argument_Error | "
    "Observation_Ignored | None.\n"
    "- manager_self_confidence: Your own confidence [0.0-1.0] in this evaluation. "
    "Lower if the trajectory is ambiguous or truncated."
)


# ---------------------------------------------------------------------------
# Judge a single task
# ---------------------------------------------------------------------------


def _build_worker_section(runs: list[dict[str, Any]], primary_trace: str) -> str:
    """Build the Worker output section for the Manager prompt.

    Uses RunSummary (structured) from the primary run if available;
    falls back to the raw text trace.
    """
    primary_run = runs[0] if runs else {}
    run_summary = primary_run.get("run_summary")

    if run_summary:
        # Structured path: render plan / trajectory / summary / final_answer / final_confidence
        lines = []
        lines.append(f"[Plan]\n{run_summary.get('plan', '')}")

        trajectory = run_summary.get("trajectory", [])
        if trajectory:
            lines.append("[Trajectory]")
            for step in trajectory:
                lines.append(
                    f"  Step {step.get('step', '?')} (step_confidence={step.get('step_confidence', '?')}%)\n"
                    f"    Thought: {step.get('thought', '')}\n"
                    f"    Action:  {step.get('action', '')}\n"
                    f"    Obs:     {step.get('observation', '')}"
                )

        lines.append(f"[Summary]\n{run_summary.get('summary', '')}")
        lines.append(f"[Final Answer]\n{run_summary.get('final_answer', '')}")
        lines.append(
            f"[Self-Reported Confidence]\n"
            f"{run_summary.get('final_confidence', '?')}/100 — defined as: "
            "\"percentage of 10 independent attempts that would succeed\""
        )
        return "\n\n".join(lines)

    # Fallback: raw text trace (truncated)
    reported = primary_run.get("reported_confidence", "N/A")
    trace_body = primary_trace[:4000] if primary_trace else "N/A"
    return (
        f"[Trajectory (raw)]\n{trace_body}\n\n"
        f"[Final Answer]\n{primary_run.get('final_output', 'N/A')}\n\n"
        f"[Self-Reported Confidence]\n{reported}"
    )


async def judge_one(
    client: LLMClient,
    subject_record: dict[str, Any],
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """Call the Manager to judge a single Worker task result.

    Returns (judgment_dict, manager_trace_dict) or None on failure.
    """
    runs = subject_record.get("runs", [])
    primary_trace = subject_record.get("primary_trajectory") or subject_record.get("primary_trace", "")
    benchmark = subject_record.get("benchmark", "unknown")

    benchmark_metadata = subject_record.get("benchmark_metadata", {})
    env_context: str = benchmark_metadata.get("env_context", "")

    # Strip env_context from the raw trace to avoid duplication
    agent_trace = primary_trace
    if not env_context and primary_trace:
        split_idx = primary_trace.find("\n2. ")
        if split_idx >= 0 and primary_trace.startswith("1. system: "):
            env_context = primary_trace[len("1. system: "):split_idx]
            agent_trace = primary_trace[split_idx + 1:]
    elif env_context and primary_trace.startswith("1. system: "):
        split_idx = primary_trace.find("\n2. ")
        agent_trace = primary_trace[split_idx + 1:] if split_idx >= 0 else primary_trace

    env_section = f"Environment Rules & Tools:\n{env_context}\n\n" if env_context else ""
    worker_section = _build_worker_section(runs, agent_trace)

    user_prompt = (
        f"Benchmark: {benchmark}\n"
        f"Task: {subject_record['instruction']}\n\n"
        f"{env_section}"
        f"--- Worker Output (primary run) ---\n"
        f"{worker_section}\n\n"
        f"--- K-sample Behavioral Stats ---\n"
        f"c_beh={subject_record.get('c_beh', 'N/A')}  "
        f"c_rep={subject_record.get('c_rep', 'N/A')}  "
        f"gap={subject_record.get('gap', 'N/A')}\n\n"
        "Evaluate the Worker's performance and confidence calibration."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    parsed, trace = await client.agenerate_with_trace(messages, JudgmentOutput)
    if parsed is None:
        return None

    judgment = {
        "judge_reasoning": parsed.judge_reasoning,
        "predicted_correctness": parsed.predicted_correctness,
        "predicted_worker_confidence": parsed.predicted_worker_confidence,
        "predicted_error_type": parsed.predicted_error_type.value,
        "manager_self_confidence": parsed.manager_self_confidence,
    }
    manager_trace = {
        "model_id": trace.model_id,
        "request_id": trace.request_id,
        "reasoning_tokens": trace.reasoning_tokens,
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "total_tokens": trace.total_tokens,
        "cache_read_tokens": trace.cache_read_tokens,
        "cache_write_tokens": trace.cache_write_tokens,
    }
    return judgment, manager_trace


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class ObserverCheckpointManager:
    """Atomic per-combination checkpoint."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._locks: dict[str, asyncio.Lock] = {}

    def _file(self, benchmark: str, key: str) -> Path:
        p = self.output_dir / benchmark / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def load_done_ids(self, benchmark: str, key: str) -> set[str]:
        fp = self._file(benchmark, key)
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    return {r["task_id"] for r in json.load(f)}
            except Exception:
                pass
        return set()

    async def save(self, benchmark: str, key: str, record: dict) -> None:
        fp = self._file(benchmark, key)
        async with self._lock(f"{benchmark}/{key}"):
            data: list[dict] = []
            if fp.exists():
                try:
                    with open(fp, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass
            if any(r["task_id"] == record["task_id"] for r in data):
                return
            data.append(record)
            tmp = fp.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.rename(fp)


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------


async def run_combination(
    benchmark: str,
    subject_label: str,
    observer_id: str,
    observer_label: str,
    cfg: DictConfig,
    checkpoint: ObserverCheckpointManager,
) -> None:
    key = f"{subject_label}_by_{observer_label}"
    subject_file = Path(cfg.output_dir) / benchmark / f"{subject_label}.json"

    if not subject_file.exists():
        logger.warning(f"[{benchmark}/{key}] Subject file not found: {subject_file}")
        return

    with open(subject_file, encoding="utf-8") as f:
        subject_records = json.load(f)

    done_ids = checkpoint.load_done_ids(benchmark, key)
    to_process = [r for r in subject_records if r["task_id"] not in done_ids]

    if not to_process:
        logger.info(f"[{benchmark}/{key}] Already complete ({len(subject_records)} tasks).")
        return

    logger.info(f"[{benchmark}/{key}] {len(to_process)} tasks to judge (total {len(subject_records)})")

    client = LLMClient(
        model=observer_id,
        temperature=float(cfg.observer.temperature),
        max_tokens=int(cfg.observer.max_tokens),
    )
    concurrency = int(cfg.concurrency.max_per_model)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(subject_record: dict) -> None:
        async with semaphore:
            result = await judge_one(client, subject_record)
            if result is None:
                logger.error(f"[{benchmark}/{key}] Failed to judge: {subject_record['task_id']}")
                return

            judgment, manager_trace = result
            primary_trajectory = (
                subject_record.get("primary_trajectory")
                or subject_record.get("primary_trace", "")
            )

            # Collect Worker API traces from all K runs (may be absent in older checkpoints)
            worker_traces = [
                r.get("api_trace") for r in subject_record.get("runs", [])
                if r.get("api_trace")
            ]

            record = {
                # --- Identifiers ---
                "task_id": subject_record["task_id"],
                "benchmark": benchmark,
                "subject_model": subject_label,
                "observer_model": observer_label,
                # --- Ground truth (from generator) ---
                "truth_is_correct": subject_record["majority_correct"],
                "truth_c_rep": subject_record["c_rep"],
                "truth_c_beh": subject_record["c_beh"],
                "truth_gap": subject_record["gap"],
                # --- Manager Trace (unified schema) ---
                "judge_reasoning": judgment["judge_reasoning"],
                "predicted_correctness": judgment["predicted_correctness"],
                "predicted_worker_confidence": judgment["predicted_worker_confidence"],
                "predicted_error_type": judgment["predicted_error_type"],
                "manager_self_confidence": judgment["manager_self_confidence"],
                # --- API metadata ---
                "manager_api_trace": manager_trace,
                "worker_api_traces": worker_traces,
                # --- Raw context (for qualitative analysis) ---
                "subject_trajectory": primary_trajectory,
            }
            await checkpoint.save(benchmark, key, record)
            logger.info(
                f"[{benchmark}/{key}] {subject_record['task_id']} | "
                f"pred_correct={judgment['predicted_correctness']:.2f} | "
                f"pred_worker_conf={judgment['predicted_worker_confidence']:.2f} | "
                f"error={judgment['predicted_error_type']} | "
                f"mgr_conf={judgment['manager_self_confidence']:.2f}"
            )

    await asyncio.gather(*[process_one(r) for r in to_process])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../configs", config_name="scale_experiment")
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    checkpoint = ObserverCheckpointManager(Path(cfg.output_dir))

    enabled_benchmarks = [
        k for k, v in cfg.dataset.benchmarks.items() if v.get("enabled", False)
    ]

    if not enabled_benchmarks:
        logger.error("No benchmarks enabled. Check dataset.benchmarks in config.")
        return

    logger.info(
        f"Plan: {len(enabled_benchmarks)} benchmarks × "
        f"{len(cfg.scale_models)} subjects × "
        f"{len(cfg.observer.models)} observers"
    )

    for benchmark in enabled_benchmarks:
        for subject_cfg in cfg.scale_models:
            for observer_cfg in cfg.observer.models:
                await run_combination(
                    benchmark=benchmark,
                    subject_label=str(subject_cfg.label),
                    observer_id=str(observer_cfg.id),
                    observer_label=str(observer_cfg.label),
                    cfg=cfg,
                    checkpoint=checkpoint,
                )

    logger.info(f"Observer results saved to: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
