"""Native dynamic benchmark dispatcher.

This entrypoint keeps the existing scale-experiment UX and checkpointing, but
dispatches each benchmark to its own native execution loop:
  - tau_bench: official environment + agent/tool loop
  - bird_sql: SQL generation + execution-match evaluation
  - plancraft: native PlancraftGymWrapper interaction loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 on stdout/stderr so emoji from model outputs don't crash on
# Windows consoles using narrow encodings like cp950.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Any, Awaitable, Callable, Optional

import hydra
from omegaconf import DictConfig

from confidence_tom.client import LLMClient
from confidence_tom.evaluators import extract_sql
from confidence_tom.task_models import DynamicTask, NativeRun, NativeTaskResult, RunSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "external" / "tau-bench"

PLANCRAFT_SYSTEM_PROMPT = """You are solving a Plancraft crafting task.
Each turn you receive an observation showing inventory slots (e.g. `oak_log [I1] quantity 3`)
and crafting grid slots (A1-A3, B1-B3, C1-C3) and output slot [0].

Valid actions (output exactly one per turn):
- move: from [SlotID] to [SlotID] with quantity N
- smelt: from [SlotID] to [SlotID] with quantity N
- impossible: <reason>

Slot IDs:
- Inventory: [I1] to [I36]
- Crafting grid: [A1][A2][A3] / [B1][B2][B3] / [C1][C2][C3]
- Crafting output: [0]

Rules:
- Use `move` to place items into the crafting grid, then `move` [0] to inventory to collect the output.
- Reference slots by their ID shown in the observation (e.g. `oak_log [I3]` → use `[I3]`).
- Do not invent items or slots not shown in the observation."""

TAU_CONFIDENCE_ADDON = """
## Reflection requirement
After this conversation ends you will be asked to produce a structured self-reflection. Prepare by mentally tracking:
- plan: your initial strategy at the start of the task.
- trajectory: for each action you took, record thought (reasoning before the action), action (the exact tool call or message), observation (what the tool returned or the user said), and step_confidence (0-100: how confident you were that this specific step was correct at the time).
- summary: a final synthesis of all observations once the task is complete.
- final_answer: the final result or outcome you achieved.
- final_confidence: if you attempted this exact task 10 times independently from scratch, what percentage would succeed? (0-100)"""


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _serialize_native(result: NativeTaskResult) -> dict[str, Any]:
    data = result.model_dump()
    data["c_beh"] = _round_or_none(data["c_beh"])
    data["c_rep"] = _round_or_none(data["c_rep"])
    data["gap"] = _round_or_none(data["gap"])
    # Backward-compatible aliases for downstream observer/analysis code.
    data["primary_trajectory"] = data.get("primary_trace", "")
    for run in data["runs"]:
        run["reward"] = _round_or_none(run.get("reward"))
        run["reported_confidence"] = _round_or_none(run.get("reported_confidence"))
        run["final_answer"] = run.get("final_output", "")
    return data


def _aggregate_native_task(
    task_id: str,
    benchmark: str,
    instruction: str,
    benchmark_metadata: dict[str, Any],
    runs: list[NativeRun],
) -> NativeTaskResult:
    if not runs:
        raise ValueError(f"No native runs collected for {task_id}")

    successes = sum(1 for run in runs if run.is_correct)
    c_beh = successes / len(runs)
    reported = [run.reported_confidence for run in runs if run.reported_confidence is not None]
    c_rep = (sum(reported) / len(reported)) if reported else None
    gap = (c_rep - c_beh) if c_rep is not None else None
    primary = next((run for run in runs if run.is_correct), runs[0])

    return NativeTaskResult(
        task_id=task_id,
        benchmark=benchmark,
        instruction=instruction,
        benchmark_metadata=_json_safe(benchmark_metadata),
        majority_correct=c_beh >= 0.5,
        c_beh=c_beh,
        c_rep=c_rep,
        gap=gap,
        k_samples=len(runs),
        primary_trace=primary.trace_text,
        summary={
            "num_successes": successes,
            "num_failures": len(runs) - successes,
            "has_reported_confidence": bool(reported),
        },
        runs=runs,
    )


def _ensure_tau_imports() -> None:
    if str(TAU_ROOT) not in sys.path:
        sys.path.insert(0, str(TAU_ROOT))


def _load_tau_tasks(env_name: str, split: str) -> list[Any]:
    _ensure_tau_imports()
    if env_name == "retail":
        if split == "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks  # type: ignore
        elif split == "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks  # type: ignore
        elif split == "dev":
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks  # type: ignore
        else:
            raise ValueError(f"Unsupported tau_bench retail split: {split}")
        return list(tasks)

    if env_name == "airline":
        if split == "test":
            from tau_bench.envs.airline.tasks_test import TASKS as tasks  # type: ignore
        else:
            raise ValueError(f"Unsupported tau_bench airline split: {split}")
        return list(tasks)

    raise ValueError(f"Unsupported tau_bench env: {env_name}")


def _format_tau_trace(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        role = str(msg.get("role", "unknown"))
        if role == "assistant" and msg.get("tool_calls"):
            tool_call = msg["tool_calls"][0]
            name = tool_call.get("function", {}).get("name", "")
            args = tool_call.get("function", {}).get("arguments", "")
            lines.append(f"{idx}. assistant tool_call {name}({args})")
            continue
        if role == "tool":
            lines.append(f"{idx}. tool {msg.get('name', '')}: {msg.get('content', '')}")
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{idx}. {role}: {content or ''}")
    return "\n".join(lines)


def _extract_tau_final_output(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if msg.get("tool_calls"):
                tool_call = msg["tool_calls"][0]
                name = tool_call.get("function", {}).get("name", "")
                args = tool_call.get("function", {}).get("arguments", "")
                return f"{name}({args})"
    return ""



def _shorten_log_text(text: Any, limit: int = 180) -> str:
    content = str(text or "").replace("\n", " ").strip()
    if len(content) <= limit:
        return content
    return content[: limit - 3] + "..."


class _TauBenchUserSimulator:
    """LLM-backed tau-bench user simulator using the project's OpenRouter client."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.messages: list[dict[str, str]] = []

    def _build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = f"\n\nInstruction: {instruction}\n" if instruction else ""
        return (
            f"You are a user interacting with an agent.{instruction_display}"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction.\n"
            "- If the instruction goal is satisfied, generate '###STOP###' as a standalone message without anything else.\n"
            "- Do not repeat the exact instruction in the conversation. Use your own words.\n"
            "- Keep the conversation natural and consistent with the personality in the instruction."
        )

    def _generate_next_message(self) -> str:
        text = self.client.generate_text(self.messages)
        return text.strip() or "###STOP###"

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {"role": "system", "content": self._build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        reply = self._generate_next_message()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        reply = self._generate_next_message()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def get_total_cost(self) -> float:
        return 0.0


def _message_to_tau_action(message: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tool_call = tool_calls[0]
        function = tool_call.get("function", {})
        name = str(function.get("name", "")).strip()
        raw_args = function.get("arguments", "") or "{}"
        try:
            kwargs = json.loads(raw_args)
        except Exception:
            kwargs = {}
        return name, kwargs, f"{name}({raw_args})"

    content = str(message.get("content") or "").strip()
    return "respond", {"content": content}, content


async def _run_tau_native_task(
    task_spec: dict[str, Any],
    get_env: Callable[..., Any],
    action_factory: Callable[..., Any],
    tau_cfg: Any,
    agent_client: LLMClient,
    user_client: LLMClient,
    k_samples: int,
    log_prefix: str,
    use_tool_calling: bool = True,
    on_run: Optional[Callable[["NativeRun"], None]] = None,
    preloaded_runs: Optional[list["NativeRun"]] = None,
) -> NativeTaskResult:
    task_index = int(task_spec["task_index"])
    tau_tasks = _load_tau_tasks(str(tau_cfg.env), str(tau_cfg.split))
    task = tau_tasks[task_index]
    runs: list[NativeRun] = []
    tau_wiki: str = ""  # captured from first env instance

    for trial in range(k_samples):
        if preloaded_runs and trial < len(preloaded_runs):
            runs.append(preloaded_runs[trial])
            logger.info(f"{log_prefix} sample {trial + 1}/{k_samples} SKIP (preloaded)")
            continue
        try:
            logger.info(f"{log_prefix} sample {trial + 1}/{k_samples} start")
            isolated_env = get_env(
                tau_cfg.env,
                user_strategy="human",
                user_model="unused",
                user_provider=None,
                task_split=str(tau_cfg.split),
                task_index=task_index,
            )
            isolated_env.user = _TauBenchUserSimulator(user_client)
            reset_res = await asyncio.to_thread(isolated_env.reset, task_index)
            if not tau_wiki:
                tau_wiki = getattr(isolated_env, "wiki", "")
            reward = 0.0
            done = False
            tau_system = isolated_env.wiki + TAU_CONFIDENCE_ADDON
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": tau_system},
                {"role": "user", "content": reset_res.observation},
            ]
            trace: list[dict[str, Any]] = list(messages)
            info = reset_res.info.model_dump()
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_reasoning_tokens = 0

            max_steps = int(tau_cfg.get("max_steps", 30))
            logger.info(
                f"{log_prefix} sample {trial + 1}/{k_samples} user_init="
                f"{_shorten_log_text(reset_res.observation)}"
            )

            for step_idx in range(max_steps):
                if use_tool_calling:
                    assistant_message, trace_meta = await agent_client.agenerate_tool_message(
                        messages=messages,
                        tools=isolated_env.tools_info,
                    )
                else:
                    assistant_message, trace_meta = await agent_client.agenerate_react_message(
                        messages=messages,
                        tools=isolated_env.tools_info,
                    )
                if assistant_message is None:
                    logger.warning(f"{log_prefix} sample {trial + 1}/{k_samples} step {step_idx + 1}/{max_steps} -> no model output")
                    break

                total_prompt_tokens += trace_meta.prompt_tokens
                total_completion_tokens += trace_meta.completion_tokens
                total_reasoning_tokens += trace_meta.reasoning_tokens

                action_name, action_kwargs, action_text = _message_to_tau_action(assistant_message)
                logger.info(
                    f"{log_prefix} sample {trial + 1}/{k_samples} step {step_idx + 1}/{max_steps} -> "
                    f"{_shorten_log_text(action_text)}"
                )
                env_response = await asyncio.to_thread(
                    isolated_env.step,
                    action_factory(name=action_name, kwargs=action_kwargs),
                )
                reward = float(env_response.reward)
                info = {**info, **env_response.info.model_dump()}
                done = bool(env_response.done)
                logger.info(
                    f"{log_prefix} sample {trial + 1}/{k_samples} step {step_idx + 1}/{max_steps} obs <- "
                    f"{_shorten_log_text(env_response.observation)}"
                )

                trace.append(assistant_message)
                messages.append(assistant_message)
                if action_name != "respond":
                    tool_call = (assistant_message.get("tool_calls") or [None])[0]
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", "") if tool_call else "",
                        "name": action_name,
                        "content": env_response.observation,
                    }
                    messages.append(tool_message)
                    trace.append(tool_message)
                else:
                    user_message = {"role": "user", "content": env_response.observation}
                    messages.append(user_message)
                    trace.append(user_message)

                if done:
                    logger.info(
                        f"{log_prefix} sample {trial + 1}/{k_samples} finished at step {step_idx + 1}/{max_steps} "
                        f"reward={reward:.2f}"
                    )
                    break

            final_output = _extract_tau_final_output(trace)
            trace_text = _format_tau_trace(trace)
            run_summary = await agent_client.aelicit_run_summary(messages, trace_text=trace_text)
            reported_confidence = (
                round(run_summary.final_confidence / 100.0, 4) if run_summary else None
            )
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=reward >= (1.0 - 1e-6),
                    reward=reward,
                    reported_confidence=reported_confidence,
                    final_output=run_summary.final_answer if run_summary else final_output,
                    trace_text=trace_text,
                    trace=trace,
                    benchmark_payload={
                        "info": _json_safe(info),
                        "token_usage": {
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "reasoning_tokens": total_reasoning_tokens,
                        },
                    },
                    run_summary=run_summary,
                )
            )
            if on_run:
                on_run(runs[-1])
            if not done:
                logger.info(f"{log_prefix} sample {trial + 1}/{k_samples} reached step limit reward={reward:.2f}")
        except Exception as exc:
            logger.warning(f"{log_prefix} sample {trial + 1}/{k_samples} error: {exc}")
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=False,
                    reward=0.0,
                    final_output="",
                    trace_text=f"ERROR: {exc}",
                    trace=[],
                    benchmark_payload={
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            if on_run:
                on_run(runs[-1])

    return _aggregate_native_task(
        task_id=str(task_spec["task_id"]),
        benchmark="tau_bench",
        instruction=task.instruction,
        benchmark_metadata={
            "env": str(tau_cfg.env),
            "split": str(tau_cfg.split),
            "task_index": task_index,
            "user_id": task.user_id,
            "expected_actions": [action.model_dump() for action in task.actions],
            "expected_outputs": list(task.outputs),
            "env_context": tau_wiki,
        },
        runs=runs,
    )


def _load_plancraft_examples(split: str, limit: int) -> list[dict[str, Any]]:
    from plancraft.simple import get_plancraft_examples

    feasible = [ex for ex in get_plancraft_examples(split=split) if not ex.impossible][:limit]
    specs: list[dict[str, Any]] = []
    for i, example in enumerate(feasible):
        inventory_desc = ", ".join(f"{item} x{qty}" for item, qty in example.inventory.items()) or "empty"
        instruction = (
            f"You are playing Minecraft. Craft the following item: {example.target}.\n"
            f"Your current inventory: {inventory_desc}."
        )
        specs.append(
            {
                "task_id": f"plancraft_{split}_{example.id}",
                "instruction": instruction,
                "example": example,
                "metadata": {
                    "split": split,
                    "target_item": example.target,
                    "initial_inventory": dict(example.inventory),
                    "example_index": i,
                    "example_id": example.id,
                },
            }
        )
    return specs


async def _run_plancraft_native_task(
    task_spec: dict[str, Any],
    client: LLMClient,
    k_samples: int,
    max_steps: int = 30,
    on_run: Optional[Callable[["NativeRun"], None]] = None,
    preloaded_runs: Optional[list["NativeRun"]] = None,
) -> NativeTaskResult:
    from plancraft.simple import PlancraftGymWrapper

    example = task_spec["example"]
    runs: list[NativeRun] = []

    for trial in range(k_samples):
        if preloaded_runs and trial < len(preloaded_runs):
            runs.append(preloaded_runs[trial])
            continue
        try:
            env = PlancraftGymWrapper(example=example, max_steps=max_steps)
            obs, reward, terminated, truncated, info = env.step()
            messages = [
                {"role": "system", "content": PLANCRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": obs["text"]},
            ]
            trajectory: list[dict[str, Any]] = [
                {
                    "observation": obs.get("text", ""),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": _json_safe(info),
                }
            ]
            last_action = ""

            while not terminated and not truncated:
                action = await asyncio.to_thread(client.generate_text, messages)
                action = action.strip().splitlines()[0] if action.strip() else "impossible: no action produced"
                last_action = action
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append(
                    {
                        "action": action,
                        "observation": obs.get("text", ""),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": _json_safe(info),
                    }
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": action},
                        {"role": "user", "content": obs.get("text", "")},
                    ]
                )

            trace_text = "\n".join(
                (
                    step.get("action", "[initial]")
                    + ("\n  " + step.get("observation", "") if step.get("observation") else "")
                )
                for step in trajectory
            )
            run_summary = await client.aelicit_run_summary(messages, trace_text=trace_text)
            reported_confidence = (
                round(run_summary.final_confidence / 100.0, 4) if run_summary else None
            )
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=bool(float(reward) >= 1.0),
                    reward=float(reward),
                    reported_confidence=reported_confidence,
                    final_output=run_summary.final_answer if run_summary else (last_action or f"crafted={example.target}"),
                    trace_text=trace_text,
                    trace=trajectory,
                    benchmark_payload={"final_info": _json_safe(info)},
                    run_summary=run_summary,
                )
            )
            if on_run:
                on_run(runs[-1])
        except Exception as exc:
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=False,
                    reward=0.0,
                    final_output="",
                    trace_text=f"ERROR: {exc}",
                    trace=[],
                    benchmark_payload={
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            if on_run:
                on_run(runs[-1])

    return _aggregate_native_task(
        task_id=str(task_spec["task_id"]),
        benchmark="plancraft",
        instruction=str(task_spec["instruction"]),
        benchmark_metadata={**task_spec["metadata"], "env_context": PLANCRAFT_SYSTEM_PROMPT},
        runs=runs,
    )


async def _run_bird_native_task(
    task: DynamicTask,
    client: LLMClient,
    k_samples: int,
    on_run: Optional[Callable[["NativeRun"], None]] = None,
    preloaded_runs: Optional[list["NativeRun"]] = None,
) -> NativeTaskResult:
    from confidence_tom.benchmarks.bird_sql import evaluate_sql

    messages = [
        {"role": "system", "content": "You are a text-to-SQL model. Output only a valid SQLite SQL query."},
        {"role": "user", "content": task.instruction},
    ]
    runs: list[NativeRun] = []

    for trial in range(k_samples):
        if preloaded_runs and trial < len(preloaded_runs):
            runs.append(preloaded_runs[trial])
            continue
        try:
            sql_text = await asyncio.to_thread(client.generate_text, messages)
            sql = extract_sql(sql_text) or sql_text.strip()
            correct = evaluate_sql(
                predicted_sql=sql,
                ground_truth_sql=str(task.ground_truth),
                db_path=str(task.metadata["db_path"]),
            )
            conf_messages = list(messages) + [{"role": "assistant", "content": sql_text}]
            run_summary: Optional[RunSummary] = await client.aelicit_run_summary(
                conf_messages, trace_text=f"Predicted SQL:\n{sql}"
            )
            reported_confidence = (
                round(run_summary.final_confidence / 100.0, 4) if run_summary else None
            )
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=correct,
                    reward=1.0 if correct else 0.0,
                    reported_confidence=reported_confidence,
                    final_output=run_summary.final_answer if run_summary else sql,
                    trace_text=f"Predicted SQL:\n{sql}",
                    trace=[{"predicted_sql": sql}],
                    benchmark_payload={
                        "db_id": task.metadata["db_id"],
                        "ground_truth_sql": str(task.ground_truth),
                    },
                    run_summary=run_summary,
                )
            )
            if on_run:
                on_run(runs[-1])
        except Exception as exc:
            runs.append(
                NativeRun(
                    trial=trial,
                    is_correct=False,
                    reward=0.0,
                    final_output="",
                    trace_text=f"ERROR: {exc}",
                    trace=[],
                    benchmark_payload={
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            if on_run:
                on_run(runs[-1])

    return _aggregate_native_task(
        task_id=task.task_id,
        benchmark="bird_sql",
        instruction=task.instruction,
        benchmark_metadata=_json_safe(task.metadata),
        runs=runs,
    )


def _load_benchmark_specs(cfg: DictConfig) -> dict[str, list[Any]]:
    """Load benchmark-native task specs in dataset order."""
    n = int(cfg.dataset.tasks_per_benchmark)
    tasks: dict[str, list[Any]] = {}
    bm = cfg.dataset.benchmarks

    if bm.tau_bench.enabled:
        tau_tasks = _load_tau_tasks(str(bm.tau_bench.env), str(bm.tau_bench.split))
        count = min(n, len(tau_tasks))
        tasks["tau_bench"] = [
            {
                "task_id": f"tau_{bm.tau_bench.env}_{bm.tau_bench.split}_{i:04d}",
                "task_index": i,
            }
            for i in range(count)
        ]

    if bm.bird_sql.enabled:
        from confidence_tom.benchmarks.bird_sql import load_bird_sql

        tasks["bird_sql"] = load_bird_sql(
            split=str(bm.bird_sql.split),
            num_samples=n,
        )

    if bm.plancraft.enabled:
        tasks["plancraft"] = _load_plancraft_examples(
            split=str(bm.plancraft.split),
            limit=n,
        )

    if bm.intercode.enabled:
        raise NotImplementedError("Native intercode dispatch is not implemented yet.")

    return tasks


class CheckpointManager:
    """Atomic, per-file checkpoint with resume support."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._locks: dict[str, asyncio.Lock] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._processed: dict[str, set[str]] = {}
        self._counters: dict[str, dict[str, int]] = {}

    def _key(self, benchmark: str, model_label: str) -> str:
        return f"{benchmark}/{model_label}"

    def _file(self, benchmark: str, model_label: str) -> Path:
        path = self.output_dir / benchmark / f"{model_label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def load(self, benchmark: str, model_label: str) -> int:
        key = self._key(benchmark, model_label)
        self._results[key] = []
        self._processed[key] = set()
        self._counters[key] = {"success": 0, "failed": 0, "skipped": 0}

        fp = self._file(benchmark, model_label)
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    existing = json.load(f)
                self._results[key] = existing
                self._processed[key] = {record["task_id"] for record in existing}
                self._counters[key]["skipped"] = len(existing)
                return len(existing)
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Corrupt checkpoint {fp}, starting fresh")
        return 0

    def is_done(self, benchmark: str, model_label: str, task_id: str) -> bool:
        return task_id in self._processed.get(self._key(benchmark, model_label), set())

    async def save(self, benchmark: str, model_label: str, record: dict[str, Any]) -> None:
        key = self._key(benchmark, model_label)
        fp = self._file(benchmark, model_label)
        async with self._lock(key):
            self._results[key].append(record)
            self._processed[key].add(str(record["task_id"]))
            self._counters[key]["success"] += 1
            tmp = fp.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._results[key], f, indent=2, ensure_ascii=False)
            tmp.replace(fp)

    async def fail(self, benchmark: str, model_label: str) -> None:
        key = self._key(benchmark, model_label)
        async with self._lock(key):
            self._counters[key]["failed"] += 1

    def counts(self, benchmark: str, model_label: str) -> dict[str, int]:
        return self._counters.get(self._key(benchmark, model_label), {})

    def total_done(self, benchmark: str, model_label: str) -> int:
        counts = self.counts(benchmark, model_label)
        return counts.get("success", 0) + counts.get("skipped", 0)

    def _partial_file(self, benchmark: str, model_label: str, task_id: str) -> Path:
        path = self.output_dir / benchmark / ".partial" / f"{model_label}__{task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_partial(self, benchmark: str, model_label: str, task_id: str) -> list[dict[str, Any]]:
        fp = self._partial_file(benchmark, model_label, task_id)
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning(f"Corrupt partial checkpoint {fp}, ignoring")
        return []

    def append_partial(self, benchmark: str, model_label: str, task_id: str, run: dict[str, Any]) -> None:
        fp = self._partial_file(benchmark, model_label, task_id)
        existing = self.load_partial(benchmark, model_label, task_id)
        existing.append(run)
        tmp = fp.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        tmp.replace(fp)

    def cleanup_partial(self, benchmark: str, model_label: str, task_id: str) -> None:
        fp = self._partial_file(benchmark, model_label, task_id)
        if fp.exists():
            fp.unlink()


def _task_id(task_spec: Any) -> str:
    return str(task_spec.task_id if hasattr(task_spec, "task_id") else task_spec["task_id"])


def _build_native_task_runner(
    benchmark: str,
    model_id: str,
    cfg: DictConfig,
    tool_use: bool = True,
) -> Callable[[Any], Awaitable[NativeTaskResult]]:
    k_samples = int(cfg.generator.k_samples)
    bm = cfg.dataset.benchmarks

    if benchmark == "bird_sql":
        client = LLMClient(
            model=model_id,
            temperature=float(cfg.generator.temperature),
            max_tokens=int(cfg.generator.max_tokens),
        )

        async def run_task(task: DynamicTask, on_run=None, preloaded_runs=None) -> NativeTaskResult:
            return await _run_bird_native_task(task, client, k_samples, on_run=on_run, preloaded_runs=preloaded_runs)

        return run_task

    if benchmark == "plancraft":
        client = LLMClient(
            model=model_id,
            temperature=float(cfg.generator.temperature),
            max_tokens=min(int(cfg.generator.max_tokens), 256),
        )

        async def run_task(task_spec: dict[str, Any], on_run=None, preloaded_runs=None) -> NativeTaskResult:
            return await _run_plancraft_native_task(task_spec, client, k_samples, on_run=on_run, preloaded_runs=preloaded_runs)

        return run_task

    if benchmark == "tau_bench":
        _ensure_tau_imports()
        from tau_bench.envs import get_env  # type: ignore
        from tau_bench.types import Action  # type: ignore

        tau_cfg = bm.tau_bench
        agent_client = LLMClient(
            model=model_id,
            temperature=float(cfg.generator.temperature),
            max_tokens=int(cfg.generator.max_tokens),
        )
        user_client = LLMClient(
            model=str(tau_cfg.get("user_model", "openai/gpt-4o-mini")),
            temperature=0.0,
            max_tokens=256,
        )

        async def run_task(task_spec: dict[str, Any], on_run=None, preloaded_runs=None) -> NativeTaskResult:
            enriched = dict(task_spec)
            enriched["model_id"] = model_id
            return await _run_tau_native_task(
                enriched,
                lambda *args, **kwargs: get_env(*args, **kwargs),
                Action,
                tau_cfg,
                agent_client,
                user_client,
                k_samples,
                log_prefix=f"[tau_bench/{model_id}/{task_spec['task_id']}]",
                use_tool_calling=tool_use,
                on_run=on_run,
                preloaded_runs=preloaded_runs,
            )

        return run_task

    raise ValueError(f"Unsupported benchmark for native dispatch: {benchmark}")


async def run_model(
    model_id: str,
    model_label: str,
    benchmark: str,
    tasks: list[Any],
    cfg: DictConfig,
    checkpoint: CheckpointManager,
    semaphore: asyncio.Semaphore,
    shutdown: asyncio.Event,
    tool_use: bool = True,
) -> None:
    loaded = checkpoint.load(benchmark, model_label)
    if loaded > 0:
        logger.info(f"[{benchmark}/{model_label}] Resumed: {loaded} already done")
    else:
        logger.info(f"[{benchmark}/{model_label}] Starting run")

    total = len(tasks)
    start = time.time()
    logger.info(
        f"[{benchmark}/{model_label}] Active model={model_id} | "
        f"tasks={total} | k_samples={int(cfg.generator.k_samples)} | mode=native"
    )
    run_native_task = _build_native_task_runner(benchmark, model_id, cfg, tool_use=tool_use)

    async def process_one(task_spec: Any) -> None:
        task_id = _task_id(task_spec)
        if shutdown.is_set():
            logger.warning(f"[{benchmark}/{model_label}] Stop requested; not starting {task_id}")
            return
        if checkpoint.is_done(benchmark, model_label, task_id):
            logger.info(f"[{benchmark}/{model_label}] SKIP {task_id} (already in checkpoint)")
            return

        partial_raw = checkpoint.load_partial(benchmark, model_label, task_id)
        preloaded: list[NativeRun] = []
        if partial_raw:
            try:
                preloaded = [NativeRun(**r) for r in partial_raw]
                logger.info(f"[{benchmark}/{model_label}] {task_id}: resuming from {len(preloaded)} partial runs")
            except Exception as exc:
                logger.warning(f"[{benchmark}/{model_label}] Failed to load partial runs for {task_id}: {exc}")

        def on_run(run: NativeRun) -> None:
            try:
                checkpoint.append_partial(benchmark, model_label, task_id, _json_safe(run.model_dump()))
                logger.info(f"[{benchmark}/{model_label}] {task_id}: saved partial run {run.trial + 1}")
            except Exception as exc:
                logger.warning(f"[{benchmark}/{model_label}] Failed to save partial run for {task_id}: {exc}")

        logger.info(f"[{benchmark}/{model_label}] RUN {task_id}")
        async with semaphore:
            try:
                result = await run_native_task(task_spec, on_run=on_run, preloaded_runs=preloaded or None)
                await checkpoint.save(benchmark, model_label, _serialize_native(result))
                checkpoint.cleanup_partial(benchmark, model_label, task_id)
                done = checkpoint.total_done(benchmark, model_label)
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                gap_display = "n/a" if result.gap is None else f"{result.gap:+.2f}"
                c_rep_display = "n/a" if result.c_rep is None else f"{result.c_rep:.2f}"
                logger.info(
                    f"[{benchmark}/{model_label}] [{done}/{total}] {task_id} | "
                    f"correct={result.majority_correct} | "
                    f"c_beh={result.c_beh:.2f} | "
                    f"c_rep={c_rep_display} | "
                    f"gap={gap_display} | "
                    f"ETA {eta:.0f}s"
                )
            except Exception:
                await checkpoint.fail(benchmark, model_label)
                logger.exception(f"[{benchmark}/{model_label}] FAILED: {task_id}")

    for task_spec in tasks:
        if shutdown.is_set():
            logger.warning(f"[{benchmark}/{model_label}] Shutdown requested; stopping after current checkpointed task")
            break
        await process_one(task_spec)
        if shutdown.is_set():
            logger.warning(f"[{benchmark}/{model_label}] Shutdown requested; exiting task loop")
            break

    counts = checkpoint.counts(benchmark, model_label)
    logger.info(
        f"[{benchmark}/{model_label}] Done in {time.time() - start:.0f}s | "
        f"success={counts.get('success', 0)} failed={counts.get('failed', 0)} skipped={counts.get('skipped', 0)}"
    )


def _fix_logging_encoding() -> None:
    """Patch all logging handlers to use UTF-8 so model emoji don't crash cp950 consoles.
    Uses reconfigure() in-place to avoid closing the underlying file descriptor."""
    for handler in logging.root.handlers:
        stream = getattr(handler, "stream", None)
        if stream is None:
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


@hydra.main(version_base="1.3", config_path="../configs", config_name="scale_experiment")
def main(cfg: DictConfig) -> None:
    _fix_logging_encoding()
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    all_tasks = _load_benchmark_specs(cfg)
    if not all_tasks:
        logger.error("No benchmarks enabled. Check dataset.benchmarks in config.")
        return

    total_tasks = sum(len(task_list) for task_list in all_tasks.values())
    logger.info(f"Loaded {total_tasks} native tasks across {len(all_tasks)} benchmarks")

    checkpoint = CheckpointManager(Path(cfg.output_dir))
    max_per_model = int(cfg.concurrency.max_per_model)
    sequential = bool(cfg.concurrency.get("sequential_models", False))

    shutdown = asyncio.Event()
    stop_signals = {"count": 0}

    def _handle_with_force(sig: int, frame: Any) -> None:
        stop_signals["count"] += 1
        if stop_signals["count"] == 1:
            logger.warning("Ctrl+C received: finishing the current in-flight task, saving checkpoints, then stopping...")
            shutdown.set()
            return
        logger.warning("Second Ctrl+C received: forcing immediate exit.")
        os._exit(130)

    signal.signal(signal.SIGINT, _handle_with_force)
    signal.signal(signal.SIGTERM, _handle_with_force)

    n_models = len(cfg.scale_models)
    n_benchmarks = len(all_tasks)
    max_tasks = max(len(task_list) for task_list in all_tasks.values())
    k = int(cfg.generator.k_samples)
    logger.info(
        f"Plan: {n_models} models x {n_benchmarks} benchmarks x "
        f"up to {max_tasks} tasks x K={k} native runs"
    )

    TOOL_USE_BENCHMARKS = {"tau_bench"}

    all_coroutines = []
    for benchmark, tasks in all_tasks.items():
        for model_cfg in cfg.scale_models:
            tool_use = bool(model_cfg.get("tool_use", True))
            if benchmark in TOOL_USE_BENCHMARKS and not tool_use:
                logger.info(
                    f"[{benchmark}/{model_cfg.label}] Skipping — model does not support tool use"
                )
                continue
            sem = asyncio.Semaphore(max_per_model)
            all_coroutines.append(
                run_model(
                    model_id=str(model_cfg.id),
                    model_label=str(model_cfg.label),
                    benchmark=benchmark,
                    tasks=tasks,
                    cfg=cfg,
                    checkpoint=checkpoint,
                    semaphore=sem,
                    shutdown=shutdown,
                    tool_use=tool_use,
                )
            )

    if sequential:
        for coro in all_coroutines:
            if shutdown.is_set():
                logger.warning("Shutdown requested before starting next benchmark/model worker")
                break
            await coro
    else:
        try:
            await asyncio.gather(*[asyncio.create_task(coro) for coro in all_coroutines])
        except asyncio.CancelledError:
            logger.warning("Cancelled; checkpoints already saved.")

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    for benchmark, tasks in all_tasks.items():
        for model_cfg in cfg.scale_models:
            label = str(model_cfg.label)
            tool_use = bool(model_cfg.get("tool_use", True))
            if benchmark in TOOL_USE_BENCHMARKS and not tool_use:
                logger.info(f"  {benchmark}/{label}: SKIPPED (no tool use)")
                continue
            done = checkpoint.total_done(benchmark, label)
            counts = checkpoint.counts(benchmark, label)
            logger.info(
                f"  {benchmark}/{label}: {done}/{len(tasks)} | "
                f"new={counts.get('success', 0)} failed={counts.get('failed', 0)} resumed={counts.get('skipped', 0)}"
            )

    logger.info(f"Results saved to: {cfg.output_dir}/")
    if shutdown.is_set():
        sys.exit(0)


if __name__ == "__main__":
    main()
