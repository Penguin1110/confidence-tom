"""tau-bench task loader.

Loads tasks from the local tau-bench checkout at external/tau-bench.
Supports both 'retail' and 'airline' environments.

Setup:
    pip install -e external/tau-bench
"""

import logging
import sys
from pathlib import Path
from typing import Literal

from confidence_tom.task_models import DynamicTask

logger = logging.getLogger(__name__)

TAU_BENCH_DIR = Path(__file__).resolve().parents[3] / "external" / "tau-bench"


def _load_env_assets(env: Literal["retail", "airline"], split: str):
    """Load tau-bench tasks plus environment rules/tools for prompting."""
    if env == "retail":
        from tau_bench.envs.retail.rules import RULES
        from tau_bench.envs.retail.tools import ALL_TOOLS

        if split == "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks_raw
        elif split == "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks_raw
        else:
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks_raw
        return tasks_raw, RULES, ALL_TOOLS

    from tau_bench.envs.airline.rules import RULES
    from tau_bench.envs.airline.tools import ALL_TOOLS
    from tau_bench.envs.airline.tasks_test import TASKS_TEST as tasks_raw
    return tasks_raw, RULES, ALL_TOOLS


def _format_tool_catalog(tool_classes: list[type]) -> str:
    """Render a concise tool catalog from tau-bench tool schemas."""
    blocks: list[str] = []
    for tool_cls in tool_classes:
        info = tool_cls.get_info().get("function", {})
        name = info.get("name", "")
        description = info.get("description", "").strip()
        params = info.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []

        lines = [f"- {name}: {description}"]
        if props:
            arg_parts = []
            for key, spec in props.items():
                label = key
                if key in required:
                    label += " (required)"
                arg_desc = spec.get("description", "").strip()
                if arg_desc:
                    arg_parts.append(f"{label} - {arg_desc}")
                else:
                    arg_parts.append(label)
            lines.append("  Args: " + "; ".join(arg_parts))
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _build_tau_instruction(
    user_instruction: str,
    env: Literal["retail", "airline"],
    rules: list[str],
    tool_classes: list[type],
) -> str:
    """Build a benchmark-faithful instruction for the generator."""
    tools_text = _format_tool_catalog(tool_classes)
    rules_text = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"You are solving a tau-bench {env} environment task.\n\n"
        "Environment constraints:\n"
        "- You must operate only through the tau-bench environment tools listed below.\n"
        "- Do not use external websites, search engines, email, phone calls, customer-service chats, or invented systems.\n"
        "- Do not invent product names, order outcomes, prices, fees, confirmations, or tool observations.\n"
        "- If a backend-changing action is needed, follow the environment policy exactly.\n"
        "- In trajectory.action, write one tau-bench tool call at a time using the exact tool name and concrete arguments.\n"
        "- In observation, only state what the tool returned or what the user explicitly said.\n"
        "- In final_answer, summarize the completed in-environment result only; if the task cannot be completed from available evidence, say so briefly.\n\n"
        f"Environment rules:\n{rules_text}\n\n"
        f"Available tools:\n{tools_text}\n\n"
        f"User task:\n{user_instruction}"
    )


def _ensure_tau_bench_on_path() -> None:
    if str(TAU_BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(TAU_BENCH_DIR))


def load_tau_bench(
    env: Literal["retail", "airline"] = "retail",
    split: Literal["train", "test", "dev"] = "test",
    num_samples: int = 50,
) -> list[DynamicTask]:
    """Load tau-bench tasks as DynamicTask objects.

    Args:
        env: Which environment to load ('retail' or 'airline').
        split: Task split to use.
        num_samples: Maximum number of tasks to load.

    Returns:
        List of DynamicTask ready for the agent runner.
    """
    _ensure_tau_bench_on_path()

    try:
        tasks_raw, rules, tool_classes = _load_env_assets(env, split)
    except ImportError as e:
        raise ImportError(
            f"tau-bench not found. Run: pip install -e external/tau-bench\n({e})"
        ) from e

    tasks: list[DynamicTask] = []
    for i, t in enumerate(tasks_raw[:num_samples]):
        instruction = _build_tau_instruction(
            user_instruction=t.instruction,
            env=env,
            rules=rules,
            tool_classes=tool_classes,
        )
        tasks.append(
            DynamicTask(
                task_id=f"tau_{env}_{split}_{i:04d}",
                benchmark="tau-bench",
                instruction=instruction,
                ground_truth={
                    "actions": [a.model_dump() for a in t.actions],
                    "outputs": t.outputs,
                },
                metadata={
                    "env": env,
                    "split": split,
                    "user_id": t.user_id,
                    "task_index": i,
                    "rules": list(rules),
                    "tool_names": [tool.get_info()["function"]["name"] for tool in tool_classes],
                },
            )
        )

    logger.info(f"Loaded {len(tasks)} tau-bench {env}/{split} tasks")
    return tasks
