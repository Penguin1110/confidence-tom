"""InterCode task loader.

Loads bash/SQL tasks from the InterCode benchmark.

Setup:
    pip install intercode-bench
    Docker must be running (environments are containerised).
"""

import logging

from confidence_tom.data.task_models import DynamicTask

logger = logging.getLogger(__name__)


def load_intercode(
    env: str = "bash",
    num_samples: int = 50,
) -> list[DynamicTask]:
    """Load InterCode tasks as DynamicTask objects.

    The agent must execute shell commands or SQL queries to accomplish a goal.
    Correctness is evaluated by the InterCode environment after the agent
    submits its final answer.

    Args:
        env: Environment type ('bash' or 'sql').
        num_samples: Maximum number of tasks to load.

    Returns:
        List of DynamicTask ready for the agent runner.

    Raises:
        ImportError: If intercode-bench is not installed.
        RuntimeError: If Docker is not running.
    """
    try:
        import docker
        from intercode.envs import BashEnv, SqlEnv
    except ImportError as e:
        raise ImportError(
            "intercode-bench not installed or Docker unavailable.\n"
            "Run: pip install intercode-bench  (Docker must be running)\n"
            f"({e})"
        ) from e

    # Verify Docker is reachable before loading tasks
    try:
        docker.from_env().ping()
    except Exception as e:
        raise RuntimeError(
            f"Docker is not running or not reachable: {e}\nStart Docker and retry."
        ) from e

    EnvClass = BashEnv if env == "bash" else SqlEnv
    ic_env = EnvClass(image_name=None, data_path=None)  # uses defaults
    raw_tasks = ic_env.data[:num_samples]

    tasks: list[DynamicTask] = []
    for i, item in enumerate(raw_tasks):
        query = item.get("query", item.get("instruction", ""))
        gold = item.get("gold", item.get("answer", ""))

        tasks.append(
            DynamicTask(
                task_id=f"intercode_{env}_{i:04d}",
                benchmark="intercode",
                instruction=query,
                ground_truth=gold,
                metadata={
                    "env": env,
                    "task_index": i,
                },
            )
        )

    logger.info(f"Loaded {len(tasks)} InterCode {env} tasks")
    return tasks
