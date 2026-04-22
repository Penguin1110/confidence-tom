"""Plancraft task loader.

Loads Minecraft crafting tasks from the Plancraft benchmark.

Setup:
    pip install plancraft
    (Windows: run tools/setup_dynamic_benchmarks.py --benchmarks plancraft
     to also apply the path-separator bug patch)
"""

import logging

from confidence_tom.data.task_models import DynamicTask

logger = logging.getLogger(__name__)


def load_plancraft(
    split: str = "test",
    num_samples: int = 50,
) -> list[DynamicTask]:
    """Load Plancraft tasks as DynamicTask objects.

    Each task asks the agent to craft a target item given a starting inventory.
    Correctness = target item appears in the final inventory.

    Args:
        split: Dataset split ('train', 'val', 'test').
        num_samples: Maximum number of tasks to load.

    Returns:
        List of DynamicTask ready for the agent runner.

    Raises:
        ImportError: If plancraft is not installed.
    """
    try:
        from plancraft.simple import get_plancraft_examples
    except ImportError as e:
        raise ImportError(
            "plancraft not installed. Run: pip install plancraft\n"
            "(On Windows also run: python tools/setup_dynamic_benchmarks.py "
            "--benchmarks plancraft)\n"
            f"({e})"
        ) from e

    all_examples = get_plancraft_examples(split=split)
    # Filter out impossible tasks — they have no valid crafting path
    feasible = [ex for ex in all_examples if not ex.impossible]

    tasks: list[DynamicTask] = []
    for i, ex in enumerate(feasible[:num_samples]):
        target = ex.target
        # ex.inventory is a plain dict {item_name: quantity}
        inventory_desc = ", ".join(f"{item} x{qty}" for item, qty in ex.inventory.items())
        instruction = (
            f"You are playing Minecraft. Craft the following item: {target}.\n"
            f"Your current inventory: {inventory_desc or 'empty'}.\n"
            "List the crafting steps needed and state your final answer as the target item name."
        )

        tasks.append(
            DynamicTask(
                task_id=f"plancraft_{split}_{ex.id}",
                benchmark="plancraft",
                instruction=instruction,
                ground_truth=target,
                metadata={
                    "split": split,
                    "target_item": target,
                    "initial_inventory": dict(ex.inventory),
                    "example_index": i,
                    "example_id": ex.id,
                },
            )
        )

    logger.info(f"Loaded {len(tasks)} Plancraft {split} tasks")
    return tasks
