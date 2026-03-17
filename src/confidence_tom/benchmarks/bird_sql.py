"""BIRD-SQL task loader.

Loads text-to-SQL tasks from the BIRD benchmark.

Setup:
    1. Download BIRD dataset from https://bird-bench.github.io/
    2. Place data files under external/birdsql/bird/llm/data/
       Expected structure:
         data/dev/dev.json          -- question list
         data/dev/dev_databases/    -- SQLite database files

The agent receives a natural-language question + database schema and must
produce a valid SQL query. Correctness is evaluated by executing both the
predicted and ground-truth SQL and comparing result sets (execution match).
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from confidence_tom.task_models import DynamicTask

logger = logging.getLogger(__name__)

BIRD_DATA_DIR = Path(__file__).resolve().parents[3] / "external" / "birdsql" / "bird" / "llm" / "data"


def _get_schema(db_path: Path) -> str:
    """Extract CREATE TABLE statements from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    schema_lines = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schema_lines)


def load_bird_sql(
    split: str = "dev",
    num_samples: int = 50,
) -> list[DynamicTask]:
    """Load BIRD-SQL tasks as DynamicTask objects.

    Args:
        split: Dataset split ('dev' is the standard evaluation split).
        num_samples: Maximum number of tasks to load.

    Returns:
        List of DynamicTask ready for the agent runner.

    Raises:
        FileNotFoundError: If BIRD data has not been downloaded yet.
    """
    # Support both flat layout (data/dev/dev.json) and versioned layout
    # (data/dev/dev_20240627/dev.json) as shipped in the official download.
    flat_file = BIRD_DATA_DIR / split / f"{split}.json"
    versioned_dirs = sorted((BIRD_DATA_DIR / split).glob(f"{split}_*")) if (BIRD_DATA_DIR / split).exists() else []
    versioned_file = versioned_dirs[-1] / f"{split}.json" if versioned_dirs else None

    if flat_file.exists():
        questions_file = flat_file
        db_dir = BIRD_DATA_DIR / split / f"{split}_databases"
    elif versioned_file and versioned_file.exists():
        questions_file = versioned_file
        db_dir = versioned_file.parent / f"{split}_databases"
    else:
        raise FileNotFoundError(
            f"BIRD data not found under {BIRD_DATA_DIR / split}.\n"
            "Download from https://bird-bench.github.io/ and place under "
            "external/birdsql/bird/llm/data/"
        )

    with open(questions_file, encoding="utf-8") as f:
        items = json.load(f)

    tasks: list[DynamicTask] = []
    for i, item in enumerate(items[:num_samples]):
        db_id = item["db_id"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"

        schema: Optional[str] = None
        if db_path.exists():
            try:
                schema = _get_schema(db_path)
            except Exception as e:
                logger.warning(f"Could not read schema for {db_id}: {e}")

        instruction = f"Database: {db_id}\n\nQuestion: {item['question']}"
        if item.get("evidence"):
            instruction += f"\n\nHint: {item['evidence']}"
        if schema:
            instruction += f"\n\nSchema:\n{schema}"
        instruction += "\n\nYour final_answer must be a valid SQLite SQL query that answers the question. Output only the SQL query string, nothing else."

        tasks.append(
            DynamicTask(
                task_id=f"bird_{split}_{i:04d}",
                benchmark="bird-sql",
                instruction=instruction,
                ground_truth=item["SQL"],
                metadata={
                    "db_id": db_id,
                    "db_path": str(db_path),
                    "split": split,
                    "difficulty": item.get("difficulty", "unknown"),
                },
            )
        )

    logger.info(f"Loaded {len(tasks)} BIRD-SQL {split} tasks")
    return tasks


def evaluate_sql(predicted_sql: str, ground_truth_sql: str, db_path: str) -> bool:
    """Evaluate a predicted SQL by comparing execution results.

    Returns True if the result sets match (execution match metric).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(predicted_sql)
        pred_rows = set(map(tuple, cursor.fetchall()))

        cursor.execute(ground_truth_sql)
        gt_rows = set(map(tuple, cursor.fetchall()))

        conn.close()
        return pred_rows == gt_rows
    except Exception as e:
        logger.debug(f"SQL evaluation error: {e}")
        return False
