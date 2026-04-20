from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _bucket(delta: float, eps: float) -> str:
    if delta > eps:
        return "positive"
    if delta < -eps:
        return "negative"
    return "zero"


@hydra.main(version_base=None, config_path="../configs", config_name="oracle_gain_mapping")
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    files = sorted(output_dir.glob("*.json"))
    if not files:
        print(f"No oracle-gain result files found under {output_dir}")
        return

    eps = float(cfg.get("analysis", {}).get("delta_epsilon", 1e-6))
    total_tasks = 0
    total_steps = 0
    base_correct = 0
    delta_counter: Counter[str] = Counter()
    positive_tasks: set[str] = set()
    per_task_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for file_path in files:
        rows = _load_rows(file_path)
        print(f"\n== {file_path.name} ==")
        print(f"tasks: {len(rows)}")

        for row in rows:
            total_tasks += 1
            task_id = row["task_id"]
            if row.get("base_small_correct"):
                base_correct += 1

            steps = row.get("oracle_gain_steps", [])
            total_steps += len(steps)
            for step in steps:
                bucket = _bucket(float(step.get("delta_correctness", 0.0)), eps)
                delta_counter[bucket] += 1
                per_task_counts[task_id][bucket] += 1
                if bucket == "positive":
                    positive_tasks.add(task_id)

        file_step_counter = Counter()
        for row in rows:
            for step in row.get("oracle_gain_steps", []):
                file_step_counter[_bucket(float(step.get("delta_correctness", 0.0)), eps)] += 1
        print(
            "step deltas:",
            dict(
                positive=file_step_counter["positive"],
                zero=file_step_counter["zero"],
                negative=file_step_counter["negative"],
            ),
        )

    print("\n== Overall ==")
    print(f"tasks: {total_tasks}")
    print(f"base_small_correct: {base_correct}/{total_tasks}")
    print(f"oracle_steps: {total_steps}")
    print(
        "delta buckets:",
        dict(
            positive=delta_counter["positive"],
            zero=delta_counter["zero"],
            negative=delta_counter["negative"],
        ),
    )
    print(f"tasks_with_any_positive_gain: {len(positive_tasks)}/{total_tasks}")

    if per_task_counts:
        print("\n== Per-task positive-gain summary ==")
        for task_id, counts in sorted(
            per_task_counts.items(),
            key=lambda kv: (-kv[1]["positive"], kv[0]),
        ):
            print(f"{task_id}: +{counts['positive']} / 0={counts['zero']} / -{counts['negative']}")


if __name__ == "__main__":
    main()
