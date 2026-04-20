from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _load_completed_count(run_dir: Path) -> tuple[int | None, str]:
    for path in sorted(run_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            return len(data), path.name
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            return len(data["tasks"]), path.name
    return None, "none"


def _load_partial_rows(run_dir: Path) -> list[dict[str, Any]]:
    partial_dir = run_dir / "partials"
    rows: list[dict[str, Any]] = []
    if not partial_dir.exists():
        return rows
    now = time.time()
    for path in sorted(partial_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        age_sec = int(now - path.stat().st_mtime)
        rows.append(
            {
                "name": path.name,
                "age_sec": age_sec,
                "status": payload.get("status"),
                "completed_step_index": payload.get("completed_step_index"),
                "task_id": payload.get("task_id"),
            }
        )
    rows.sort(key=lambda row: row["age_sec"])
    return rows


@hydra.main(version_base=None, config_path="../configs", config_name="prefix_family_sweep")
def main(cfg: DictConfig) -> None:
    root = Path(to_absolute_path("."))
    limit = int(cfg.dataset.limit)
    small_workers = cfg.small_workers
    large_workers = cfg.large_workers

    print(f"Prefix Sweep Progress (limit={limit})")
    print("-" * 80)

    for small in small_workers:
        for large in large_workers:
            run_name = f"{small.family}_to_{large.family}_{limit}"
            run_dir = root / "results" / run_name
            if not run_dir.exists():
                print(f"{run_name:28} missing")
                continue

            completed, filename = _load_completed_count(run_dir)
            partial_rows = _load_partial_rows(run_dir)
            completed_text = "?" if completed is None else str(completed)
            print(
                f"{run_name:28} completed={completed_text:>3}/{limit:<3} "
                f"partials={len(partial_rows):>2} file={filename}"
            )
            for row in partial_rows[:4]:
                print(
                    "  "
                    f"{row['task_id']} "
                    f"status={row['status']} "
                    f"step={row['completed_step_index']} "
                    f"age={row['age_sec']}s"
                )


if __name__ == "__main__":
    main()
