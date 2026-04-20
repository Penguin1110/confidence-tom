from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from confidence_tom.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
ROWS_CSV = EARLY_DIR / "early_decision_rows.csv"
OUT_DIR = RESULTS_DIR / "_early_decision_representation_v1"
OUT_ROWS = OUT_DIR / "rows.jsonl"
OUT_EMBEDDINGS = OUT_DIR / "representations.npz"
OUT_META = OUT_DIR / "meta.json"

REPRESENTATION_MODEL = "google/gemini-embedding-001"
MAX_PER_RUN = 120


def _find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        path
        for path in run_dir.glob("*.json")
        if path.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
        and "per_prefix_rows" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find main result JSON in {run_dir}")
    if len(candidates) > 1:
        candidates = sorted(candidates)
    return candidates[0]


def _load_prefix_text_index(run_name: str) -> dict[str, str]:
    data = json.loads(_find_result_json(run_name).read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for task in data:
        for step in task.get("prefix_oracle_steps", []):
            index[str(step.get("prefix_id", ""))] = str(step.get("prefix_text", ""))
    return index


def _stable_score(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _load_task_level_index(path: Path, key_field: str) -> dict[tuple[str, str], object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("task_groups") or data.get("rows_detail") or []
    index: dict[tuple[str, str], object] = {}
    for row in rows:
        index[(str(row["run_name"]), str(row["task_id"]))] = row.get(key_field)
    return index


def _load_sample_rows() -> list[dict[str, object]]:
    if not ROWS_CSV.exists():
        raise FileNotFoundError(f"Missing early decision rows: {ROWS_CSV}")

    msp_index = _load_task_level_index(
        EARLY_DIR / "minimal_sufficient_prefix_analysis.json", "minimal_sufficient_step"
    )
    alignment_path = EARLY_DIR / "early_decision_takeover_alignment.json"
    alignment_raw = (
        json.loads(alignment_path.read_text(encoding="utf-8"))
        if alignment_path.exists()
        else {"rows_detail": []}
    )
    positive_index: dict[tuple[str, str], int] = {}
    for row in alignment_raw.get("rows_detail", []):
        positive_index[(str(row["run_name"]), str(row["task_id"]))] = int(
            bool(row.get("positive_steps"))
        )

    with ROWS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_name"]].append(row)

    selected: list[dict[str, object]] = []
    for run_name, run_rows in sorted(by_run.items()):
        prefix_text_index = _load_prefix_text_index(run_name)
        ordered = sorted(
            run_rows, key=lambda row: _stable_score(run_name, row["task_id"], row["prefix_id"])
        )
        kept = 0
        for row in ordered:
            prefix_text = prefix_text_index.get(str(row["prefix_id"]), "").strip()
            if not prefix_text:
                continue
            key = (row["run_name"], row["task_id"])
            selected.append(
                {
                    "run_name": row["run_name"],
                    "benchmark": row["benchmark"],
                    "small_family": row["small_family"],
                    "large_family": row["large_family"],
                    "task_id": row["task_id"],
                    "prefix_id": row["prefix_id"],
                    "step_index": int(float(row["step_index"])),
                    "step_bucket": row["step_bucket"],
                    "small_full_trace_success": int(row["small_full_trace_success"]),
                    "msp_exists": int(msp_index.get(key) is not None),
                    "task_has_positive_takeover": int(positive_index.get(key, 0)),
                    "prefix_text": prefix_text,
                }
            )
            kept += 1
            if kept >= MAX_PER_RUN:
                break
    return selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_sample_rows()
    client = LLMClient(model="openai/gpt-5.4")

    vectors: list[list[float]] = []
    with OUT_ROWS.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            vec = client.embed_text(str(row["prefix_text"]), model=REPRESENTATION_MODEL)
            vectors.append(vec)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"embedded {i}/{len(rows)}")

    matrix = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(OUT_EMBEDDINGS, representations=matrix)

    meta = {
        "rows": len(rows),
        "representation_type": "api_embedding",
        "representation_model": REPRESENTATION_MODEL,
        "representation_dim": int(matrix.shape[1]) if len(matrix) else 0,
        "max_per_run": MAX_PER_RUN,
        "run_count": len({row["run_name"] for row in rows}),
        "benchmark_counts": {
            key: sum(1 for row in rows if row["benchmark"] == key)
            for key in sorted({row["benchmark"] for row in rows})
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote rows to {OUT_ROWS}")
    print(f"Wrote representations to {OUT_EMBEDDINGS}")
    print(f"Wrote metadata to {OUT_META}")


if __name__ == "__main__":
    main()
