from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.paths import results_root

RESULTS_DIR = results_root()
PREDICTOR_CSV = RESULTS_DIR / "_prefix_predictor_v1" / "prefix_predictor_rows.csv"
OUT_DIR = RESULTS_DIR / "_prefix_embedding_v1"
OUT_ROWS = OUT_DIR / "pilot_rows.jsonl"
OUT_EMBEDDINGS = OUT_DIR / "pilot_embeddings.npz"
OUT_META = OUT_DIR / "pilot_meta.json"

EMBED_MODEL = "google/gemini-embedding-001"
MAX_PER_RUN = 40


def _find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        path
        for path in run_dir.glob("*.json")
        if path.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
    ]
    json_candidates = [path for path in candidates if "per_prefix_rows" not in path.name]
    if not json_candidates:
        raise FileNotFoundError(f"Could not find main result JSON in {run_dir}")
    return json_candidates[0]


def _load_prefix_text_index(run_name: str) -> dict[str, str]:
    result_json = _find_result_json(run_name)
    data = cast(list[dict[str, Any]], json.loads(result_json.read_text(encoding="utf-8")))
    index: dict[str, str] = {}
    for task in data:
        for step in task.get("prefix_oracle_steps", []):
            index[str(step["prefix_id"])] = str(step.get("prefix_text", ""))
    return index


def _stable_score(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_sample_rows() -> list[dict[str, object]]:
    with PREDICTOR_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_name"]].append(row)

    selected: list[dict[str, object]] = []
    for run_name, run_rows in sorted(by_run.items()):
        prefix_text_index = _load_prefix_text_index(run_name)
        ordered = sorted(
            run_rows,
            key=lambda row: _stable_score(run_name, row["task_id"], row["prefix_id"]),
        )
        kept = 0
        for row in ordered:
            prefix_text = prefix_text_index.get(str(row["prefix_id"]), "").strip()
            if not prefix_text:
                continue
            selected.append(
                {
                    "run_name": row["run_name"],
                    "benchmark": row["benchmark"],
                    "small_family": row["small_family"],
                    "large_family": row["large_family"],
                    "task_id": row["task_id"],
                    "prefix_id": row["prefix_id"],
                    "delta_t": float(row["delta_t"]),
                    "delta_positive": int(row["delta_positive"]),
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

    embeddings: list[list[float]] = []
    with OUT_ROWS.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            vec = client.embed_text(cast(str, row["prefix_text"]), model=EMBED_MODEL)
            embeddings.append(vec)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"embedded {i}/{len(rows)}")

    matrix = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(OUT_EMBEDDINGS, embeddings=matrix)

    meta = {
        "rows": len(rows),
        "embedding_model": EMBED_MODEL,
        "embedding_dim": int(matrix.shape[1]) if len(matrix) else 0,
        "max_per_run": MAX_PER_RUN,
        "run_count": len({row["run_name"] for row in rows}),
        "benchmark_counts": {
            key: sum(1 for row in rows if cast(str, row["benchmark"]) == key)
            for key in sorted({cast(str, row["benchmark"]) for row in rows})
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote rows to {OUT_ROWS}")
    print(f"Wrote embeddings to {OUT_EMBEDDINGS}")
    print(f"Wrote metadata to {OUT_META}")


if __name__ == "__main__":
    main()
