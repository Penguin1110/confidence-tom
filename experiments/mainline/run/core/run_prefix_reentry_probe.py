from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import (
    load_aime_2024,
    load_gpqa_diamond,
    load_livebench_reasoning,
    load_math500,
    load_olympiadbench,
)
from confidence_tom.infra.paths import project_root, results_root
from confidence_tom.infra.representations import extract_prompt_representation

ROOT = project_root()
RESULTS_DIR = results_root()

REENTRY_SYSTEM_PROMPT = """You are the same small worker continuing an existing reasoning prefix.
Continue naturally from the given prefix and finish the problem.
Do not restart from scratch unless the prefix is unusable.
Do not restate the full problem or rebuild the derivation from the beginning if
the prefix already contains useful structure.
Do not use JSON.
End with a final line exactly in the form:
Final Answer: <answer>
"""

LOCAL_MODEL_BY_FAMILY = {
    "qwen": "Qwen/Qwen3-14B",
    "qwen3": "Qwen/Qwen3-14B",
    "qwen25": "Qwen/Qwen2.5-14B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-E4B-it",
    "gemma3": "google/gemma-3-4b-it",
    "mistral": "mistralai/Ministral-8B-Instruct-2410",
    "mistralai": "mistralai/Ministral-8B-Instruct-2410",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo": "allenai/olmo-2-13b-instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def _parse_family_model_map(entries: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in entries or []:
        item = raw.strip()
        if not item:
            continue
        family, sep, model_name = item.partition("=")
        if not sep or not family.strip() or not model_name.strip():
            raise ValueError(
                f"Invalid --local-model-map entry: {raw!r}. Expected FAMILY=MODEL."
            )
        mapping[family.strip().lower()] = model_name.strip()
    return mapping


def _resolve_local_model_name(
    small_family: str,
    explicit: str | None,
    family_model_map: dict[str, str],
) -> str:
    if explicit:
        return explicit
    normalized = small_family.lower()
    if normalized in family_model_map:
        return family_model_map[normalized]
    if normalized in LOCAL_MODEL_BY_FAMILY:
        return LOCAL_MODEL_BY_FAMILY[normalized]
    raise ValueError(f"Missing local model mapping for family={small_family!r}")


def _benchmark_from_task_id(task_id: str) -> str:
    if task_id.startswith("livebench_reasoning_"):
        return "livebench_reasoning"
    if task_id.startswith("aime_2024_"):
        return "aime_2024"
    if task_id.startswith("math500_"):
        return "math500"
    if task_id.startswith("gpqa_diamond_"):
        return "gpqa_diamond"
    return "olympiadbench"


def _load_task_map(benchmark: str) -> dict[str, StaticTask]:
    if benchmark == "olympiadbench":
        tasks = load_olympiadbench(num_samples=50)
    elif benchmark == "livebench_reasoning":
        tasks = load_livebench_reasoning(num_samples=30)
    elif benchmark == "aime_2024":
        tasks = load_aime_2024(num_samples=30)
    elif benchmark == "math500":
        tasks = load_math500(num_samples=50)
    elif benchmark == "gpqa_diamond":
        tasks = load_gpqa_diamond(num_samples=40)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return {task.id: task for task in tasks}


def _build_reentry_messages(question: str, prefix_text: str) -> list[dict[str, str]]:
    content = f"Problem:\n{question}\n\nReasoning prefix:\n{prefix_text}\n\nContinue and finish."
    return [
        {"role": "system", "content": REENTRY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _load_rows(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        rows.append(row)
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract transformer hidden-state and attention summaries for re-entry rows."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["transformers"], default="transformers")
    parser.add_argument("--local-model-name", default=None)
    parser.add_argument("--local-model-map", action="append", default=[])
    parser.add_argument("--selected-layer", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows = output_dir / "reentry_probe_rows.jsonl"
    out_summary = output_dir / "reentry_probe_summary.json"

    rows = _load_rows(rows_path, args.max_rows)
    family_model_map = _parse_family_model_map(args.local_model_map)
    task_maps: dict[str, dict[str, StaticTask]] = {}
    written: list[dict[str, Any]] = []

    with out_rows.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            benchmark = str(row.get("benchmark") or _benchmark_from_task_id(str(row["task_id"])))
            task_map = task_maps.setdefault(benchmark, _load_task_map(benchmark))
            task = task_map[str(row["task_id"])]
            prefix_text = str(row["prefix_text"])
            model_name = _resolve_local_model_name(
                str(row.get("small_family", "")),
                args.local_model_name,
                family_model_map,
            )
            probe = extract_prompt_representation(
                model_name=model_name,
                trust_remote_code=bool(args.trust_remote_code),
                messages=_build_reentry_messages(task.question, prefix_text),
                prefix_text=prefix_text,
                selected_layer=int(args.selected_layer),
            )
            result = {
                "run_name": row["run_name"],
                "benchmark": benchmark,
                "task_id": row["task_id"],
                "prefix_id": row["prefix_id"],
                "small_family": row.get("small_family", ""),
                "local_model_name": model_name,
                **probe,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written.append(result)
            print(f"processed {idx}/{len(rows)} :: {row['run_name']} :: {row['prefix_id']}")

    summary = {
        "rows": len(written),
        "backend": args.backend,
        "selected_layer": int(args.selected_layer),
        "models": sorted({str(row["local_model_name"]) for row in written}),
        "benchmarks": sorted({str(row["benchmark"]) for row in written}),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote rows to {out_rows}")
    print(f"Wrote summary to {out_summary}")


if __name__ == "__main__":
    main()
