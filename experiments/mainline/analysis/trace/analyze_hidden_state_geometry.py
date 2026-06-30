from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
PROBE_DIR = DATA_DIR / "probe"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_hidden_state_geometry_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "trace" / "hidden_state_geometry.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _family_probe_paths() -> dict[str, tuple[Path, Path]]:
    mapping: dict[str, tuple[Path, Path]] = {}
    for probe_rows in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family_dir = probe_rows.parent.name
        reentry_rows = REENTRY_DIR / family_dir / "reentry_rows.jsonl"
        if reentry_rows.exists():
            mapping[family_dir] = (probe_rows, reentry_rows)
    return mapping


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _task_taxonomy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        step_count = len(seq)
        any_small_correct = any(seq)
        first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
        first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
        local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
        last_small_correct = bool(seq[-1]) if seq else False
        full_correct = bool(task_rows[0].get("full_trace_correct", 0))
        if full_correct:
            if (
                any_small_correct
                and first_correct_frac <= EARLY_FRACTION
                and local_correct_rate >= MIN_STABLE_LOCAL_RATE
            ):
                category = "stable-success"
            else:
                category = "late-success"
        else:
            category = "late-failure" if any_small_correct else "persistent-failure"
        out[task_id] = {
            "category": category,
            "full_trace_correct": int(full_correct),
            "any_small_correct": int(any_small_correct),
            "first_correct_frac": float(first_correct_frac),
            "local_correct_rate": float(local_correct_rate),
            "last_small_correct": int(last_small_correct),
        }
    return out


def _merged_rows(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}
    taxonomy = _task_taxonomy(reentry_rows)
    merged: list[dict[str, Any]] = []
    for row in reentry_rows:
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        merged.append(
            {
                **row,
                **taxonomy[str(row["task_id"])],
                "mean_hidden": np.asarray(probe["mean_pool_hidden"], dtype=np.float64),
                "last_token_hidden": np.asarray(probe["last_token_hidden"], dtype=np.float64),
                "selected_layer": int(probe["selected_layer"]),
                "hidden_dim": int(probe["hidden_dim"]),
            }
        )
    return merged


def _task_geometry(rows: list[dict[str, Any]], vector_key: str) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    out: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        vecs = [cast(np.ndarray, row[vector_key]) for row in task_rows]
        last_vec = vecs[-1]
        norms = [float(np.linalg.norm(v)) for v in vecs]
        consecutive = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        to_final = [_cosine(v, last_vec) for v in vecs[:-1]]
        displacement = [float(np.linalg.norm(vecs[i + 1] - vecs[i])) for i in range(len(vecs) - 1)]
        out.append(
            {
                "task_id": task_id,
                "category": task_rows[0]["category"],
                "full_trace_correct": int(task_rows[0]["full_trace_correct"]),
                "first_correct_frac": float(task_rows[0]["first_correct_frac"]),
                "local_correct_rate": float(task_rows[0]["local_correct_rate"]),
                "last_small_correct": int(task_rows[0]["last_small_correct"]),
                "mean_norm": float(np.mean(norms)),
                "final_norm": float(norms[-1]),
                "mean_consecutive_cosine": float(np.mean(consecutive)) if consecutive else 1.0,
                "min_consecutive_cosine": float(np.min(consecutive)) if consecutive else 1.0,
                "mean_cosine_to_final": float(np.mean(to_final)) if to_final else 1.0,
                "first_cosine_to_final": float(to_final[0]) if to_final else 1.0,
                "mean_displacement": float(np.mean(displacement)) if displacement else 0.0,
                "max_displacement": float(np.max(displacement)) if displacement else 0.0,
                "steps": len(task_rows),
            }
        )
    return out


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(rows),
        "mean_first_correct_frac": _mean([float(r["first_correct_frac"]) for r in rows]),
        "mean_local_correct_rate": _mean([float(r["local_correct_rate"]) for r in rows]),
        "mean_last_small_correct": _mean([float(r["last_small_correct"]) for r in rows]),
        "mean_norm": _mean([float(r["mean_norm"]) for r in rows]),
        "final_norm": _mean([float(r["final_norm"]) for r in rows]),
        "mean_consecutive_cosine": _mean([float(r["mean_consecutive_cosine"]) for r in rows]),
        "min_consecutive_cosine": _mean([float(r["min_consecutive_cosine"]) for r in rows]),
        "mean_cosine_to_final": _mean([float(r["mean_cosine_to_final"]) for r in rows]),
        "first_cosine_to_final": _mean([float(r["first_cosine_to_final"]) for r in rows]),
        "mean_displacement": _mean([float(r["mean_displacement"]) for r in rows]),
        "max_displacement": _mean([float(r["max_displacement"]) for r in rows]),
        "mean_steps": _mean([float(r["steps"]) for r in rows]),
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": len(rows),
        "tasks": len({str(r["task_id"]) for r in rows}),
        "selected_layer": int(rows[0]["selected_layer"]),
        "hidden_dim": int(rows[0]["hidden_dim"]),
        "category_counts": dict(Counter(str(r["category"]) for r in rows)),
        "vectors": {},
    }
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        task_rows = _task_geometry(rows, vector_key)
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_cat[str(row["category"])].append(row)
        report["vectors"][vector_key] = {
            "overall": _summarize_block(task_rows),
            "by_category": {cat: _summarize_block(block) for cat, block in by_cat.items()},
            "late_pair_gap": {
                key: (
                    float(
                        _summarize_block(by_cat["late-success"]).get(key, 0.0)
                        - _summarize_block(by_cat["late-failure"]).get(key, 0.0)
                    )
                    if by_cat["late-success"] and by_cat["late-failure"]
                    else 0.0
                )
                for key in [
                    "mean_first_correct_frac",
                    "mean_local_correct_rate",
                    "mean_last_small_correct",
                    "mean_consecutive_cosine",
                    "mean_cosine_to_final",
                    "mean_displacement",
                    "max_displacement",
                ]
            },
        }
    return report


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        gap_keys = [
            "mean_first_correct_frac",
            "mean_local_correct_rate",
            "mean_last_small_correct",
            "mean_consecutive_cosine",
            "mean_cosine_to_final",
            "mean_displacement",
            "max_displacement",
        ]
        out[vector_key] = {
            key: float(
                np.mean(
                    [
                        float(family["vectors"][vector_key]["late_pair_gap"][key])
                        for family in families.values()
                    ]
                )
            )
            for key in gap_keys
        }
    return out


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, dict[str, Any]], summary["families"])
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    lines = [
        "# Hidden State Geometry",
        "",
        "## Question",
        "",
        "- Treat hidden states as trajectories across prefixes.",
        "- Compare state stability, drift, and similarity-to-final across trace categories.",
        "",
        "## Aggregate Late-Success Minus Late-Failure Gaps",
        "",
        "| Pooling | First-correct frac gap | Local correct rate gap | Last-small-correct gap | Consecutive cosine gap | Cosine-to-final gap | Mean displacement gap | Max displacement gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        gap = aggregate[vector_key]
        lines.append(
            f"| {vector_key} | {gap['mean_first_correct_frac']:.3f} | {gap['mean_local_correct_rate']:.3f} | "
            f"{gap['mean_last_small_correct']:.3f} | {gap['mean_consecutive_cosine']:.3f} | "
            f"{gap['mean_cosine_to_final']:.3f} | {gap['mean_displacement']:.3f} | "
            f"{gap['max_displacement']:.3f} |"
        )
    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        lines += [
            f"### `{family}`",
            "",
            f"- tasks: `{report['tasks']}`",
            f"- selected layer: `{report['selected_layer']}`",
            f"- hidden dim: `{report['hidden_dim']}`",
            f"- category counts: `{json.dumps(report['category_counts'], ensure_ascii=False)}`",
            "",
        ]
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            by_cat = cast(dict[str, Any], report["vectors"][vector_key]["by_category"])
            lines += [
                f"#### `{vector_key}`",
                "",
                "| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Mean consecutive cosine | Mean cosine-to-final | Mean displacement |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for cat in ["stable-success", "late-success", "late-failure", "persistent-failure"]:
                if cat not in by_cat:
                    continue
                row = by_cat[cat]
                lines.append(
                    f"| {cat} | {row['tasks']} | {row['mean_first_correct_frac']:.3f} | "
                    f"{row['mean_local_correct_rate']:.3f} | {row['mean_last_small_correct']:.3f} | "
                    f"{row['mean_consecutive_cosine']:.3f} | {row['mean_cosine_to_final']:.3f} | "
                    f"{row['mean_displacement']:.3f} |"
                )
            gap = report["vectors"][vector_key]["late_pair_gap"]
            lines += [
                "",
                "- late-success minus late-failure:",
                f"  first-correct frac `{gap['mean_first_correct_frac']:.3f}`, local correct rate `{gap['mean_local_correct_rate']:.3f}`, last-small-correct `{gap['mean_last_small_correct']:.3f}`",
                f"  consecutive cosine `{gap['mean_consecutive_cosine']:.3f}`, cosine-to-final `{gap['mean_cosine_to_final']:.3f}`, mean displacement `{gap['mean_displacement']:.3f}`, max displacement `{gap['max_displacement']:.3f}`",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    families: dict[str, dict[str, Any]] = {}
    for family_dir, (probe_path, reentry_path) in _family_probe_paths().items():
        rows = _merged_rows(probe_path, reentry_path)
        if rows:
            families[family_dir.replace("_reentry_livebench_local_v1_", "")] = _family_analysis(
                rows
            )
    summary = {
        "source_dir": str(DATA_DIR),
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
