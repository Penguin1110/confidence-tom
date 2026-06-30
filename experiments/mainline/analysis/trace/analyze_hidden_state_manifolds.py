from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
PROBE_DIR = DATA_DIR / "probe"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_hidden_state_manifolds_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "trace" / "hidden_state_manifolds.md"
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


def _step_bin(step_index: int, step_count: int) -> str:
    frac = step_index / max(1, step_count)
    if frac <= 0.25:
        return "q1"
    if frac <= 0.5:
        return "q2"
    if frac <= 0.75:
        return "q3"
    return "q4"


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
        task_meta = taxonomy[str(row["task_id"])]
        merged.append(
            {
                **row,
                **task_meta,
                "mean_hidden": np.asarray(probe["mean_pool_hidden"], dtype=np.float64),
                "last_token_hidden": np.asarray(probe["last_token_hidden"], dtype=np.float64),
                "selected_layer": int(probe["selected_layer"]),
                "hidden_dim": int(probe["hidden_dim"]),
            }
        )
    return merged


def _centroid(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(1, dtype=np.float64)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def _task_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    final_rows: list[dict[str, Any]] = []
    for task_rows in by_task.values():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        final_rows.append(task_rows[-1])
    return final_rows


def _build_centroids(final_rows: list[dict[str, Any]], vector_key: str) -> dict[str, np.ndarray]:
    by_cat: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in final_rows:
        by_cat[str(row["category"])].append(cast(np.ndarray, row[vector_key]))
        by_cat["success" if int(row["full_trace_correct"]) else "failure"].append(
            cast(np.ndarray, row[vector_key])
        )
    centroids = {key: _centroid(vals) for key, vals in by_cat.items()}
    return centroids


def _score_rows(
    rows: list[dict[str, Any]],
    vector_key: str,
    centroids: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    scored: list[dict[str, Any]] = []
    for task_rows in by_task.values():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        step_count = len(task_rows)
        for row in task_rows:
            vec = cast(np.ndarray, row[vector_key])
            rec = {
                "task_id": str(row["task_id"]),
                "category": str(row["category"]),
                "full_trace_correct": int(row["full_trace_correct"]),
                "step_index": int(row["step_index"]),
                "step_count": step_count,
                "step_bin": _step_bin(int(row["step_index"]), step_count),
            }
            for name, centroid in centroids.items():
                rec[f"sim_{name}"] = _cosine(vec, centroid)
            rec["margin_success_failure"] = rec["sim_success"] - rec["sim_failure"]
            if "stable-success" in centroids:
                rec["margin_stable_persistent"] = rec["sim_stable-success"] - rec.get(
                    "sim_persistent-failure", 0.0
                )
            scored.append(rec)
    return scored


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _summarize_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    keys = [key for key in rows[0].keys() if key.startswith("sim_")] + [
        "margin_success_failure",
        "margin_stable_persistent",
    ]
    out: dict[str, Any] = {"rows": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[key] = _mean(vals)
    return out


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": len(rows),
        "tasks": len({str(r["task_id"]) for r in rows}),
        "selected_layer": int(rows[0]["selected_layer"]),
        "hidden_dim": int(rows[0]["hidden_dim"]),
        "vectors": {},
    }
    final_rows = _task_final_rows(rows)
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        centroids = _build_centroids(final_rows, vector_key)
        scored = _score_rows(rows, vector_key, centroids)
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_cat_bin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_cat[str(row["category"])].append(row)
            by_cat_bin[(str(row["category"]), str(row["step_bin"]))].append(row)
        report["vectors"][vector_key] = {
            "centroid_names": sorted(centroids.keys()),
            "by_category": {cat: _summarize_block(block) for cat, block in by_cat.items()},
            "by_category_step_bin": {
                f"{cat}:{step_bin}": _summarize_block(block)
                for (cat, step_bin), block in by_cat_bin.items()
            },
        }
    return report


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        late_success = []
        late_failure = []
        for family in families.values():
            by_cat = family["vectors"][vector_key]["by_category"]
            if "late-success" in by_cat and "late-failure" in by_cat:
                late_success.append(by_cat["late-success"])
                late_failure.append(by_cat["late-failure"])
        if not late_success:
            out[vector_key] = {}
            continue
        out[vector_key] = {
            "late_success_minus_late_failure": {
                "sim_success": _mean(
                    [
                        float(a["sim_success"]) - float(b["sim_success"])
                        for a, b in zip(late_success, late_failure)
                    ]
                ),
                "sim_failure": _mean(
                    [
                        float(a["sim_failure"]) - float(b["sim_failure"])
                        for a, b in zip(late_success, late_failure)
                    ]
                ),
                "margin_success_failure": _mean(
                    [
                        float(a["margin_success_failure"]) - float(b["margin_success_failure"])
                        for a, b in zip(late_success, late_failure)
                    ]
                ),
                "sim_stable-success": _mean(
                    [
                        float(a.get("sim_stable-success", 0.0))
                        - float(b.get("sim_stable-success", 0.0))
                        for a, b in zip(late_success, late_failure)
                    ]
                ),
            }
        }
    return out


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, dict[str, Any]], summary["families"])
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    lines = [
        "# Hidden State Manifolds",
        "",
        "## Idea",
        "",
        "- Build manifolds from final-step hidden states.",
        "- Score every prefix by similarity to success/failure and category-specific final-state centroids.",
        "",
        "## Aggregate Late-Success Minus Late-Failure",
        "",
        "| Pooling | Delta sim(success) | Delta sim(failure) | Delta success-failure margin | Delta sim(stable-success) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        gap = aggregate[vector_key]["late_success_minus_late_failure"]
        lines.append(
            f"| {vector_key} | {gap['sim_success']:.3f} | {gap['sim_failure']:.3f} | "
            f"{gap['margin_success_failure']:.3f} | {gap['sim_stable-success']:.3f} |"
        )
    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        lines += [
            f"### `{family}`",
            "",
            f"- tasks: `{report['tasks']}`",
            f"- selected layer: `{report['selected_layer']}`",
            f"- hidden dim: `{report['hidden_dim']}`",
            "",
        ]
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            by_cat = cast(dict[str, Any], report["vectors"][vector_key]["by_category"])
            lines += [
                f"#### `{vector_key}`",
                "",
                "| Category | Rows | sim(success) | sim(failure) | success-failure margin | sim(stable-success) | sim(late-success) | sim(late-failure) | sim(persistent-failure) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for cat in ["stable-success", "late-success", "late-failure", "persistent-failure"]:
                if cat not in by_cat:
                    continue
                row = by_cat[cat]
                lines.append(
                    f"| {cat} | {row['rows']} | {row.get('sim_success', 0.0):.3f} | "
                    f"{row.get('sim_failure', 0.0):.3f} | {row.get('margin_success_failure', 0.0):.3f} | "
                    f"{row.get('sim_stable-success', 0.0):.3f} | {row.get('sim_late-success', 0.0):.3f} | "
                    f"{row.get('sim_late-failure', 0.0):.3f} | {row.get('sim_persistent-failure', 0.0):.3f} |"
                )
            lines += [
                "",
                "| Category:StepBin | sim(success) | sim(failure) | margin |",
                "| --- | ---: | ---: | ---: |",
            ]
            by_cat_bin = cast(dict[str, Any], report["vectors"][vector_key]["by_category_step_bin"])
            for key in sorted(by_cat_bin):
                row = by_cat_bin[key]
                lines.append(
                    f"| {key} | {row.get('sim_success', 0.0):.3f} | {row.get('sim_failure', 0.0):.3f} | "
                    f"{row.get('margin_success_failure', 0.0):.3f} |"
                )
            lines.append("")
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
