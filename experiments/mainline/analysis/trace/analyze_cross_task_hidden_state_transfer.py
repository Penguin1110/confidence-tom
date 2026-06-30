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
OUT_DIR = ROOT / "results" / "_cross_task_hidden_state_transfer_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "cross_task_hidden_state_transfer.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_positive = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


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


def _merge_family(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}
    taxonomy = _task_taxonomy(reentry_rows)
    merged: list[dict[str, Any]] = []
    for row in reentry_rows:
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        meta = taxonomy[str(row["task_id"])]
        merged.append(
            {
                **row,
                **meta,
                "mean_hidden": np.asarray(probe["mean_pool_hidden"], dtype=np.float64),
                "last_token_hidden": np.asarray(probe["last_token_hidden"], dtype=np.float64),
            }
        )
    return merged


def _build_task_records(rows: list[dict[str, Any]], vector_key: str) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    records: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        if not bool(task_rows[0]["full_trace_correct"]):
            continue
        y = np.asarray(
            [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows], dtype=np.int64
        )
        if len(np.unique(y)) < 2:
            continue
        vecs = [cast(np.ndarray, row[vector_key]) for row in task_rows]
        correct_vecs = [vec for vec, flag in zip(vecs, y) if flag == 1]
        if not correct_vecs:
            continue
        records.append(
            {
                "task_id": task_id,
                "category": str(task_rows[0]["category"]),
                "first_correct_frac": float(task_rows[0]["first_correct_frac"]),
                "local_correct_rate": float(task_rows[0]["local_correct_rate"]),
                "labels": y,
                "vectors": vecs,
                "final_correct_vector": correct_vecs[-1],
                "correct_centroid": np.mean(np.stack(correct_vecs, axis=0), axis=0),
            }
        )
    return records


def _score_target(source_vec: np.ndarray, target: dict[str, Any]) -> dict[str, float]:
    scores = np.asarray(
        [_cosine(cast(np.ndarray, vec), source_vec) for vec in target["vectors"]], dtype=np.float64
    )
    y = cast(np.ndarray, target["labels"]).astype(np.int64)
    return {
        "auroc": _roc_auc_score(y, scores),
        "score_gap": float(np.mean(scores[y == 1]) - np.mean(scores[y == 0])),
        "mean_score_correct": float(np.mean(scores[y == 1])),
        "mean_score_incorrect": float(np.mean(scores[y == 0])),
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vector_reports: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        tasks = _build_task_records(rows, vector_key)
        pair_rows: list[dict[str, Any]] = []
        for source in tasks:
            for target in tasks:
                for prototype_name in ["final_correct_vector", "correct_centroid"]:
                    metrics = _score_target(cast(np.ndarray, source[prototype_name]), target)
                    pair_rows.append(
                        {
                            "source_task_id": source["task_id"],
                            "target_task_id": target["task_id"],
                            "prototype": prototype_name,
                            "same_task": int(source["task_id"] == target["task_id"]),
                            "source_category": source["category"],
                            "target_category": target["category"],
                            **metrics,
                        }
                    )
        same = [row for row in pair_rows if row["same_task"] == 1]
        cross = [row for row in pair_rows if row["same_task"] == 0]
        vector_reports[vector_key] = {
            "tasks": len(tasks),
            "same_task_mean_auroc": float(np.mean([row["auroc"] for row in same])) if same else 0.5,
            "cross_task_mean_auroc": float(np.mean([row["auroc"] for row in cross]))
            if cross
            else 0.5,
            "cross_task_median_auroc": float(np.median([row["auroc"] for row in cross]))
            if cross
            else 0.5,
            "cross_task_share_auroc_gt_0_6": float(np.mean([row["auroc"] > 0.6 for row in cross]))
            if cross
            else 0.0,
            "cross_task_share_auroc_gt_0_7": float(np.mean([row["auroc"] > 0.7 for row in cross]))
            if cross
            else 0.0,
            "same_vs_cross_gap": (
                float(
                    np.mean([row["auroc"] for row in same])
                    - np.mean([row["auroc"] for row in cross])
                )
                if same and cross
                else 0.0
            ),
            "prototype_breakdown": {},
            "top_cross_pairs": sorted(cross, key=lambda row: float(row["auroc"]), reverse=True)[
                :10
            ],
        }
        for prototype_name in ["final_correct_vector", "correct_centroid"]:
            block = [row for row in cross if row["prototype"] == prototype_name]
            vector_reports[vector_key]["prototype_breakdown"][prototype_name] = {
                "cross_task_mean_auroc": float(np.mean([row["auroc"] for row in block]))
                if block
                else 0.5,
                "cross_task_share_auroc_gt_0_7": float(
                    np.mean([row["auroc"] > 0.7 for row in block])
                )
                if block
                else 0.0,
                "cross_task_mean_score_gap": float(np.mean([row["score_gap"] for row in block]))
                if block
                else 0.0,
            }
    return vector_reports


def _aggregate(families: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        rows = [family[vector_key] for family in families.values() if vector_key in family]
        out[vector_key] = {
            "mean_same_task_auroc": float(np.mean([row["same_task_mean_auroc"] for row in rows]))
            if rows
            else 0.5,
            "mean_cross_task_auroc": float(np.mean([row["cross_task_mean_auroc"] for row in rows]))
            if rows
            else 0.5,
            "mean_same_vs_cross_gap": float(np.mean([row["same_vs_cross_gap"] for row in rows]))
            if rows
            else 0.0,
            "mean_cross_share_auroc_gt_0_6": float(
                np.mean([row["cross_task_share_auroc_gt_0_6"] for row in rows])
            )
            if rows
            else 0.0,
            "mean_cross_share_auroc_gt_0_7": float(
                np.mean([row["cross_task_share_auroc_gt_0_7"] for row in rows])
            )
            if rows
            else 0.0,
        }
    return out


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, Any], summary["families"])
    aggregate = cast(dict[str, Any], summary["aggregate"])
    lines = [
        "# Cross-Task Hidden-State Transfer",
        "",
        "## Question",
        "",
        "- Can hidden states be compared across different tasks at all?",
        "- If task A is correct, can a score derived from task A's correct-state geometry predict the correct/incorrect prefix distribution of task B?",
        "",
        "## Aggregate",
        "",
        "| Pooling | Mean same-task AUROC | Mean cross-task AUROC | Mean same-vs-cross gap | Cross-task share AUROC > 0.6 | Cross-task share AUROC > 0.7 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        block = aggregate[vector_key]
        lines.append(
            f"| {vector_key} | {block['mean_same_task_auroc']:.3f} | {block['mean_cross_task_auroc']:.3f} | "
            f"{block['mean_same_vs_cross_gap']:.3f} | {block['mean_cross_share_auroc_gt_0_6']:.3f} | "
            f"{block['mean_cross_share_auroc_gt_0_7']:.3f} |"
        )
    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        lines.append(f"### `{family}`")
        lines.append("")
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            block = report[vector_key]
            lines.append(f"#### `{vector_key}`")
            lines.append("")
            lines.append(f"- correct tasks usable for transfer: `{block['tasks']}`")
            lines.append(f"- same-task mean AUROC: `{block['same_task_mean_auroc']:.3f}`")
            lines.append(f"- cross-task mean AUROC: `{block['cross_task_mean_auroc']:.3f}`")
            lines.append(f"- same-vs-cross gap: `{block['same_vs_cross_gap']:.3f}`")
            lines.append(
                f"- cross-task share AUROC > 0.7: `{block['cross_task_share_auroc_gt_0_7']:.3f}`"
            )
            lines.append("")
            lines.append(
                "| Prototype | Cross-task mean AUROC | Cross-task share AUROC > 0.7 | Mean score gap |"
            )
            lines.append("| --- | ---: | ---: | ---: |")
            for prototype_name, proto in block["prototype_breakdown"].items():
                lines.append(
                    f"| {prototype_name} | {proto['cross_task_mean_auroc']:.3f} | "
                    f"{proto['cross_task_share_auroc_gt_0_7']:.3f} | {proto['cross_task_mean_score_gap']:.3f} |"
                )
            lines.append("")
        lines.append("")
    lines += [
        "## Read",
        "",
        "- If cross-task AUROC is much above 0.5, then task-level correct-state geometry is at least partially shared across tasks.",
        "- If same-task AUROC is much higher than cross-task AUROC, then hidden-state scores are only weakly transferable and remain task-specific.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, Any] = {}
    for probe_path in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family_dir = probe_path.parent.name
        reentry_path = REENTRY_DIR / family_dir / "reentry_rows.jsonl"
        if not reentry_path.exists():
            continue
        rows = _merge_family(probe_path, reentry_path)
        if not rows:
            continue
        family = family_dir.replace("_reentry_livebench_local_v1_", "")
        families[family] = _family_analysis(rows)
    summary = {
        "source_dir": str(DATA_DIR),
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
