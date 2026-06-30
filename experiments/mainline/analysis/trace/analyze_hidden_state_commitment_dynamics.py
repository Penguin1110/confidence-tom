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
OUT_DIR = ROOT / "results" / "_hidden_state_commitment_dynamics_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "hidden_state_commitment_dynamics.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5
FINAL_LOCK_PROGRESS = 0.90


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


def _centroid(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _task_category(full_correct: bool, seq: list[int]) -> str:
    step_count = len(seq)
    any_correct = any(seq)
    first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
    first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
    local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
    if full_correct:
        if (
            any_correct
            and first_correct_frac <= EARLY_FRACTION
            and local_correct_rate >= MIN_STABLE_LOCAL_RATE
        ):
            return "stable-success"
        return "late-success"
    return "late-failure" if any_correct else "persistent-failure"


def _merge_family(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reentry_rows:
        by_task[str(row["task_id"])].append(row)

    taxonomy: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda row: int(row["step_index"]))
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        full_correct = bool(task_rows[0].get("full_trace_correct", 0))
        first_correct_step = next((idx + 1 for idx, flag in enumerate(seq) if flag), None)
        taxonomy[task_id] = {
            "category": _task_category(full_correct, seq),
            "full_trace_correct": int(full_correct),
            "first_correct_frac": (first_correct_step / len(seq)) if first_correct_step else 1.0,
            "local_correct_rate": float(sum(seq) / len(seq)),
            "last_small_correct": int(seq[-1]),
            "seq": seq,
        }

    merged: list[dict[str, Any]] = []
    for row in reentry_rows:
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        meta = taxonomy[str(row["task_id"])]
        merged.append(
            {
                **row,
                **{k: v for k, v in meta.items() if k != "seq"},
                "mean_hidden": np.asarray(probe["mean_pool_hidden"], dtype=np.float64),
                "last_token_hidden": np.asarray(probe["last_token_hidden"], dtype=np.float64),
                "selected_layer": int(probe["selected_layer"]),
                "hidden_dim": int(probe["hidden_dim"]),
            }
        )
    return merged


def _task_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    return {
        task_id: sorted(block, key=lambda row: int(row["step_index"]))
        for task_id, block in by_task.items()
    }


def _final_rows(by_task: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {task_id: rows[-1] for task_id, rows in by_task.items()}


def _loo_margin(
    task_id: str,
    vec: np.ndarray,
    final_rows: dict[str, dict[str, Any]],
    vector_key: str,
) -> float | None:
    success_vecs: list[np.ndarray] = []
    failure_vecs: list[np.ndarray] = []
    for other_id, row in final_rows.items():
        if other_id == task_id:
            continue
        target = success_vecs if int(row["full_trace_correct"]) else failure_vecs
        target.append(cast(np.ndarray, row[vector_key]))
    success = _centroid(success_vecs)
    failure = _centroid(failure_vecs)
    if success is None or failure is None:
        return None
    return _cosine(vec, success) - _cosine(vec, failure)


def _loo_final_margins(
    final_rows: dict[str, dict[str, Any]],
    vector_key: str,
) -> dict[str, float]:
    margins: dict[str, float] = {}
    for task_id, row in final_rows.items():
        margin = _loo_margin(task_id, cast(np.ndarray, row[vector_key]), final_rows, vector_key)
        if margin is not None:
            margins[task_id] = margin
    return margins


def _stable_first(values: list[float], predicate: Any) -> int | None:
    for idx in range(len(values)):
        if all(predicate(value) for value in values[idx:]):
            return idx
    return None


def _first_stable_binary(seq: list[int], value: int) -> int | None:
    for idx in range(len(seq)):
        if all(flag == value for flag in seq[idx:]):
            return idx
    return None


def _final_lock_step(vecs: list[np.ndarray]) -> tuple[int | None, list[float]]:
    if not vecs:
        return None, []
    final_vec = vecs[-1]
    sims = [_cosine(vec, final_vec) for vec in vecs]
    first = sims[0]
    final = sims[-1]
    denom = final - first
    if abs(denom) < 1e-9:
        progress = [1.0 for _ in sims]
    else:
        progress = [(sim - first) / denom for sim in sims]
    lock = _stable_first(progress, lambda value: value >= FINAL_LOCK_PROGRESS)
    return lock, sims


def _family_threshold(
    final_margins: dict[str, float], final_rows: dict[str, dict[str, Any]]
) -> float:
    success = [
        margin
        for task_id, margin in final_margins.items()
        if int(final_rows[task_id]["full_trace_correct"]) == 1
    ]
    failure = [
        margin
        for task_id, margin in final_margins.items()
        if int(final_rows[task_id]["full_trace_correct"]) == 0
    ]
    if not success or not failure:
        return 0.0
    return float((np.median(success) + np.median(failure)) / 2.0)


def _median_or_zero(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _task_commitment_records(
    rows: list[dict[str, Any]], vector_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task = _task_rows(rows)
    final_rows = _final_rows(by_task)
    final_margins = _loo_final_margins(final_rows, vector_key)
    threshold = _family_threshold(final_margins, final_rows)

    records: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        vecs = [cast(np.ndarray, row[vector_key]) for row in task_rows]
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        margins = [_loo_margin(task_id, vec, final_rows, vector_key) for vec in vecs]
        if any(margin is None for margin in margins):
            continue
        typed_margins = [float(margin) for margin in margins if margin is not None]
        success_lock = _stable_first(typed_margins, lambda value: value >= threshold)
        failure_lock = _stable_first(typed_margins, lambda value: value < threshold)
        final_lock, final_sims = _final_lock_step(vecs)
        correct_lock = _first_stable_binary(seq, 1)
        incorrect_lock = _first_stable_binary(seq, 0)
        full_correct = int(task_rows[0]["full_trace_correct"])

        candidates: list[tuple[str, int]] = []
        if success_lock is not None:
            candidates.append(("success", success_lock))
        if failure_lock is not None:
            candidates.append(("failure", failure_lock))
        direction = "uncommitted"
        commit_idx: int | None = None
        if candidates:
            direction, commit_idx = sorted(candidates, key=lambda item: item[1])[0]

        expected_direction = "success" if full_correct else "failure"
        early_failure_lock = int(
            direction == "failure"
            and commit_idx is not None
            and ((commit_idx + 1) / len(seq)) <= EARLY_FRACTION
        )
        premature_wrong_lock = int(
            direction != "uncommitted"
            and direction != expected_direction
            and commit_idx is not None
            and ((commit_idx + 1) / len(seq)) <= EARLY_FRACTION
        )
        records.append(
            {
                "task_id": task_id,
                "category": str(task_rows[0]["category"]),
                "full_trace_correct": full_correct,
                "steps": len(seq),
                "first_correct_frac": float(task_rows[0]["first_correct_frac"]),
                "local_correct_rate": float(task_rows[0]["local_correct_rate"]),
                "last_small_correct": int(task_rows[0]["last_small_correct"]),
                "margin_threshold": threshold,
                "initial_margin": typed_margins[0],
                "final_margin": typed_margins[-1],
                "margin_delta": typed_margins[-1] - typed_margins[0],
                "commit_direction": direction,
                "commit_frac": ((commit_idx + 1) / len(seq)) if commit_idx is not None else 1.0,
                "commit_matches_full_outcome": int(direction == expected_direction),
                "early_failure_lock": early_failure_lock,
                "premature_wrong_lock": premature_wrong_lock,
                "success_lock_frac": ((success_lock + 1) / len(seq))
                if success_lock is not None
                else 1.0,
                "failure_lock_frac": ((failure_lock + 1) / len(seq))
                if failure_lock is not None
                else 1.0,
                "final_state_lock_frac": ((final_lock + 1) / len(seq))
                if final_lock is not None
                else 1.0,
                "correctness_lock_frac": ((correct_lock + 1) / len(seq))
                if correct_lock is not None
                else 1.0,
                "incorrectness_lock_frac": ((incorrect_lock + 1) / len(seq))
                if incorrect_lock is not None
                else 1.0,
                "mean_final_similarity": _mean(final_sims),
            }
        )

    threshold_meta = {
        "margin_threshold": threshold,
        "final_success_margin_mean": _mean(
            [
                margin
                for task_id, margin in final_margins.items()
                if int(final_rows[task_id]["full_trace_correct"]) == 1
            ]
        ),
        "final_failure_margin_mean": _mean(
            [
                margin
                for task_id, margin in final_margins.items()
                if int(final_rows[task_id]["full_trace_correct"]) == 0
            ]
        ),
        "final_success_margin_median": _median_or_zero(
            [
                margin
                for task_id, margin in final_margins.items()
                if int(final_rows[task_id]["full_trace_correct"]) == 1
            ]
        ),
        "final_failure_margin_median": _median_or_zero(
            [
                margin
                for task_id, margin in final_margins.items()
                if int(final_rows[task_id]["full_trace_correct"]) == 0
            ]
        ),
    }
    return records, threshold_meta


def _summarize_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(records),
        "full_correct_rate": _mean([float(record["full_trace_correct"]) for record in records]),
        "mean_first_correct_frac": _mean(
            [float(record["first_correct_frac"]) for record in records]
        ),
        "mean_commit_frac": _mean([float(record["commit_frac"]) for record in records]),
        "mean_success_lock_frac": _mean([float(record["success_lock_frac"]) for record in records]),
        "mean_failure_lock_frac": _mean([float(record["failure_lock_frac"]) for record in records]),
        "mean_final_state_lock_frac": _mean(
            [float(record["final_state_lock_frac"]) for record in records]
        ),
        "mean_correctness_lock_frac": _mean(
            [float(record["correctness_lock_frac"]) for record in records]
        ),
        "commit_matches_full_outcome_rate": _mean(
            [float(record["commit_matches_full_outcome"]) for record in records]
        ),
        "early_failure_lock_rate": _mean(
            [float(record["early_failure_lock"]) for record in records]
        ),
        "premature_wrong_lock_rate": _mean(
            [float(record["premature_wrong_lock"]) for record in records]
        ),
        "mean_margin_delta": _mean([float(record["margin_delta"]) for record in records]),
        "commit_direction_counts": dict(
            Counter(str(record["commit_direction"]) for record in records)
        ),
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": len(rows),
        "tasks": len({str(row["task_id"]) for row in rows}),
        "selected_layer": int(rows[0]["selected_layer"]),
        "hidden_dim": int(rows[0]["hidden_dim"]),
        "vectors": {},
    }
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        records, threshold_meta = _task_commitment_records(rows, vector_key)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_full: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_category[str(record["category"])].append(record)
            by_full["full_correct" if int(record["full_trace_correct"]) else "full_wrong"].append(
                record
            )

        early_wrong_examples = sorted(
            [
                record
                for record in records
                if int(record["full_trace_correct"]) == 0 and int(record["early_failure_lock"]) == 1
            ],
            key=lambda record: (float(record["commit_frac"]), str(record["task_id"])),
        )[:8]
        report["vectors"][vector_key] = {
            "threshold_meta": threshold_meta,
            "overall": _summarize_block(records),
            "by_category": {key: _summarize_block(value) for key, value in by_category.items()},
            "by_full_outcome": {key: _summarize_block(value) for key, value in by_full.items()},
            "early_failure_lock_examples": early_wrong_examples,
        }
    return report


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        family_blocks = [family["vectors"][vector_key] for family in families.values()]
        out[vector_key] = {
            "mean_commit_frac": _mean(
                [float(block["overall"]["mean_commit_frac"]) for block in family_blocks]
            ),
            "mean_final_state_lock_frac": _mean(
                [float(block["overall"]["mean_final_state_lock_frac"]) for block in family_blocks]
            ),
            "mean_commit_matches_full_outcome_rate": _mean(
                [
                    float(block["overall"]["commit_matches_full_outcome_rate"])
                    for block in family_blocks
                ]
            ),
            "mean_early_failure_lock_rate": _mean(
                [float(block["overall"]["early_failure_lock_rate"]) for block in family_blocks]
            ),
            "mean_full_wrong_early_failure_lock_rate": _mean(
                [
                    float(
                        block["by_full_outcome"]
                        .get("full_wrong", {})
                        .get("early_failure_lock_rate", 0.0)
                    )
                    for block in family_blocks
                ]
            ),
            "mean_full_correct_success_lock_frac": _mean(
                [
                    float(
                        block["by_full_outcome"]
                        .get("full_correct", {})
                        .get("mean_success_lock_frac", 1.0)
                    )
                    for block in family_blocks
                ]
            ),
        }
    return out


def _short_task(task_id: str) -> str:
    return task_id.replace("livebench_reasoning_", "")[:12]


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, dict[str, Any]], summary["families"])
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    lines = [
        "# Hidden-State Commitment Dynamics",
        "",
        "## Question",
        "",
        "- Ask when hidden states become committed, not just whether a final answer is correct.",
        "- Use a leave-one-task-out success-vs-failure margin as a cross-task commitment signal.",
        "- Also compute a task-internal final-state lock time as a trajectory convergence signal.",
        "",
        "## Aggregate",
        "",
        "| Pooling | Commit frac | Final-state lock frac | Commit matches full outcome | Early failure-lock rate | Full-wrong early failure-lock rate | Full-correct success-lock frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        block = aggregate[vector_key]
        lines.append(
            f"| {vector_key} | {block['mean_commit_frac']:.3f} | "
            f"{block['mean_final_state_lock_frac']:.3f} | "
            f"{block['mean_commit_matches_full_outcome_rate']:.3f} | "
            f"{block['mean_early_failure_lock_rate']:.3f} | "
            f"{block['mean_full_wrong_early_failure_lock_rate']:.3f} | "
            f"{block['mean_full_correct_success_lock_frac']:.3f} |"
        )

    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        lines.append(f"### `{family}`")
        lines.append("")
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            block = report["vectors"][vector_key]
            overall = block["overall"]
            threshold = block["threshold_meta"]
            lines.append(f"#### `{vector_key}`")
            lines.append("")
            lines.append(f"- tasks: `{overall['tasks']}`")
            lines.append(f"- margin threshold: `{threshold['margin_threshold']:.4f}`")
            lines.append(
                f"- final success/failure margin mean: "
                f"`{threshold['final_success_margin_mean']:.4f}` / `{threshold['final_failure_margin_mean']:.4f}`"
            )
            lines.append(f"- mean commit frac: `{overall['mean_commit_frac']:.3f}`")
            lines.append(
                f"- mean final-state lock frac: `{overall['mean_final_state_lock_frac']:.3f}`"
            )
            lines.append(f"- early failure-lock rate: `{overall['early_failure_lock_rate']:.3f}`")
            lines.append("")
            lines.append(
                "| Category | Tasks | Commit frac | Success-lock frac | Failure-lock frac | Final-state lock frac | Early failure-lock | Margin delta |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            by_category = block["by_category"]
            for category in [
                "stable-success",
                "late-success",
                "late-failure",
                "persistent-failure",
            ]:
                cat = by_category.get(category)
                if not cat:
                    continue
                lines.append(
                    f"| {category} | {cat['tasks']} | {cat['mean_commit_frac']:.3f} | "
                    f"{cat['mean_success_lock_frac']:.3f} | {cat['mean_failure_lock_frac']:.3f} | "
                    f"{cat['mean_final_state_lock_frac']:.3f} | {cat['early_failure_lock_rate']:.3f} | "
                    f"{cat['mean_margin_delta']:.3f} |"
                )
            examples = block["early_failure_lock_examples"]
            if examples:
                lines.append("")
                lines.append("Early failure-lock examples:")
                for example in examples[:4]:
                    lines.append(
                        f"- `{_short_task(str(example['task_id']))}` ({example['category']}): "
                        f"commit_frac={example['commit_frac']:.3f}, "
                        f"first_correct_frac={example['first_correct_frac']:.3f}, "
                        f"margin_delta={example['margin_delta']:.3f}"
                    )
            lines.append("")

    lines += [
        "## Read",
        "",
        "- `commit_frac` is the first point where the cross-task margin stays on one side of the success/failure threshold.",
        "- `final-state lock frac` is the first point where the trajectory has reached 90% of its own final-state similarity progress and never drops below it.",
        "- `early failure-lock` is the proxy for premature convergence to a wrong state: the trajectory locks onto the failure side within the first third.",
        "- This is still observational. A causal version would intervene before vs after the lock point and test reversibility.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, dict[str, Any]] = {}
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
        "final_lock_progress": FINAL_LOCK_PROGRESS,
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
