from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_trace_persistence_subtypes_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "trace_persistence_subtypes.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_category(full_correct: bool, seq: list[int]) -> str:
    step_count = len(seq)
    any_small_correct = any(seq)
    first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
    first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
    local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
    if full_correct:
        if (
            any_small_correct
            and first_correct_frac <= EARLY_FRACTION
            and local_correct_rate >= MIN_STABLE_LOCAL_RATE
        ):
            return "stable-success"
        return "late-success"
    return "late-failure" if any_small_correct else "persistent-failure"


def _streaks(seq: list[int]) -> list[int]:
    out: list[int] = []
    run = 0
    for flag in seq:
        if flag:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def _subtype(category: str, seq: list[int], full_correct: bool) -> str:
    if category == "stable-success":
        return "early-lock-in"
    if category == "persistent-failure":
        return "never-correct"

    first = seq.index(1)
    tail = seq[first:]
    collapses = sum(1 for i in range(len(tail) - 1) if tail[i] == 1 and tail[i + 1] == 0)
    total_correct = sum(seq)
    last = seq[-1]

    if category == "late-success":
        if all(flag == 1 for flag in tail):
            return "delayed-lock-in"
        if last == 1:
            return "oscillatory-success"
        return "anomalous-late-success"

    if last == 1:
        return "terminal-near-miss"
    if total_correct == 1:
        return "fragile-flash"
    if collapses > 0:
        return "rescue-then-collapse"
    return "partial-runout"


def _normalize_bin(pos: float) -> str:
    if pos <= 0.25:
        return "q1"
    if pos <= 0.50:
        return "q2"
    if pos <= 0.75:
        return "q3"
    return "q4"


def _task_record(family: str, task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["step_index"]))
    seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in rows]
    full_correct = bool(rows[0].get("full_trace_correct", 0))
    category = _task_category(full_correct, seq)
    first_idx = seq.index(1) if any(seq) else None
    first_frac = ((first_idx + 1) / len(seq)) if first_idx is not None else 1.0
    tail = seq[first_idx:] if first_idx is not None else []
    tail_rate = float(np.mean(tail)) if tail else 0.0
    collapse_count = (
        sum(1 for i in range(len(tail) - 1) if tail[i] == 1 and tail[i + 1] == 0) if tail else 0
    )
    longest_streak = max(_streaks(seq), default=0)
    streak_after_onset = max(_streaks(tail), default=0) if tail else 0
    subtype = _subtype(category, seq, full_correct)
    onset_bin = _normalize_bin(first_frac)
    return {
        "family": family,
        "task_id": task_id,
        "category": category,
        "subtype": subtype,
        "full_trace_correct": int(full_correct),
        "steps": len(seq),
        "sequence": seq,
        "first_correct_frac": float(first_frac),
        "local_correct_rate": float(sum(seq) / len(seq)),
        "last_small_correct": int(seq[-1]),
        "tail_correct_rate": tail_rate,
        "collapse_count": int(collapse_count),
        "longest_correct_streak": int(longest_streak),
        "streak_after_onset": int(streak_after_onset),
        "onset_bin": onset_bin,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cat[str(record["category"])].append(record)
        by_subtype[str(record["subtype"])].append(record)
        by_family[str(record["family"])].append(record)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tasks": len(rows),
            "mean_first_correct_frac": float(np.mean([r["first_correct_frac"] for r in rows]))
            if rows
            else 0.0,
            "mean_local_correct_rate": float(np.mean([r["local_correct_rate"] for r in rows]))
            if rows
            else 0.0,
            "mean_last_small_correct": float(np.mean([r["last_small_correct"] for r in rows]))
            if rows
            else 0.0,
            "mean_tail_correct_rate": float(np.mean([r["tail_correct_rate"] for r in rows]))
            if rows
            else 0.0,
            "mean_collapse_count": float(np.mean([r["collapse_count"] for r in rows]))
            if rows
            else 0.0,
            "mean_longest_correct_streak": float(
                np.mean([r["longest_correct_streak"] for r in rows])
            )
            if rows
            else 0.0,
            "mean_streak_after_onset": float(np.mean([r["streak_after_onset"] for r in rows]))
            if rows
            else 0.0,
            "onset_bin_counts": dict(Counter(str(r["onset_bin"]) for r in rows)),
        }

    late_success = by_cat["late-success"]
    late_failure = by_cat["late-failure"]
    return {
        "total_tasks": len(records),
        "category_counts": dict(Counter(str(r["category"]) for r in records)),
        "subtype_counts": dict(Counter(str(r["subtype"]) for r in records)),
        "by_category": {key: block(value) for key, value in by_cat.items()},
        "by_subtype": {key: block(value) for key, value in by_subtype.items()},
        "by_family": {
            key: {
                "tasks": len(value),
                "category_counts": dict(Counter(str(r["category"]) for r in value)),
                "subtype_counts": dict(Counter(str(r["subtype"]) for r in value)),
            }
            for key, value in by_family.items()
        },
        "headline_gaps": {
            "late_success_minus_late_failure_first_correct_frac": float(
                block(late_success)["mean_first_correct_frac"]
                - block(late_failure)["mean_first_correct_frac"]
            )
            if late_success and late_failure
            else 0.0,
            "late_success_minus_late_failure_tail_correct_rate": float(
                block(late_success)["mean_tail_correct_rate"]
                - block(late_failure)["mean_tail_correct_rate"]
            )
            if late_success and late_failure
            else 0.0,
            "late_success_minus_late_failure_collapse_count": float(
                block(late_success)["mean_collapse_count"]
                - block(late_failure)["mean_collapse_count"]
            )
            if late_success and late_failure
            else 0.0,
        },
        "records": records,
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    by_category = cast(dict[str, dict[str, Any]], summary["by_category"])
    by_subtype = cast(dict[str, dict[str, Any]], summary["by_subtype"])
    by_family = cast(dict[str, dict[str, Any]], summary["by_family"])
    lines = [
        "# Trace Persistence Subtypes",
        "",
        "## Question",
        "",
        "- Split the main taxonomy into more persistence-oriented subtypes.",
        "- Ask whether `late-success` and `late-failure` differ more by onset timing or by post-onset survival.",
        "",
        "## Category Summary",
        "",
        "| Category | Tasks | First-correct frac | Local correct rate | Last-small-correct | Tail correct rate | Collapse count | Longest correct streak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in ["stable-success", "late-success", "late-failure", "persistent-failure"]:
        block = by_category.get(category)
        if not block:
            continue
        lines.append(
            f"| {category} | {block['tasks']} | {block['mean_first_correct_frac']:.3f} | "
            f"{block['mean_local_correct_rate']:.3f} | {block['mean_last_small_correct']:.3f} | "
            f"{block['mean_tail_correct_rate']:.3f} | {block['mean_collapse_count']:.3f} | "
            f"{block['mean_longest_correct_streak']:.3f} |"
        )
    lines += [
        "",
        "## Subtype Summary",
        "",
        "| Subtype | Tasks | First-correct frac | Tail correct rate | Collapse count | Streak after onset |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    subtype_order = [
        "early-lock-in",
        "delayed-lock-in",
        "oscillatory-success",
        "fragile-flash",
        "rescue-then-collapse",
        "terminal-near-miss",
        "partial-runout",
        "never-correct",
    ]
    for subtype in subtype_order:
        block = by_subtype.get(subtype)
        if not block:
            continue
        lines.append(
            f"| {subtype} | {block['tasks']} | {block['mean_first_correct_frac']:.3f} | "
            f"{block['mean_tail_correct_rate']:.3f} | {block['mean_collapse_count']:.3f} | "
            f"{block['mean_streak_after_onset']:.3f} |"
        )
    lines += [
        "",
        "## Family Breakdown",
        "",
    ]
    for family, block in sorted(by_family.items()):
        lines.append(f"### `{family}`")
        lines.append("")
        lines.append(f"- tasks: `{block['tasks']}`")
        lines.append(
            f"- category counts: `{json.dumps(block['category_counts'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- subtype counts: `{json.dumps(block['subtype_counts'], ensure_ascii=False)}`"
        )
        lines.append("")
    lines += [
        "## Main Read",
        "",
        "- `late-success` is later on average, but once correctness appears it is much more likely to survive.",
        "- `late-failure` contains at least two distinct modes: a fragile one-flash mode and a rescue-then-collapse mode.",
        "- This supports a `state formation vs state preservation` decomposition rather than a single-step sufficiency story.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for reentry_rows in sorted(REENTRY_DIR.glob("*_no_outliers/reentry_rows.jsonl")):
        family = reentry_rows.parent.name.replace("_reentry_livebench_local_v1_", "").replace(
            "_no_outliers", ""
        )
        rows = _load_jsonl(reentry_rows)
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)
        for task_id, task_rows in by_task.items():
            records.append(_task_record(family, task_id, task_rows))
    summary = _summarize(records)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
