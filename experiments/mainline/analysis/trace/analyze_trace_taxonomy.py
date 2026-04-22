from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "_trace_taxonomy_v1"
OUT_JSON = OUT_DIR / "trace_taxonomy_summary.json"
OUT_MD = ROOT / "docs" / "trace_taxonomy_analysis.md"

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


@dataclass(frozen=True)
class TaskRecord:
    run_name: str
    benchmark: str
    family: str
    task_id: str
    full_correct: bool
    step_count: int
    any_small_correct: bool
    first_correct_step: int | None
    first_correct_frac: float
    local_correct_rate: float
    last_small_correct: bool
    category: str


def _is_final_result_file(path: Path) -> bool:
    if "/partials/" in str(path):
        return False
    if path.name.endswith(".bak") or ".bak_" in path.name:
        return False
    return True


def _load_final_json_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        if not _is_final_result_file(path):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and "prefix_oracle_steps" in data[0]
        ):
            files.append(path)
    return files


def _mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _load_records() -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for path in _load_final_json_files():
        data = json.loads(path.read_text())
        for task in data:
            steps = task.get("prefix_oracle_steps", [])
            small_corrects = [bool(step.get("small_continue_correct")) for step in steps]
            step_count = len(small_corrects)
            any_small_correct = any(small_corrects)
            first_correct_step = next(
                (idx + 1 for idx, flag in enumerate(small_corrects) if flag), None
            )
            first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
            local_correct_rate = (sum(small_corrects) / step_count) if step_count else 0.0
            last_small_correct = bool(small_corrects[-1]) if small_corrects else False
            full_correct = bool(task.get("full_trace_correct"))

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
                category = "fragile-success" if any_small_correct else "persistent-failure"

            records.append(
                TaskRecord(
                    run_name=str(task.get("small_model", path.stem)),
                    benchmark=str(task.get("benchmark", "")),
                    family=str(task.get("small_model", "")).split("/")[0]
                    if task.get("small_model")
                    else "",
                    task_id=str(task.get("task_id", "")),
                    full_correct=full_correct,
                    step_count=step_count,
                    any_small_correct=any_small_correct,
                    first_correct_step=first_correct_step,
                    first_correct_frac=first_correct_frac,
                    local_correct_rate=local_correct_rate,
                    last_small_correct=last_small_correct,
                    category=category,
                )
            )
    return records


def _group_stats(records: list[TaskRecord]) -> dict[str, object]:
    by_cat: dict[str, list[TaskRecord]] = defaultdict(list)
    by_bench: dict[str, list[TaskRecord]] = defaultdict(list)
    by_family: dict[str, list[TaskRecord]] = defaultdict(list)
    for r in records:
        by_cat[r.category].append(r)
        by_bench[r.benchmark].append(r)
        by_family[r.family].append(r)

    def _stats(block: list[TaskRecord]) -> dict[str, float | int]:
        return {
            "n": len(block),
            "full_rate": _mean_or_zero([1.0 if r.full_correct else 0.0 for r in block]),
            "any_small_rate": _mean_or_zero([1.0 if r.any_small_correct else 0.0 for r in block]),
            "last_small_rate": _mean_or_zero([1.0 if r.last_small_correct else 0.0 for r in block]),
            "mean_steps": _mean_or_zero([float(r.step_count) for r in block]),
            "mean_first_correct_frac": _mean_or_zero([float(r.first_correct_frac) for r in block]),
            "median_first_correct_frac": float(median([float(r.first_correct_frac) for r in block]))
            if block
            else 0.0,
            "mean_local_correct_rate": _mean_or_zero([float(r.local_correct_rate) for r in block]),
        }

    direct = {
        True: [r for r in records if r.full_correct],
        False: [r for r in records if not r.full_correct],
    }
    direct_summary = {
        str(k): {
            "n": len(v),
            "any_small_rate": _mean_or_zero([1.0 if r.any_small_correct else 0.0 for r in v]),
            "last_small_rate": _mean_or_zero([1.0 if r.last_small_correct else 0.0 for r in v]),
            "mean_first_correct_frac": _mean_or_zero([float(r.first_correct_frac) for r in v]),
            "mean_local_correct_rate": _mean_or_zero([float(r.local_correct_rate) for r in v]),
        }
        for k, v in direct.items()
    }

    transition = Counter((r.full_correct, r.any_small_correct) for r in records)

    return {
        "counts": {k: int(v) for k, v in Counter(r.category for r in records).items()},
        "by_category": {cat: _stats(block) for cat, block in by_cat.items()},
        "by_benchmark": {bench: _stats(block) for bench, block in by_bench.items()},
        "by_family": {family: _stats(block) for family, block in by_family.items()},
        "direct_vs_reentry": direct_summary,
        "transition_matrix": {
            "full_and_any_small_correct": int(transition[(True, True)]),
            "full_and_no_small_correct": int(transition[(True, False)]),
            "wrong_and_any_small_correct": int(transition[(False, True)]),
            "wrong_and_no_small_correct": int(transition[(False, False)]),
        },
        "taxonomy_definition": {
            "stable_success": (
                "full_correct and first_correct_frac <= "
                f"{EARLY_FRACTION:.3f} and local_correct_rate >= "
                f"{MIN_STABLE_LOCAL_RATE:.3f}"
            ),
            "late_success": "full_correct and not stable_success",
            "fragile_success": "not full_correct and any_small_correct",
            "persistent_failure": "not full_correct and not any_small_correct",
        },
        "total_records": len(records),
    }


def _render_markdown(summary: dict[str, object], records: list[TaskRecord]) -> str:
    by_category = cast(dict[str, dict[str, float | int]], summary["by_category"])
    direct_vs_reentry = cast(dict[str, dict[str, float | int]], summary["direct_vs_reentry"])
    transition_matrix = cast(dict[str, int], summary["transition_matrix"])
    by_benchmark = cast(dict[str, dict[str, float | int]], summary["by_benchmark"])
    by_family = cast(dict[str, dict[str, float | int]], summary["by_family"])
    lines: list[str] = []
    lines.append("# Trace Taxonomy Analysis")
    lines.append("")
    lines.append("## Definition")
    lines.append("")
    lines.append(
        (
            "- Stable-success: full correct, early local re-entry correctness, "
            "and local correctness stays reasonably high."
        )
    )
    lines.append("- Late-success: full correct but not stable-success.")
    lines.append("- Fragile-success: some local re-entry correctness, but final full trace wrong.")
    lines.append("- Persistent-failure: no local re-entry correctness and final full trace wrong.")
    lines.append("")
    lines.append("## Overall Counts")
    lines.append("")
    lines.append("| Category | Count | Share | Mean first-correct frac | Mean local correct rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    total = len(records)
    for cat in ["stable-success", "late-success", "fragile-success", "persistent-failure"]:
        block = [r for r in records if r.category == cat]
        if not block:
            continue
        stats = by_category[cat]
        lines.append(
            (
                f"| {cat} | {stats['n']} | {stats['n'] / total:.3f} | "
                f"{stats['mean_first_correct_frac']:.3f} | "
                f"{stats['mean_local_correct_rate']:.3f} |"
            )
        )
    lines.append("")
    lines.append("## Direct vs Re-entry")
    lines.append("")
    lines.append(
        (
            "| Direct full correctness | N | Any small correct | Last small "
            "correct | Mean first-correct frac | Mean local correct rate |"
        )
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k in ["True", "False"]:
        direct_stats = direct_vs_reentry[k]
        lines.append(
            (
                f"| {k} | {direct_stats['n']} | {direct_stats['any_small_rate']:.3f} | "
                f"{direct_stats['last_small_rate']:.3f} | "
                f"{direct_stats['mean_first_correct_frac']:.3f} | "
                f"{direct_stats['mean_local_correct_rate']:.3f} |"
            )
        )
    lines.append("")
    lines.append("## Transition Matrix")
    lines.append("")
    lines.append(
        f"- full correct & any small correct: `{transition_matrix['full_and_any_small_correct']}`"
    )
    lines.append(
        f"- full correct & no small correct: `{transition_matrix['full_and_no_small_correct']}`"
    )
    lines.append(
        f"- full wrong & any small correct: `{transition_matrix['wrong_and_any_small_correct']}`"
    )
    lines.append(
        f"- full wrong & no small correct: `{transition_matrix['wrong_and_no_small_correct']}`"
    )
    lines.append("")
    lines.append("## Benchmark Breakdown")
    lines.append("")
    lines.append("| Benchmark | Stable | Late | Fragile | Persistent | Mean first-correct frac |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for bench, stats in sorted(by_benchmark.items()):
        block = [r for r in records if r.benchmark == bench]
        counts = Counter(r.category for r in block)
        lines.append(
            (
                f"| {bench} | {counts['stable-success']} | "
                f"{counts['late-success']} | {counts['fragile-success']} | "
                f"{counts['persistent-failure']} | "
                f"{stats['mean_first_correct_frac']:.3f} |"
            )
        )
    lines.append("")
    lines.append("## Family Breakdown")
    lines.append("")
    lines.append("| Family | Stable | Late | Fragile | Persistent | Mean first-correct frac |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for family, stats in sorted(by_family.items()):
        block = [r for r in records if r.family == family]
        counts = Counter(r.category for r in block)
        lines.append(
            (
                f"| {family} | {counts['stable-success']} | "
                f"{counts['late-success']} | {counts['fragile-success']} | "
                f"{counts['persistent-failure']} | "
                f"{stats['mean_first_correct_frac']:.3f} |"
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The taxonomy uses an early cutoff of 1/3 of the trace for stable-success.")
    lines.append(
        (
            "- Tasks with full correctness but no local small-model correctness "
            "are absorbed into late-success under this definition."
        )
    )
    lines.append(
        (
            "- The key hypothesis is competence-conditioned rescue: re-entry "
            "helps most when the trace already has partial signal."
        )
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    records = _load_records()
    if not records:
        raise SystemExit("No trace results found.")

    summary = _group_stats(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"summary": summary, "records": [r.__dict__ for r in records]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    OUT_MD.write_text(_render_markdown(summary, records), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
