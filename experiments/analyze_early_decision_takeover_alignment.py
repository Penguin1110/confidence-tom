from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EARLY_DIR = ROOT / "results" / "_early_decision_v1"
PREDICTOR_DIR = ROOT / "results" / "_prefix_predictor_v1"

EARLY_MSP_JSON = EARLY_DIR / "minimal_sufficient_prefix_analysis.json"
BOTTLENECK_JSON = EARLY_DIR / "early_decision_bottlenecks.json"
PREDICTOR_CSV = PREDICTOR_DIR / "prefix_predictor_rows.csv"

OUTPUT_JSON = EARLY_DIR / "early_decision_takeover_alignment.json"
OUTPUT_MD = ROOT / "docs" / "early_decision_takeover_alignment.md"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def main() -> None:
    early = _load_json(EARLY_MSP_JSON)
    bottleneck = _load_json(BOTTLENECK_JSON)

    msp_by_key = {}
    for row in early["task_groups"]:
        msp_by_key[(row["run_name"], row["task_id"])] = row

    bottleneck_by_key = {}
    for row in bottleneck["task_groups"]:
        bottleneck_by_key[(row["run_name"], row["task_id"])] = row

    task_prefix_stats: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "positive_steps": 0,
            "negative_steps": 0,
            "total_steps": 0,
            "earliest_positive_step": None,
            "earliest_negative_step": None,
            "any_positive": 0,
            "any_negative": 0,
            "benchmark": None,
            "small_family": None,
            "large_family": None,
        }
    )

    with PREDICTOR_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["run_name"], row["task_id"])
            stats = task_prefix_stats[key]
            step = int(float(row["step_index"]))
            delta_pos = int(row["delta_positive"])
            delta_t = float(row["delta_t"])
            stats["benchmark"] = row["benchmark"]
            stats["small_family"] = row["small_family"]
            stats["large_family"] = row["large_family"]
            stats["total_steps"] += 1
            if delta_pos == 1:
                stats["positive_steps"] += 1
                stats["any_positive"] = 1
                if (
                    stats["earliest_positive_step"] is None
                    or step < stats["earliest_positive_step"]
                ):
                    stats["earliest_positive_step"] = step
            if delta_t < 0:
                stats["negative_steps"] += 1
                stats["any_negative"] = 1
                if (
                    stats["earliest_negative_step"] is None
                    or step < stats["earliest_negative_step"]
                ):
                    stats["earliest_negative_step"] = step

    aligned_rows = []
    for key, msp_row in msp_by_key.items():
        if key not in task_prefix_stats:
            continue
        prefix_stats = task_prefix_stats[key]
        bottleneck_row = bottleneck_by_key.get(key)
        earliest_pos = prefix_stats["earliest_positive_step"]
        msp = msp_row.get("minimal_sufficient_step")
        first_cross_60 = bottleneck_row.get("first_cross_60") if bottleneck_row else None
        first_cross_70 = bottleneck_row.get("first_cross_70") if bottleneck_row else None
        aligned_rows.append(
            {
                "run_name": key[0],
                "task_id": key[1],
                "benchmark": prefix_stats["benchmark"],
                "small_family": prefix_stats["small_family"],
                "large_family": prefix_stats["large_family"],
                "minimal_sufficient_step": msp,
                "first_cross_60": first_cross_60,
                "first_cross_70": first_cross_70,
                "earliest_positive_step": earliest_pos,
                "earliest_negative_step": prefix_stats["earliest_negative_step"],
                "any_positive": prefix_stats["any_positive"],
                "any_negative": prefix_stats["any_negative"],
                "positive_steps": prefix_stats["positive_steps"],
                "negative_steps": prefix_stats["negative_steps"],
                "total_steps": prefix_stats["total_steps"],
                "msp_before_or_equal_positive": int(
                    msp is not None and earliest_pos is not None and msp <= earliest_pos
                ),
                "cross60_before_or_equal_positive": int(
                    first_cross_60 is not None
                    and earliest_pos is not None
                    and first_cross_60 <= earliest_pos
                ),
                "cross70_before_or_equal_positive": int(
                    first_cross_70 is not None
                    and earliest_pos is not None
                    and first_cross_70 <= earliest_pos
                ),
            }
        )

    def subset(pred):
        return [row for row in aligned_rows if pred(row)]

    with_positive = subset(lambda r: r["any_positive"] == 1)
    with_msp_and_positive = subset(
        lambda r: r["minimal_sufficient_step"] is not None and r["any_positive"] == 1
    )
    with_cross60_and_positive = subset(
        lambda r: r["first_cross_60"] is not None and r["any_positive"] == 1
    )
    with_cross70_and_positive = subset(
        lambda r: r["first_cross_70"] is not None and r["any_positive"] == 1
    )

    summary = {
        "rows": len(aligned_rows),
        "with_any_positive": len(with_positive),
        "overall": {
            "msp_before_or_equal_positive_rate": (
                _safe_mean([row["msp_before_or_equal_positive"] for row in with_msp_and_positive])
            ),
            "cross60_before_or_equal_positive_rate": (
                _safe_mean(
                    [row["cross60_before_or_equal_positive"] for row in with_cross60_and_positive]
                )
            ),
            "cross70_before_or_equal_positive_rate": (
                _safe_mean(
                    [row["cross70_before_or_equal_positive"] for row in with_cross70_and_positive]
                )
            ),
            "mean_earliest_positive_step": _safe_mean(
                [
                    row["earliest_positive_step"]
                    for row in with_positive
                    if row["earliest_positive_step"] is not None
                ]
            ),
            "mean_msp_given_positive": _safe_mean(
                [row["minimal_sufficient_step"] for row in with_msp_and_positive]
            ),
            "mean_cross60_given_positive": _safe_mean(
                [row["first_cross_60"] for row in with_cross60_and_positive]
            ),
            "mean_cross70_given_positive": _safe_mean(
                [row["first_cross_70"] for row in with_cross70_and_positive]
            ),
        },
        "by_benchmark": {},
        "by_small_family": {},
        "representative_aligned_tasks": [],
        "representative_misaligned_tasks": [],
        "rows_detail": aligned_rows,
        "notes": [
            "This joins Early Decision / MSP outputs with oracle-gain task-level prefix statistics.",
            "The key question is whether early diagnosability appears before or by the time positive takeover opportunities begin.",
        ],
    }

    for benchmark in sorted({row["benchmark"] for row in aligned_rows}):
        sub = [
            row
            for row in aligned_rows
            if row["benchmark"] == benchmark and row["any_positive"] == 1
        ]
        sub_msp = [row for row in sub if row["minimal_sufficient_step"] is not None]
        sub_c60 = [row for row in sub if row["first_cross_60"] is not None]
        sub_c70 = [row for row in sub if row["first_cross_70"] is not None]
        summary["by_benchmark"][benchmark] = {
            "tasks_with_positive": len(sub),
            "msp_before_or_equal_positive_rate": _safe_mean(
                [row["msp_before_or_equal_positive"] for row in sub_msp]
            ),
            "cross60_before_or_equal_positive_rate": _safe_mean(
                [row["cross60_before_or_equal_positive"] for row in sub_c60]
            ),
            "cross70_before_or_equal_positive_rate": _safe_mean(
                [row["cross70_before_or_equal_positive"] for row in sub_c70]
            ),
            "mean_earliest_positive_step": _safe_mean(
                [row["earliest_positive_step"] for row in sub]
            ),
            "mean_msp_given_positive": _safe_mean(
                [row["minimal_sufficient_step"] for row in sub_msp]
            ),
        }

    for family in sorted({row["small_family"] for row in aligned_rows}):
        sub = [
            row
            for row in aligned_rows
            if row["small_family"] == family and row["any_positive"] == 1
        ]
        sub_msp = [row for row in sub if row["minimal_sufficient_step"] is not None]
        sub_c60 = [row for row in sub if row["first_cross_60"] is not None]
        sub_c70 = [row for row in sub if row["first_cross_70"] is not None]
        summary["by_small_family"][family] = {
            "tasks_with_positive": len(sub),
            "msp_before_or_equal_positive_rate": _safe_mean(
                [row["msp_before_or_equal_positive"] for row in sub_msp]
            ),
            "cross60_before_or_equal_positive_rate": _safe_mean(
                [row["cross60_before_or_equal_positive"] for row in sub_c60]
            ),
            "cross70_before_or_equal_positive_rate": _safe_mean(
                [row["cross70_before_or_equal_positive"] for row in sub_c70]
            ),
            "mean_earliest_positive_step": _safe_mean(
                [row["earliest_positive_step"] for row in sub]
            ),
            "mean_msp_given_positive": _safe_mean(
                [row["minimal_sufficient_step"] for row in sub_msp]
            ),
        }

    aligned_candidates = [
        row
        for row in aligned_rows
        if row["any_positive"] == 1
        and row["minimal_sufficient_step"] is not None
        and row["earliest_positive_step"] is not None
    ]
    summary["representative_aligned_tasks"] = sorted(
        [row for row in aligned_candidates if row["msp_before_or_equal_positive"] == 1],
        key=lambda r: (
            int(r["minimal_sufficient_step"]),
            int(r["earliest_positive_step"]),
            r["benchmark"],
            r["small_family"],
            r["task_id"],
        ),
    )[:10]
    summary["representative_misaligned_tasks"] = sorted(
        [row for row in aligned_candidates if row["msp_before_or_equal_positive"] == 0],
        key=lambda r: (
            int(r["earliest_positive_step"]) - int(r["minimal_sufficient_step"]),
            r["benchmark"],
            r["small_family"],
            r["task_id"],
        ),
        reverse=True,
    )[:10]

    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Early Decision / Takeover Alignment",
        "",
        "這份分析把 Early Decision / MSP 結果接回 oracle gain，檢查 early diagnosability 是否通常早於或同步於 positive takeover opportunity。",
        "",
        "## 整體",
        f"- tasks with any positive takeover: `{summary['with_any_positive']}` / `{summary['rows']}`",
        f"- mean earliest positive step: `{summary['overall']['mean_earliest_positive_step']}`",
        f"- mean MSP given positive: `{summary['overall']['mean_msp_given_positive']}`",
        f"- MSP <= earliest positive rate: `{summary['overall']['msp_before_or_equal_positive_rate']}`",
        f"- cross60 <= earliest positive rate: `{summary['overall']['cross60_before_or_equal_positive_rate']}`",
        f"- cross70 <= earliest positive rate: `{summary['overall']['cross70_before_or_equal_positive_rate']}`",
        "",
        "## By Benchmark",
    ]
    for benchmark, stats in summary["by_benchmark"].items():
        lines.append(
            f"- `{benchmark}`: positive_tasks=`{stats['tasks_with_positive']}`, mean_pos_step=`{stats['mean_earliest_positive_step']}`, mean_msp=`{stats['mean_msp_given_positive']}`, MSP<=pos=`{stats['msp_before_or_equal_positive_rate']}`, cross60<=pos=`{stats['cross60_before_or_equal_positive_rate']}`, cross70<=pos=`{stats['cross70_before_or_equal_positive_rate']}`"
        )
    lines += ["", "## By Small Family"]
    for family, stats in summary["by_small_family"].items():
        lines.append(
            f"- `{family}`: positive_tasks=`{stats['tasks_with_positive']}`, mean_pos_step=`{stats['mean_earliest_positive_step']}`, mean_msp=`{stats['mean_msp_given_positive']}`, MSP<=pos=`{stats['msp_before_or_equal_positive_rate']}`, cross60<=pos=`{stats['cross60_before_or_equal_positive_rate']}`, cross70<=pos=`{stats['cross70_before_or_equal_positive_rate']}`"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis JSON to {OUTPUT_JSON}")
    print(f"Wrote analysis markdown to {OUTPUT_MD}")
    print(
        json.dumps(
            {
                "overall": summary["overall"],
                "by_benchmark": summary["by_benchmark"],
                "by_small_family": summary["by_small_family"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
