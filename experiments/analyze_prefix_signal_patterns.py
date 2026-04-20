from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "results" / "_prefix_predictor_v1" / "prefix_predictor_rows.csv"
OUT_DIR = ROOT / "results" / "_prefix_predictor_v1"
OUT_JSON = OUT_DIR / "signal_pattern_analysis.json"
OUT_MD = ROOT / "docs" / "prefix_signal_pattern_analysis.md"


SIGNALS = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
    "semantic_drift_score",
    "hedge_density",
]


def _load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cooked: dict[str, object] = dict(row)
            for key in [
                "delta_t",
                "step_index",
                "prefix_segments_count",
                "prefix_tokens",
                "current_segment_tokens",
                "semantic_drift_score",
                "hedge_density",
                "confidence_proxy",
            ]:
                cooked[key] = float(row[key])
            for key in ["delta_positive", "delta_nonzero", "positive_gain"]:
                cooked[key] = int(row[key])
            rows.append(cooked)
    return rows


def _quantile_bins(values: list[float], bins: int = 5) -> list[tuple[float, float]]:
    if not values:
        return []
    xs = sorted(values)
    edges = [xs[0]]
    n = len(xs)
    for i in range(1, bins):
        idx = min(n - 1, math.floor(i * n / bins))
        edges.append(xs[idx])
    edges.append(xs[-1])

    merged: list[tuple[float, float]] = []
    left = edges[0]
    for right in edges[1:]:
        if right <= left:
            continue
        merged.append((left, right))
        left = right
    if not merged:
        merged.append((xs[0], xs[-1]))
    return merged


def _bucket_label(left: float, right: float, final: bool) -> str:
    return f"[{left:.3f}, {right:.3f}{']' if final else ')'}"


def _conditional_positive_rate(
    rows: list[dict[str, object]], signal: str
) -> list[dict[str, object]]:
    values = [float(row[signal]) for row in rows]
    bins = _quantile_bins(values, bins=5)
    if not bins:
        return []

    grouped: list[list[dict[str, object]]] = [[] for _ in bins]
    for row in rows:
        value = float(row[signal])
        for i, (left, right) in enumerate(bins):
            final = i == len(bins) - 1
            if (left <= value < right) or (final and left <= value <= right):
                grouped[i].append(row)
                break

    out: list[dict[str, object]] = []
    for i, ((left, right), bucket_rows) in enumerate(zip(bins, grouped, strict=True)):
        if not bucket_rows:
            continue
        positive = sum(int(row["delta_positive"]) for row in bucket_rows)
        total = len(bucket_rows)
        avg_delta = sum(float(row["delta_t"]) for row in bucket_rows) / total
        out.append(
            {
                "bucket": _bucket_label(left, right, i == len(bins) - 1),
                "count": total,
                "positive_rate": positive / total,
                "avg_delta_t": avg_delta,
            }
        )
    return out


def _task_level_agreement(rows: list[dict[str, object]]) -> dict[str, object]:
    task_run_positive_fraction: dict[str, dict[str, float]] = defaultdict(dict)
    task_run_any_positive: dict[str, dict[str, int]] = defaultdict(dict)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["run_name"]))].append(row)

    for (task_id, run_name), block in grouped.items():
        positives = sum(int(row["delta_positive"]) for row in block)
        total = len(block)
        frac = positives / total if total else 0.0
        task_run_positive_fraction[task_id][run_name] = frac
        task_run_any_positive[task_id][run_name] = int(positives > 0)

    task_stats: list[dict[str, object]] = []
    all_binary_values: list[float] = []
    for task_id in sorted(task_run_positive_fraction):
        run_map = task_run_positive_fraction[task_id]
        values = list(run_map.values())
        mean_frac = sum(values) / len(values)
        variance = sum((v - mean_frac) ** 2 for v in values) / len(values)
        binary_values = list(task_run_any_positive[task_id].values())
        all_binary_values.extend(binary_values)
        agreement = sum(binary_values) / len(binary_values)
        task_stats.append(
            {
                "task_id": task_id,
                "mean_positive_fraction": mean_frac,
                "variance_positive_fraction": variance,
                "positive_run_count": int(sum(binary_values)),
                "run_count": len(binary_values),
                "agreement_any_positive_rate": agreement,
            }
        )

    task_stats.sort(
        key=lambda row: (
            -float(row["agreement_any_positive_rate"]),
            -float(row["mean_positive_fraction"]),
            float(row["variance_positive_fraction"]),
        )
    )

    high_agreement_positive = [row for row in task_stats if row["positive_run_count"] >= 5]
    unstable_tasks = sorted(
        task_stats,
        key=lambda row: float(row["variance_positive_fraction"]),
        reverse=True,
    )[:10]

    return {
        "task_count": len(task_stats),
        "high_agreement_positive_tasks": high_agreement_positive[:10],
        "highest_variance_tasks": unstable_tasks,
        "all_task_stats": task_stats,
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Prefix Signal Pattern Analysis")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append("這份分析補兩件事：")
    lines.append("")
    lines.append("1. `P(Δ_t > 0 | signal)` 的條件統計")
    lines.append("2. cross-family 的 task-level agreement / variance")
    lines.append("")
    lines.append(
        "說明：由於不同 family 的 segmentation 不完全對齊，cross-family 穩定性分析先用 task-level 聚合，不硬做 prefix-level 對齊。"
    )
    lines.append("")
    lines.append("## 條件統計")
    lines.append("")
    for signal, buckets in report["conditional_positive_rate"].items():
        lines.append(f"### `{signal}`")
        lines.append("")
        lines.append("| Bucket | Count | Positive Rate | Avg `Delta_t` |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in buckets:
            lines.append(
                f"| `{row['bucket']}` | {row['count']} | {row['positive_rate']:.3f} | {row['avg_delta_t']:.3f} |"
            )
        lines.append("")

    agreement = report["task_level_agreement"]
    lines.append("## Cross-Family Task-Level Agreement")
    lines.append("")
    lines.append("### 最穩定的可救題")
    lines.append("")
    lines.append("| Task | Positive Runs | Mean Positive Fraction | Variance |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in agreement["high_agreement_positive_tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/{row['run_count']} | "
            f"{row['mean_positive_fraction']:.3f} | {row['variance_positive_fraction']:.4f} |"
        )
    lines.append("")
    lines.append("### 最不穩定的題")
    lines.append("")
    lines.append("| Task | Positive Runs | Mean Positive Fraction | Variance |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in agreement["highest_variance_tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/{row['run_count']} | "
            f"{row['mean_positive_fraction']:.3f} | {row['variance_positive_fraction']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = _load_rows()
    conditional = {signal: _conditional_positive_rate(rows, signal) for signal in SIGNALS}
    agreement = _task_level_agreement(rows)

    report = {
        "rows": len(rows),
        "signals": SIGNALS,
        "conditional_positive_rate": conditional,
        "task_level_agreement": agreement,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report to {OUT_JSON}")
    print(f"Wrote Markdown report to {OUT_MD}")


if __name__ == "__main__":
    main()
