from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from typing import Any, Sequence, cast

from confidence_tom.infra.paths import project_root

ROOT = project_root()
DATASET_CSV = ROOT / "outputs" / "results" / "_prefix_predictor_v1" / "prefix_predictor_rows.csv"
OUT_DIR = ROOT / "outputs" / "results" / "_prefix_predictor_v1"
SIGNAL_JSON = OUT_DIR / "signal_pattern_analysis.json"
SIGNAL_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "prefix"
    / "prefix_signal_pattern_analysis.md"
)
BASELINE_JSON = OUT_DIR / "baseline_results.json"
FAILURE_JSON = OUT_DIR / "predictor_failure_analysis.json"
FAILURE_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "prefix"
    / "prefix_predictor_failure_analysis.md"
)

SIGNALS = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
    "semantic_drift_score",
    "hedge_density",
]

FEATURES = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
    "semantic_drift_score",
    "hedge_density",
    "confidence_proxy",
    "prefix_text_tokens",
    "backtracking_flag",
    "backtracking_mentions",
    "self_correction_cue_density",
    "certainty_density",
    "commitment_score",
]

Row = dict[str, Any]
FeatureStat = dict[str, float | str]


def _load_rows() -> list[Row]:
    rows: list[Row] = []
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cooked: Row = dict(row)
            for key in [
                "delta_t",
                "step_index",
                "prefix_segments_count",
                "prefix_tokens",
                "current_segment_tokens",
                "semantic_drift_score",
                "hedge_density",
                "confidence_proxy",
                "prefix_text_tokens",
                "backtracking_flag",
                "backtracking_mentions",
                "self_correction_cue_density",
                "certainty_density",
                "commitment_score",
            ]:
                if key in row:
                    cooked[key] = float(row[key])
            for key in ["delta_positive", "delta_nonzero", "positive_gain"]:
                if key in row:
                    cooked[key] = int(row[key])
            rows.append(cooked)
    return rows


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)


def _effect_size(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    pooled = math.sqrt((_variance(pos) + _variance(neg)) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (_mean(pos) - _mean(neg)) / pooled


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


def _conditional_positive_rate(rows: list[Row], signal: str) -> list[dict[str, Any]]:
    values = [float(row[signal]) for row in rows]
    bins = _quantile_bins(values, bins=5)
    if not bins:
        return []
    grouped: list[list[Row]] = [[] for _ in bins]
    for row in rows:
        value = float(row[signal])
        for i, (left, right) in enumerate(bins):
            final = i == len(bins) - 1
            if (left <= value < right) or (final and left <= value <= right):
                grouped[i].append(row)
                break
    out: list[dict[str, Any]] = []
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


def _task_level_agreement(rows: list[Row]) -> dict[str, Any]:
    task_run_positive_fraction: dict[str, dict[str, float]] = defaultdict(dict)
    task_run_any_positive: dict[str, dict[str, int]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["run_name"]))].append(row)
    for (task_id, run_name), block in grouped.items():
        positives = sum(int(row["delta_positive"]) for row in block)
        total = len(block)
        frac = positives / total if total else 0.0
        task_run_positive_fraction[task_id][run_name] = frac
        task_run_any_positive[task_id][run_name] = int(positives > 0)

    task_stats: list[dict[str, Any]] = []
    for task_id in sorted(task_run_positive_fraction):
        values = list(task_run_positive_fraction[task_id].values())
        binary_values = list(task_run_any_positive[task_id].values())
        mean_frac = _mean(values)
        task_stats.append(
            {
                "task_id": task_id,
                "mean_positive_fraction": mean_frac,
                "variance_positive_fraction": _variance(values),
                "positive_run_count": int(sum(binary_values)),
                "run_count": len(binary_values),
                "agreement_any_positive_rate": _mean(binary_values),
            }
        )
    task_stats.sort(
        key=lambda row: (
            -float(row["agreement_any_positive_rate"]),
            -float(row["mean_positive_fraction"]),
            float(row["variance_positive_fraction"]),
        )
    )
    return {
        "task_count": len(task_stats),
        "high_agreement_positive_tasks": [
            row for row in task_stats if row["positive_run_count"] >= 5
        ][:10],
        "highest_variance_tasks": sorted(
            task_stats,
            key=lambda row: float(row["variance_positive_fraction"]),
            reverse=True,
        )[:10],
        "all_task_stats": task_stats,
    }


def _baseline_summary() -> dict[str, Any]:
    data = cast(dict[str, Any], json.loads(BASELINE_JSON.read_text(encoding="utf-8")))
    return {
        "experiments": [
            {
                "name": exp["name"],
                "feature_count": exp["feature_count"],
                "test_metrics": exp["test_metrics"],
                "top_coefficients": exp["top_coefficients"][:10],
            }
            for exp in data["experiments"]
        ]
    }


def _feature_separation(rows: list[Row]) -> dict[str, Any]:
    positives = [row for row in rows if int(row["delta_positive"]) == 1]
    negatives = [row for row in rows if int(row["delta_positive"]) == 0]
    overall: list[FeatureStat] = []
    for feature in FEATURES:
        pos_vals = [float(row[feature]) for row in positives]
        neg_vals = [float(row[feature]) for row in negatives]
        overall.append(
            {
                "feature": feature,
                "positive_mean": _mean(pos_vals),
                "non_positive_mean": _mean(neg_vals),
                "effect_size": _effect_size(pos_vals, neg_vals),
            }
        )
    overall.sort(key=lambda row: abs(float(row["effect_size"])), reverse=True)
    by_small: dict[str, list[FeatureStat]] = {}
    for family in sorted({str(row["small_family"]) for row in rows}):
        fam_rows = [row for row in rows if str(row["small_family"]) == family]
        fam_pos = [row for row in fam_rows if int(row["delta_positive"]) == 1]
        fam_neg = [row for row in fam_rows if int(row["delta_positive"]) == 0]
        stats: list[FeatureStat] = []
        for feature in FEATURES:
            pos_vals = [float(row[feature]) for row in fam_pos]
            neg_vals = [float(row[feature]) for row in fam_neg]
            stats.append(
                {
                    "feature": feature,
                    "positive_mean": _mean(pos_vals),
                    "non_positive_mean": _mean(neg_vals),
                    "effect_size": _effect_size(pos_vals, neg_vals),
                }
            )
        stats.sort(key=lambda row: abs(float(row["effect_size"])), reverse=True)
        by_small[family] = stats
    return {"overall": overall, "by_small_family": by_small}


def _signal_markdown(report: dict[str, Any]) -> str:
    lines = ["# Prefix Signal Pattern Analysis", "", "## 目的", ""]
    lines.extend(
        [
            "這份分析補兩件事：",
            "",
            "1. `P(Δ_t > 0 | signal)` 的條件統計",
            "2. cross-family 的 task-level agreement / variance",
            "",
            (
                "說明：由於不同 family 的 segmentation 不完全對齊，"
                "cross-family 穩定性分析先用 task-level 聚合，不硬做 prefix-level 對齊。"
            ),
            "",
            "## 條件統計",
            "",
        ]
    )
    conditional_rate = cast(
        dict[str, list[dict[str, Any]]],
        report["conditional_positive_rate"],
    )
    for signal, buckets in conditional_rate.items():
        lines.append(f"### `{signal}`")
        lines.extend(
            [
                "",
                "| Bucket | Count | Positive Rate | Avg `Delta_t` |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in buckets:
            lines.append(
                f"| `{row['bucket']}` | {row['count']} | "
                f"{row['positive_rate']:.3f} | {row['avg_delta_t']:.3f} |"
            )
        lines.append("")
    agreement = cast(dict[str, Any], report["task_level_agreement"])
    lines.extend(["## Cross-Family Task-Level Agreement", "", "### 最穩定的可救題", ""])
    lines.extend(
        [
            "| Task | Positive Runs | Mean Positive Fraction | Variance |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in agreement["high_agreement_positive_tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/{row['run_count']} | "
            f"{row['mean_positive_fraction']:.3f} | {row['variance_positive_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### 最不穩定的題",
            "",
            "| Task | Positive Runs | Mean Positive Fraction | Variance |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in agreement["highest_variance_tasks"]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/{row['run_count']} | "
            f"{row['mean_positive_fraction']:.3f} | {row['variance_positive_fraction']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _failure_markdown(report: dict[str, Any]) -> str:
    lines = ["# Prefix Predictor Failure Analysis", "", "## 核心問題", ""]
    lines.extend(
        [
            "這份分析不是再追求更高分，而是回答：為什麼目前的 baseline predictor 只顯示有限訊號？",
            "",
            "## 1. 類別不平衡與 family heterogeneity",
            "",
            f"- overall positive rate: `{report['positive_rates']['overall_positive_rate']:.3f}`",
            "",
            "| Pair | Count | Positive Rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, value in cast(dict[str, dict[str, Any]], report["positive_rates"]["by_pair"]).items():
        lines.append(f"| `{key}` | {value['count']} | {value['positive_rate']:.3f} |")
    lines.extend(
        [
            "",
            "## 2. 單一特徵的分離度",
            "",
            "| Feature | Positive Mean | Non-Positive Mean | Effect Size |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in cast(list[FeatureStat], report["feature_separation"]["overall"])[:10]:
        lines.append(
            f"| `{row['feature']}` | {row['positive_mean']:.3f} | "
            f"{row['non_positive_mean']:.3f} | {row['effect_size']:.3f} |"
        )
    lines.extend(["", "## 3. 依 small family 分開看", ""])
    by_small_family = cast(
        dict[str, list[FeatureStat]],
        report["feature_separation"]["by_small_family"],
    )
    for family, rows in by_small_family.items():
        lines.append(f"### `{family}`")
        lines.extend(["", "| Feature | Effect Size |", "| --- | ---: |"])
        for row in rows[:8]:
            lines.append(f"| `{row['feature']}` | {row['effect_size']:.3f} |")
        lines.append("")
    lines.extend(["## 4. Baseline 結果", ""])
    for exp in cast(list[dict[str, Any]], report["baseline"]["experiments"]):
        m = exp["test_metrics"]
        lines.append(
            f"- `{exp['name']}`: AUROC={m['auroc']:.3f}, "
            f"F1={m['f1']:.3f}, Precision={m['precision']:.3f}, "
            f"Recall={m['recall']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def _run_signal_patterns(rows: list[Row]) -> None:
    report: dict[str, Any] = {
        "rows": len(rows),
        "signals": SIGNALS,
        "conditional_positive_rate": {
            signal: _conditional_positive_rate(rows, signal) for signal in SIGNALS
        },
        "task_level_agreement": _task_level_agreement(rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_MD.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    SIGNAL_MD.write_text(_signal_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report to {SIGNAL_JSON}")
    print(f"Wrote Markdown report to {SIGNAL_MD}")


def _run_predictor_failure(rows: list[Row]) -> None:
    report: dict[str, Any] = {
        "positive_rates": _group_positive_rates(rows),
        "feature_separation": _feature_separation(rows),
        "baseline": _baseline_summary(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_MD.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    FAILURE_MD.write_text(_failure_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report to {FAILURE_JSON}")
    print(f"Wrote Markdown report to {FAILURE_MD}")


def _group_positive_rates(rows: list[Row]) -> dict[str, Any]:
    by_pair: dict[str, list[int]] = defaultdict(list)
    by_small: dict[str, list[int]] = defaultdict(list)
    by_large: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_pair[f"{row['small_family']}->{row['large_family']}"].append(int(row["delta_positive"]))
        by_small[str(row["small_family"])].append(int(row["delta_positive"]))
        by_large[str(row["large_family"])].append(int(row["delta_positive"]))
    return {
        "overall_positive_rate": _mean([int(row["delta_positive"]) for row in rows]),
        "by_pair": {
            key: {"count": len(values), "positive_rate": _mean(values)}
            for key, values in sorted(by_pair.items())
        },
        "by_small_family": {
            key: {"count": len(values), "positive_rate": _mean(values)}
            for key, values in sorted(by_small.items())
        },
        "by_large_family": {
            key: {"count": len(values), "positive_rate": _mean(values)}
            for key, values in sorted(by_large.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified diagnostics for prefix predictor analyses."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("signal-patterns")
    subparsers.add_parser("predictor-failure")
    args = parser.parse_args()
    rows = _load_rows()
    if args.command == "signal-patterns":
        _run_signal_patterns(rows)
    else:
        _run_predictor_failure(rows)


if __name__ == "__main__":
    main()
