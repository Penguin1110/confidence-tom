from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "results" / "_prefix_predictor_v1" / "prefix_predictor_rows.csv"
BASELINE_JSON = ROOT / "results" / "_prefix_predictor_v1" / "baseline_results.json"
OUT_JSON = ROOT / "results" / "_prefix_predictor_v1" / "predictor_failure_analysis.json"
OUT_MD = ROOT / "docs" / "prefix_predictor_failure_analysis.md"


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


def _load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cooked: dict[str, object] = dict(row)
            cooked["delta_positive"] = int(row["delta_positive"])
            for feature in FEATURES:
                cooked[feature] = float(row[feature])
            rows.append(cooked)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def _effect_size(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    mu1 = _mean(pos)
    mu0 = _mean(neg)
    pooled = math.sqrt((_variance(pos) + _variance(neg)) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (mu1 - mu0) / pooled


def _group_positive_rates(rows: list[dict[str, object]]) -> dict[str, object]:
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


def _feature_separation(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if int(row["delta_positive"]) == 1]
    negatives = [row for row in rows if int(row["delta_positive"]) == 0]

    overall = []
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

    by_small: dict[str, list[dict[str, object]]] = {}
    for family in sorted({str(row["small_family"]) for row in rows}):
        fam_rows = [row for row in rows if str(row["small_family"]) == family]
        fam_pos = [row for row in fam_rows if int(row["delta_positive"]) == 1]
        fam_neg = [row for row in fam_rows if int(row["delta_positive"]) == 0]
        stats = []
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


def _baseline_summary() -> dict[str, object]:
    data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    cleaned = []
    for exp in data["experiments"]:
        cleaned.append(
            {
                "name": exp["name"],
                "feature_count": exp["feature_count"],
                "test_metrics": exp["test_metrics"],
                "top_coefficients": exp["top_coefficients"][:10],
            }
        )
    return {"experiments": cleaned}


def _to_markdown(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Prefix Predictor Failure Analysis")
    lines.append("")
    lines.append("## 核心問題")
    lines.append("")
    lines.append(
        "這份分析不是再追求更高分，而是回答：為什麼目前的 baseline predictor 只顯示有限訊號？"
    )
    lines.append("")

    rates = report["positive_rates"]
    lines.append("## 1. 類別不平衡與 family heterogeneity")
    lines.append("")
    lines.append(f"- overall positive rate: `{rates['overall_positive_rate']:.3f}`")
    lines.append("")
    lines.append("| Pair | Count | Positive Rate |")
    lines.append("| --- | ---: | ---: |")
    for key, value in rates["by_pair"].items():
        lines.append(f"| `{key}` | {value['count']} | {value['positive_rate']:.3f} |")
    lines.append("")
    lines.append(
        "解讀：如果不同 family pairing 的正例率差很多，pooled baseline 很容易先學到 family prior，而不是更細的 state signal。"
    )
    lines.append("")

    lines.append("## 2. 單一特徵的分離度")
    lines.append("")
    lines.append("| Feature | Positive Mean | Non-Positive Mean | Effect Size |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["feature_separation"]["overall"][:10]:
        lines.append(
            f"| `{row['feature']}` | {row['positive_mean']:.3f} | "
            f"{row['non_positive_mean']:.3f} | {row['effect_size']:.3f} |"
        )
    lines.append("")
    lines.append("解讀：如果 effect size 普遍不大，就表示單一表面特徵本身對正 gain 的分離度有限。")
    lines.append("")

    lines.append("## 3. 依 small family 分開看")
    lines.append("")
    for family, rows in report["feature_separation"]["by_small_family"].items():
        lines.append(f"### `{family}`")
        lines.append("")
        lines.append("| Feature | Effect Size |")
        lines.append("| --- | ---: |")
        for row in rows[:8]:
            lines.append(f"| `{row['feature']}` | {row['effect_size']:.3f} |")
        lines.append("")

    lines.append("## 4. Baseline 結果")
    lines.append("")
    for exp in report["baseline"]["experiments"]:
        m = exp["test_metrics"]
        lines.append(
            f"- `{exp['name']}`: "
            f"AUROC={m['auroc']:.3f}, F1={m['f1']:.3f}, "
            f"Precision={m['precision']:.3f}, Recall={m['recall']:.3f}"
        )
    lines.append("")
    lines.append("## 暫時結論")
    lines.append("")
    lines.append("1. baseline 弱，不是因為完全沒訊號，而是因為正例本來就稀少。")
    lines.append("2. pooled data 有明顯 family heterogeneity，會稀釋單一 signal 的效果。")
    lines.append(
        "3. confidence-related features 有 modest 增量，但目前還不足以單獨撐起 predictor。"
    )
    lines.append(
        "4. 下一步若要再進一步，最合理的是補更接近 reasoning state 的 fragility signal，或改做 family-conditioned predictor。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = _load_rows()
    report = {
        "positive_rates": _group_positive_rates(rows),
        "feature_separation": _feature_separation(rows),
        "baseline": _baseline_summary(),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report to {OUT_JSON}")
    print(f"Wrote Markdown report to {OUT_MD}")


if __name__ == "__main__":
    main()
