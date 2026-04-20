from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "results" / "_early_decision_v1"
DATASET_CSV = DATASET_DIR / "early_decision_rows.csv"
OUTPUT_JSON = DATASET_DIR / "minimal_sufficient_prefix_analysis.json"
OUTPUT_MD = ROOT / "docs" / "minimal_sufficient_prefix_analysis.md"

STATE_FEATURES = [
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


def _stable_test_split(task_id: str, test_ratio: float = 0.2) -> bool:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, *, lr: float = 0.05, steps: int = 4000, l2: float = 1e-3
) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        probs = _sigmoid(x @ weights)
        grad = (x.T @ (probs - y)) / len(y)
        grad += l2 * weights
        weights -= lr * grad
    return weights


def _binary_report_at_threshold(
    y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, len(y_true))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate([np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5]), probs]))
    best_threshold = 0.5
    best_report = _binary_report_at_threshold(y_true, probs, best_threshold)
    best_score = best_report["f1"]
    for threshold in candidates:
        report = _binary_report_at_threshold(y_true, probs, float(threshold))
        score = report["f1"]
        if score > best_score or (
            math.isclose(score, best_score) and report["recall"] > best_report["recall"]
        ):
            best_threshold = float(threshold)
            best_report = report
            best_score = score
    return best_threshold, best_report


def _load_rows() -> list[dict[str, str]]:
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_matrix(rows: list[dict[str, str]]):
    train_rows, test_rows = [], []
    for row in rows:
        (test_rows if _stable_test_split(row["task_id"]) else train_rows).append(row)

    small_families = sorted({row["small_family"] for row in rows})
    benchmarks = sorted({row["benchmark"] for row in rows})
    feature_names = list(STATE_FEATURES)
    feature_names += [f"small_family={name}" for name in small_families]
    feature_names += [f"benchmark={name}" for name in benchmarks]

    def encode(selected_rows: list[dict[str, str]]):
        x = np.zeros((len(selected_rows), len(feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0
            offset = 1
            for feature in STATE_FEATURES:
                x[i, offset] = float(row[feature])
                offset += 1
            for name in small_families:
                x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                offset += 1
            for name in benchmarks:
                x[i, offset] = 1.0 if row["benchmark"] == name else 0.0
                offset += 1
            y[i] = float(row["small_full_trace_success"])
        return x, y

    x_train, y_train = encode(train_rows)
    x_test, y_test = encode(test_rows)
    means = np.mean(x_train[:, 1:], axis=0)
    stds = np.std(x_train[:, 1:], axis=0)
    stds[stds == 0.0] = 1.0
    x_train[:, 1:] = (x_train[:, 1:] - means) / stds
    x_test[:, 1:] = (x_test[:, 1:] - means) / stds
    return train_rows, test_rows, x_train, y_train, x_test, y_test, ["bias", *feature_names]


def _minimal_sufficient_step(sorted_rows: list[dict[str, object]], label: int) -> int | None:
    preds = [int(row["pred"]) for row in sorted_rows]
    steps = [int(float(row["step_index"])) for row in sorted_rows]
    for i, pred in enumerate(preds):
        if pred != label:
            continue
        if all(p == label for p in preds[i:]):
            return steps[i]
    return None


def _first_correct_step(sorted_rows: list[dict[str, object]], label: int) -> int | None:
    for row in sorted_rows:
        if int(row["pred"]) == label:
            return int(float(row["step_index"]))
    return None


def _summarize(groups: list[dict[str, object]]) -> dict[str, object]:
    total = len(groups)
    with_msp = [g for g in groups if g["minimal_sufficient_step"] is not None]
    counter = Counter(str(g["minimal_sufficient_step"]) for g in with_msp)
    return {
        "tasks": total,
        "tasks_with_msp": len(with_msp),
        "msp_coverage": len(with_msp) / total if total else 0.0,
        "mean_msp": (sum(int(g["minimal_sufficient_step"]) for g in with_msp) / len(with_msp))
        if with_msp
        else None,
        "median_like_msp": sorted(int(g["minimal_sufficient_step"]) for g in with_msp)[
            len(with_msp) // 2
        ]
        if with_msp
        else None,
        "distribution": dict(
            sorted(counter.items(), key=lambda kv: (999 if kv[0] == "None" else int(kv[0])))
        ),
    }


def main() -> None:
    rows = _load_rows()
    train_rows, test_rows, x_train, y_train, x_test, y_test, feature_names = _build_matrix(rows)
    weights = _fit_logistic_regression(x_train, y_train)
    train_probs = _sigmoid(x_train @ weights)
    test_probs = _sigmoid(x_test @ weights)
    threshold, train_report = _best_threshold(y_train, train_probs)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row, prob, y in zip(test_rows, test_probs, y_test, strict=True):
        enriched = dict(row)
        enriched["prob"] = float(prob)
        enriched["pred"] = int(prob >= threshold)
        enriched["label"] = int(y)
        key = (row["run_name"], row["task_id"])
        grouped[key].append(enriched)

    task_groups: list[dict[str, object]] = []
    for (run_name, task_id), group_rows in grouped.items():
        group_rows.sort(key=lambda r: int(float(r["step_index"])))
        label = int(group_rows[0]["label"])
        msp = _minimal_sufficient_step(group_rows, label)
        first_correct = _first_correct_step(group_rows, label)
        task_groups.append(
            {
                "run_name": run_name,
                "task_id": task_id,
                "benchmark": group_rows[0]["benchmark"],
                "small_family": group_rows[0]["small_family"],
                "label": label,
                "num_prefixes": len(group_rows),
                "first_correct_step": first_correct,
                "minimal_sufficient_step": msp,
                "trajectory": [
                    {
                        "step_index": int(float(r["step_index"])),
                        "prob": round(float(r["prob"]), 6),
                        "pred": int(r["pred"]),
                    }
                    for r in group_rows
                ],
            }
        )

    overall = _summarize(task_groups)

    by_benchmark = {}
    for benchmark in sorted({g["benchmark"] for g in task_groups}):
        subset = [g for g in task_groups if g["benchmark"] == benchmark]
        by_benchmark[benchmark] = _summarize(subset)

    by_small_family = {}
    for family in sorted({g["small_family"] for g in task_groups}):
        subset = [g for g in task_groups if g["small_family"] == family]
        by_small_family[family] = _summarize(subset)

    representative_early = sorted(
        [g for g in task_groups if g["minimal_sufficient_step"] is not None],
        key=lambda g: (
            int(g["minimal_sufficient_step"]),
            g["num_prefixes"],
            g["benchmark"],
            g["small_family"],
            g["task_id"],
        ),
    )[:12]
    representative_none = sorted(
        [g for g in task_groups if g["minimal_sufficient_step"] is None],
        key=lambda g: (-g["num_prefixes"], g["benchmark"], g["small_family"], g["task_id"]),
    )[:12]

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "feature_set": "state_plus_small_family_plus_benchmark",
        "feature_names": feature_names[1:],
        "target": "small_full_trace_success",
        "threshold": threshold,
        "train_metrics": train_report,
        "test_rows": int(len(test_rows)),
        "test_tasks": int(len(task_groups)),
        "overall": overall,
        "by_benchmark": by_benchmark,
        "by_small_family": by_small_family,
        "representative_early_msp_tasks": representative_early,
        "representative_no_msp_tasks": representative_none,
        "task_groups": task_groups,
        "notes": [
            "Minimal sufficient prefix is defined as the earliest step whose prediction matches the true label and remains unchanged for all later steps.",
            "This first version uses a discrete prediction trajectory with a single global threshold chosen on train split.",
            "Large-family features are intentionally excluded because the target is small full-trace success, not takeover value.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Minimal Sufficient Prefix Analysis",
        "",
        "## 設定",
        "- target: `small_full_trace_success`",
        "- feature set: `state + small_family + benchmark`",
        "- 定義: 最早一步 `t*`，使得該步 prediction 已等於真實 label，且之後所有步都不再翻轉",
        "",
        "## 整體",
        f"- test tasks: `{summary['test_tasks']}`",
        f"- tasks with MSP: `{overall['tasks_with_msp']}` / `{overall['tasks']}` ({overall['msp_coverage']:.3f})",
        f"- mean MSP: `{overall['mean_msp']}`",
        f"- distribution: `{overall['distribution']}`",
        "",
        "## By Benchmark",
    ]
    for benchmark, stats in by_benchmark.items():
        lines += [
            f"### `{benchmark}`",
            f"- coverage: `{stats['tasks_with_msp']}/{stats['tasks']}` ({stats['msp_coverage']:.3f})",
            f"- mean MSP: `{stats['mean_msp']}`",
            f"- distribution: `{stats['distribution']}`",
            "",
        ]
    lines += ["## By Small Family"]
    for family, stats in by_small_family.items():
        lines += [
            f"### `{family}`",
            f"- coverage: `{stats['tasks_with_msp']}/{stats['tasks']}` ({stats['msp_coverage']:.3f})",
            f"- mean MSP: `{stats['mean_msp']}`",
            f"- distribution: `{stats['distribution']}`",
            "",
        ]
    lines += ["## Representative Early-MSP Tasks"]
    for row in representative_early[:8]:
        lines.append(
            f"- `{row['run_name']} :: {row['task_id']}`: msp=`{row['minimal_sufficient_step']}`, prefixes=`{row['num_prefixes']}`, label=`{row['label']}`"
        )
    lines += ["", "## Representative No-MSP Tasks"]
    for row in representative_none[:8]:
        lines.append(
            f"- `{row['run_name']} :: {row['task_id']}`: prefixes=`{row['num_prefixes']}`, label=`{row['label']}`"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis JSON to {OUTPUT_JSON}")
    print(f"Wrote analysis markdown to {OUTPUT_MD}")
    print(
        json.dumps(
            {
                "overall": overall,
                "by_benchmark": by_benchmark,
                "by_small_family": by_small_family,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
