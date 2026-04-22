from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, TypedDict, cast

import numpy as np

from confidence_tom.infra.paths import results_root

DATASET_DIR = results_root() / "_prefix_predictor_v1"
DATASET_CSV = DATASET_DIR / "prefix_predictor_rows.csv"
OUTPUT_JSON = DATASET_DIR / "baseline_results.json"


STRUCTURAL_FEATURES = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
]

STATE_FEATURES = STRUCTURAL_FEATURES + [
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

Row = dict[str, str]
RowFilter = Callable[[Row], bool]


class CoefficientRow(TypedDict):
    feature: str
    weight: float


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def _stable_test_split(task_id: str, test_ratio: float = 0.2) -> bool:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    steps: int = 4000,
    l2: float = 1e-3,
) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        probs = _sigmoid(x @ weights)
        grad = (x.T @ (probs - y)) / len(y)
        grad += l2 * weights
        weights -= lr * grad
    return weights


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


def _binary_report_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
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


def _classification_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    report = _binary_report_at_threshold(y_true, probs, threshold)
    report["threshold"] = threshold
    report["auroc"] = _roc_auc_score(y_true, probs)
    report["brier"] = float(np.mean((probs - y_true) ** 2))
    report["base_rate"] = float(np.mean(y_true))
    return report


def _load_rows() -> list[Row]:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing dataset CSV: {DATASET_CSV}. Run build_prefix_predictor_dataset.py first."
        )
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_matrix(
    rows: list[Row],
    numeric_features: list[str],
    include_family_onehot: bool,
    include_benchmark_onehot: bool = False,
    *,
    train_filter: RowFilter | None = None,
    test_filter: RowFilter | None = None,
) -> SplitData:
    if train_filter is None and test_filter is None:
        train_rows: list[Row] = []
        test_rows: list[Row] = []
        for row in rows:
            (test_rows if _stable_test_split(row["task_id"]) else train_rows).append(row)
    else:
        assert train_filter is not None
        assert test_filter is not None
        train_rows = [row for row in rows if train_filter(row)]
        test_rows = [row for row in rows if test_filter(row)]

    small_families = sorted({row["small_family"] for row in rows})
    large_families = sorted({row["large_family"] for row in rows})
    benchmarks = sorted({row["benchmark"] for row in rows})

    feature_names = list(numeric_features)
    if include_family_onehot:
        feature_names += [f"small_family={name}" for name in small_families]
        feature_names += [f"large_family={name}" for name in large_families]
    if include_benchmark_onehot:
        feature_names += [f"benchmark={name}" for name in benchmarks]

    def encode(selected_rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(selected_rows), len(feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0  # bias
            offset = 1
            for feature in numeric_features:
                x[i, offset] = float(row[feature])
                offset += 1
            if include_family_onehot:
                for name in small_families:
                    x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                    offset += 1
                for name in large_families:
                    x[i, offset] = 1.0 if row["large_family"] == name else 0.0
                    offset += 1
            if include_benchmark_onehot:
                for name in benchmarks:
                    x[i, offset] = 1.0 if row["benchmark"] == name else 0.0
                    offset += 1
            y[i] = float(row["delta_positive"])
        return x, y

    x_train, y_train = encode(train_rows)
    x_test, y_test = encode(test_rows)

    means = np.mean(x_train[:, 1:], axis=0)
    stds = np.std(x_train[:, 1:], axis=0)
    stds[stds == 0.0] = 1.0
    x_train[:, 1:] = (x_train[:, 1:] - means) / stds
    x_test[:, 1:] = (x_test[:, 1:] - means) / stds

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=["bias", *feature_names],
    )


def _run_experiment(
    rows: list[Row],
    *,
    name: str,
    numeric_features: list[str],
    include_family_onehot: bool,
    include_benchmark_onehot: bool = False,
    train_filter: RowFilter | None = None,
    test_filter: RowFilter | None = None,
) -> dict[str, Any]:
    split = _build_matrix(
        rows,
        numeric_features,
        include_family_onehot,
        include_benchmark_onehot,
        train_filter=train_filter,
        test_filter=test_filter,
    )
    if len(split.y_train) == 0 or len(split.y_test) == 0:
        return {
            "name": name,
            "error": "empty_train_or_test_split",
            "train_size": int(len(split.y_train)),
            "test_size": int(len(split.y_test)),
        }
    if len(set(split.y_train.tolist())) < 2 or len(set(split.y_test.tolist())) < 2:
        return {
            "name": name,
            "error": "single_class_split",
            "train_size": int(len(split.y_train)),
            "test_size": int(len(split.y_test)),
            "train_base_rate": float(np.mean(split.y_train)) if len(split.y_train) else None,
            "test_base_rate": float(np.mean(split.y_test)) if len(split.y_test) else None,
        }
    weights = _fit_logistic_regression(split.x_train, split.y_train)
    train_probs = _sigmoid(split.x_train @ weights)
    test_probs = _sigmoid(split.x_test @ weights)
    threshold, train_binary = _best_threshold(split.y_train, train_probs)

    coefs: list[CoefficientRow] = [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in zip(split.feature_names, weights, strict=True)
    ]
    coefs.sort(key=lambda item: abs(item["weight"]), reverse=True)

    return {
        "name": name,
        "feature_count": len(split.feature_names) - 1,
        "features": split.feature_names[1:],
        "train_size": int(len(split.y_train)),
        "test_size": int(len(split.y_test)),
        "train_metrics": {
            **_classification_report(split.y_train, train_probs, threshold=threshold),
            "train_selected_threshold_metrics": train_binary,
        },
        "test_metrics": _classification_report(split.y_test, test_probs, threshold=threshold),
        "top_coefficients": coefs[:12],
    }


def main() -> None:
    rows = _load_rows()
    experiments: list[dict[str, Any]] = [
        _run_experiment(
            rows,
            name="structural_only",
            numeric_features=STRUCTURAL_FEATURES,
            include_family_onehot=False,
        ),
        _run_experiment(
            rows,
            name="state_signals",
            numeric_features=STATE_FEATURES,
            include_family_onehot=False,
        ),
        _run_experiment(
            rows,
            name="state_plus_family",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
        ),
        _run_experiment(
            rows,
            name="state_plus_family_plus_benchmark",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
            include_benchmark_onehot=True,
        ),
        _run_experiment(
            rows,
            name="olympiad_only_state_plus_family",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
            train_filter=lambda row: row["benchmark"] == "olympiadbench"
            and not _stable_test_split(row["task_id"]),
            test_filter=lambda row: row["benchmark"] == "olympiadbench"
            and _stable_test_split(row["task_id"]),
        ),
        _run_experiment(
            rows,
            name="livebench_only_state_plus_family",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
            train_filter=lambda row: row["benchmark"] == "livebench_reasoning"
            and not _stable_test_split(row["task_id"]),
            test_filter=lambda row: row["benchmark"] == "livebench_reasoning"
            and _stable_test_split(row["task_id"]),
        ),
        _run_experiment(
            rows,
            name="cross_domain_train_olympiad_test_livebench",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
            train_filter=lambda row: row["benchmark"] == "olympiadbench",
            test_filter=lambda row: row["benchmark"] == "livebench_reasoning",
        ),
        _run_experiment(
            rows,
            name="cross_domain_train_livebench_test_olympiad",
            numeric_features=STATE_FEATURES,
            include_family_onehot=True,
            train_filter=lambda row: row["benchmark"] == "livebench_reasoning",
            test_filter=lambda row: row["benchmark"] == "olympiadbench",
        ),
    ]

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "rows": len(rows),
        "target": "delta_positive",
        "task_split": "stable hash split by task_id (80/20)",
        "notes": [
            "This is a first baseline on prefix-observable features only.",
            "It predicts whether large takeover yields positive gain (delta_t > 0).",
            "Rollout-derived leakage features are intentionally excluded.",
        ],
        "experiments": experiments,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote baseline results to {OUTPUT_JSON}")
    for exp in experiments:
        test = cast(dict[str, float], exp["test_metrics"])
        print(
            f"{exp['name']}: "
            + (
                f"error={exp['error']}"
                if "error" in exp
                else (
                    f"auroc={test['auroc']:.3f} "
                    f"f1={test['f1']:.3f} "
                    f"precision={test['precision']:.3f} "
                    f"recall={test['recall']:.3f}"
                )
            )
        )


if __name__ == "__main__":
    main()
