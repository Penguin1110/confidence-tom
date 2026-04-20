from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "results" / "_early_decision_v1"
DATASET_CSV = DATASET_DIR / "early_decision_rows.csv"
OUTPUT_JSON = DATASET_DIR / "probability_stable_msp_analysis.json"
OUTPUT_MD = ROOT / "docs" / "probability_stable_msp_analysis.md"

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
THRESHOLDS = [0.6, 0.7, 0.8]


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
    return test_rows, x_train, y_train, x_test, y_test, ["bias", *feature_names]


def _stable_correct_prob(prob: float, label: int) -> float:
    return prob if label == 1 else (1.0 - prob)


def _probability_stable_msp(
    sorted_rows: list[dict[str, object]], label: int, threshold: float
) -> int | None:
    probs = [_stable_correct_prob(float(row["prob"]), label) for row in sorted_rows]
    steps = [int(float(row["step_index"])) for row in sorted_rows]
    for i, p in enumerate(probs):
        if p < threshold:
            continue
        if all(q >= threshold for q in probs[i:]):
            return steps[i]
    return None


def _summarize(task_groups: list[dict[str, object]], key: str) -> dict[str, object]:
    total = len(task_groups)
    with_msp = [g for g in task_groups if g[key] is not None]
    dist = Counter(str(g[key]) for g in with_msp)
    return {
        "tasks": total,
        "tasks_with_msp": len(with_msp),
        "coverage": len(with_msp) / total if total else 0.0,
        "mean_msp": (sum(int(g[key]) for g in with_msp) / len(with_msp)) if with_msp else None,
        "distribution": dict(sorted(dist.items(), key=lambda kv: int(kv[0]))),
    }


def _slice_summary(
    task_groups: list[dict[str, object]], key: str, field: str
) -> dict[str, dict[str, object]]:
    out = {}
    for value in sorted({str(g[field]) for g in task_groups}):
        subset = [g for g in task_groups if str(g[field]) == value]
        out[value] = _summarize(subset, key)
    return out


def main() -> None:
    rows = _load_rows()
    test_rows, x_train, y_train, x_test, y_test, feature_names = _build_matrix(rows)
    weights = _fit_logistic_regression(x_train, y_train)
    probs = _sigmoid(x_test @ weights)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row, prob, y in zip(test_rows, probs, y_test, strict=True):
        enriched = dict(row)
        enriched["prob"] = float(prob)
        enriched["label"] = int(y)
        grouped[(row["run_name"], row["task_id"])].append(enriched)

    task_groups = []
    for (run_name, task_id), group_rows in grouped.items():
        group_rows.sort(key=lambda r: int(float(r["step_index"])))
        label = int(group_rows[0]["label"])
        item = {
            "run_name": run_name,
            "task_id": task_id,
            "benchmark": group_rows[0]["benchmark"],
            "small_family": group_rows[0]["small_family"],
            "label": label,
            "num_prefixes": len(group_rows),
            "trajectory": [
                {
                    "step_index": int(float(r["step_index"])),
                    "prob_success": round(float(r["prob"]), 6),
                    "prob_correct_label": round(_stable_correct_prob(float(r["prob"]), label), 6),
                }
                for r in group_rows
            ],
        }
        for threshold in THRESHOLDS:
            item[f"msp_p{int(threshold * 100)}"] = _probability_stable_msp(
                group_rows, label, threshold
            )
        task_groups.append(item)

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "feature_set": "state_plus_small_family_plus_benchmark",
        "target": "small_full_trace_success",
        "thresholds": THRESHOLDS,
        "test_tasks": len(task_groups),
        "by_threshold": {},
        "task_groups": task_groups,
        "notes": [
            "Probability-stable MSP requires confidence on the correct label to stay above threshold for all later steps.",
            "For label=0, confidence on the correct label is defined as 1 - prob_success.",
            "This is stricter than the discrete MSP used earlier.",
        ],
    }

    for threshold in THRESHOLDS:
        key = f"msp_p{int(threshold * 100)}"
        summary["by_threshold"][str(threshold)] = {
            "overall": _summarize(task_groups, key),
            "by_benchmark": _slice_summary(task_groups, key, "benchmark"),
            "by_small_family": _slice_summary(task_groups, key, "small_family"),
            "representative_tasks": [
                {
                    "run_name": g["run_name"],
                    "task_id": g["task_id"],
                    "benchmark": g["benchmark"],
                    "small_family": g["small_family"],
                    "num_prefixes": g["num_prefixes"],
                    key: g[key],
                }
                for g in sorted(
                    [x for x in task_groups if x[key] is not None],
                    key=lambda x: (int(x[key]), x["benchmark"], x["small_family"], x["task_id"]),
                )[:10]
            ],
        }

    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Probability-Stable MSP Analysis",
        "",
        "這版 MSP 比離散版更嚴格：從某一步開始，對正確 label 的機率必須一直高於門檻。",
        "",
    ]
    for threshold in THRESHOLDS:
        stats = summary["by_threshold"][str(threshold)]
        overall = stats["overall"]
        lines += [
            f"## Threshold `{threshold}`",
            f"- coverage: `{overall['tasks_with_msp']}/{overall['tasks']}` ({overall['coverage']:.3f})",
            f"- mean MSP: `{overall['mean_msp']}`",
            f"- distribution: `{overall['distribution']}`",
            "",
            "### By Benchmark",
        ]
        for benchmark, s in stats["by_benchmark"].items():
            lines.append(
                f"- `{benchmark}`: coverage=`{s['tasks_with_msp']}/{s['tasks']}` ({s['coverage']:.3f}), mean_msp=`{s['mean_msp']}`, dist=`{s['distribution']}`"
            )
        lines += ["", "### By Small Family"]
        for family, s in stats["by_small_family"].items():
            lines.append(
                f"- `{family}`: coverage=`{s['tasks_with_msp']}/{s['tasks']}` ({s['coverage']:.3f}), mean_msp=`{s['mean_msp']}`, dist=`{s['distribution']}`"
            )
        lines += [""]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis JSON to {OUTPUT_JSON}")
    print(f"Wrote analysis markdown to {OUTPUT_MD}")
    print(json.dumps(summary["by_threshold"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
