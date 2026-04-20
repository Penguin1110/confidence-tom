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
OUTPUT_JSON = DATASET_DIR / "early_decision_bottlenecks.json"
OUTPUT_MD = ROOT / "docs" / "early_decision_bottlenecks.md"

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
    return test_rows, x_train, y_train, x_test, y_test


def _correct_label_prob(prob_success: float, label: int) -> float:
    return prob_success if label == 1 else (1.0 - prob_success)


def _argmax_index(values: list[float]) -> int:
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values[1:], start=1):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def _summarize(groups: list[dict[str, object]], field: str) -> dict[str, object]:
    counter = Counter(str(g[field]) for g in groups if g[field] is not None)
    numeric = [int(g[field]) for g in groups if g[field] is not None]
    return {
        "tasks": len(groups),
        "count": len(numeric),
        "mean": (sum(numeric) / len(numeric)) if numeric else None,
        "distribution": dict(sorted(counter.items(), key=lambda kv: int(kv[0]))),
    }


def _slice(groups: list[dict[str, object]], field: str, key: str) -> dict[str, dict[str, object]]:
    out = {}
    for value in sorted({str(g[key]) for g in groups}):
        subset = [g for g in groups if str(g[key]) == value]
        out[value] = _summarize(subset, field)
    return out


def main() -> None:
    rows = _load_rows()
    test_rows, x_train, y_train, x_test, y_test = _build_matrix(rows)
    weights = _fit_logistic_regression(x_train, y_train)
    probs = _sigmoid(x_test @ weights)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row, prob, y in zip(test_rows, probs, y_test, strict=True):
        enriched = dict(row)
        enriched["prob_success"] = float(prob)
        enriched["label"] = int(y)
        enriched["prob_correct_label"] = _correct_label_prob(float(prob), int(y))
        grouped[(row["run_name"], row["task_id"])].append(enriched)

    task_groups: list[dict[str, object]] = []
    for (run_name, task_id), group_rows in grouped.items():
        group_rows.sort(key=lambda r: int(float(r["step_index"])))
        label = int(group_rows[0]["label"])
        steps = [int(float(r["step_index"])) for r in group_rows]
        correct_probs = [float(r["prob_correct_label"]) for r in group_rows]
        jumps = [correct_probs[0]] + [
            correct_probs[i] - correct_probs[i - 1] for i in range(1, len(correct_probs))
        ]
        max_jump_idx = _argmax_index(jumps)
        max_jump_step = steps[max_jump_idx]
        max_jump_value = jumps[max_jump_idx]
        first_cross_60 = next(
            (s for s, p in zip(steps, correct_probs, strict=True) if p >= 0.6), None
        )
        first_cross_70 = next(
            (s for s, p in zip(steps, correct_probs, strict=True) if p >= 0.7), None
        )
        first_cross_80 = next(
            (s for s, p in zip(steps, correct_probs, strict=True) if p >= 0.8), None
        )
        task_groups.append(
            {
                "run_name": run_name,
                "task_id": task_id,
                "benchmark": group_rows[0]["benchmark"],
                "small_family": group_rows[0]["small_family"],
                "label": label,
                "num_prefixes": len(group_rows),
                "max_jump_step": max_jump_step,
                "max_jump_value": round(max_jump_value, 6),
                "first_cross_60": first_cross_60,
                "first_cross_70": first_cross_70,
                "first_cross_80": first_cross_80,
                "trajectory": [
                    {
                        "step_index": s,
                        "prob_correct_label": round(p, 6),
                        "jump_from_prev": round(j, 6),
                    }
                    for s, p, j in zip(steps, correct_probs, jumps, strict=True)
                ],
            }
        )

    summary = {
        "dataset_csv": str(DATASET_CSV),
        "test_tasks": len(task_groups),
        "overall": {
            "max_jump_step": _summarize(task_groups, "max_jump_step"),
            "first_cross_60": _summarize(task_groups, "first_cross_60"),
            "first_cross_70": _summarize(task_groups, "first_cross_70"),
            "first_cross_80": _summarize(task_groups, "first_cross_80"),
        },
        "by_benchmark": {
            "max_jump_step": _slice(task_groups, "max_jump_step", "benchmark"),
            "first_cross_60": _slice(task_groups, "first_cross_60", "benchmark"),
            "first_cross_70": _slice(task_groups, "first_cross_70", "benchmark"),
        },
        "by_small_family": {
            "max_jump_step": _slice(task_groups, "max_jump_step", "small_family"),
            "first_cross_60": _slice(task_groups, "first_cross_60", "small_family"),
            "first_cross_70": _slice(task_groups, "first_cross_70", "small_family"),
        },
        "representative_early_bottlenecks": sorted(
            task_groups,
            key=lambda g: (
                int(g["max_jump_step"]),
                -float(g["max_jump_value"]),
                g["benchmark"],
                g["small_family"],
                g["task_id"],
            ),
        )[:12],
        "representative_late_bottlenecks": sorted(
            task_groups,
            key=lambda g: (
                -int(g["max_jump_step"]),
                -float(g["max_jump_value"]),
                g["benchmark"],
                g["small_family"],
                g["task_id"],
            ),
        )[:12],
        "task_groups": task_groups,
        "notes": [
            "max_jump_step is the step where probability on the correct label increases the most.",
            "first_cross_60/70/80 are the earliest steps where correct-label probability crosses the threshold.",
            "This is a bottleneck-style proxy analysis on top of the early-decision predictor trajectory, not a segment-removal causal ablation.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Early Decision Bottleneck Analysis",
        "",
        "這版先做最穩的 step-level bottleneck proxy：看哪一步讓對正確 label 的機率跳最多，以及最早在哪一步跨過固定門檻。",
        "",
        "## 整體",
        f"- max_jump_step distribution: `{summary['overall']['max_jump_step']['distribution']}`",
        f"- first_cross_60 distribution: `{summary['overall']['first_cross_60']['distribution']}`",
        f"- first_cross_70 distribution: `{summary['overall']['first_cross_70']['distribution']}`",
        "",
        "## By Benchmark",
    ]
    for benchmark, stats in summary["by_benchmark"]["max_jump_step"].items():
        lines.append(
            f"- `{benchmark}`: max_jump=`{stats['distribution']}`, cross60=`{summary['by_benchmark']['first_cross_60'][benchmark]['distribution']}`, cross70=`{summary['by_benchmark']['first_cross_70'][benchmark]['distribution']}`"
        )
    lines += ["", "## By Small Family"]
    for family, stats in summary["by_small_family"]["max_jump_step"].items():
        lines.append(
            f"- `{family}`: max_jump=`{stats['distribution']}`, cross60=`{summary['by_small_family']['first_cross_60'][family]['distribution']}`, cross70=`{summary['by_small_family']['first_cross_70'][family]['distribution']}`"
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
