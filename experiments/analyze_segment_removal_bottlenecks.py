from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
OUTPUT_JSON = EARLY_DIR / "segment_removal_bottlenecks.json"
OUTPUT_MD = ROOT / "docs" / "segment_removal_bottlenecks.md"


RUN_SPECS = [
    ("qwen_to_openai_50", "olympiadbench", "qwen", "openai"),
    ("qwen_to_anthropic_50", "olympiadbench", "qwen", "anthropic"),
    ("llama_to_openai_50", "olympiadbench", "llama", "openai"),
    ("llama_to_anthropic_50", "olympiadbench", "llama", "anthropic"),
    ("mistral_to_openai_50", "olympiadbench", "mistral", "openai"),
    ("mistral_to_anthropic_50", "olympiadbench", "mistral", "anthropic"),
    ("livebench_qwen_to_openai_30", "livebench_reasoning", "qwen", "openai"),
    ("livebench_qwen_to_anthropic_30", "livebench_reasoning", "qwen", "anthropic"),
    ("livebench_llama_to_openai_30", "livebench_reasoning", "llama", "openai"),
    ("livebench_llama_to_anthropic_30", "livebench_reasoning", "llama", "anthropic"),
    ("livebench_mistral_to_openai_30", "livebench_reasoning", "mistral", "openai"),
    ("livebench_mistral_to_anthropic_30", "livebench_reasoning", "mistral", "anthropic"),
]


STRUCTURAL_FEATURES = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
]

RECOMPUTABLE_TEXT_FEATURES = [
    "prefix_text_tokens",
    "backtracking_flag",
    "backtracking_mentions",
    "self_correction_cue_density",
    "certainty_density",
    "commitment_score",
    "hedge_density",
    "semantic_drift_score",
]


BACKTRACK_PATTERNS = [
    "go back",
    "back to",
    "revisit",
    "earlier",
    "previous step",
    "instead",
]

SELF_CORRECTION_PATTERNS = [
    "actually",
    "however",
    "but",
    "instead",
    "reconsider",
    "mistake",
    "correction",
    "on second thought",
]

CERTAINTY_PATTERNS = [
    "therefore",
    "thus",
    "hence",
    "so ",
    "combining these",
    "we conclude",
    "this implies",
    "it follows that",
    "final answer",
]

HEDGE_PATTERNS = [
    "maybe",
    "perhaps",
    "likely",
    "possibly",
    "seems",
    "suggests",
    "might",
    "could",
]


def stable_test_split(task_id: str, test_ratio: float = 0.2) -> bool:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    steps: int = 4000,
    l2: float = 1e-3,
) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        probs = sigmoid(x @ weights)
        grad = (x.T @ (probs - y)) / len(y)
        grad += l2 * weights
        weights -= lr * grad
    return weights


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def pattern_density(text: str, patterns: list[str]) -> tuple[int, float]:
    lowered = text.lower()
    hits = sum(lowered.count(pattern) for pattern in patterns)
    token_count = max(1, len(tokenize(text)))
    return hits, hits / token_count


def semantic_drift(prefix_before_current: str, current_text: str) -> float:
    prev = set(tokenize(prefix_before_current))
    curr = set(tokenize(current_text))
    if not curr:
        return 0.0
    if not prev:
        return 0.0
    overlap = len(prev & curr)
    return 1.0 - (overlap / max(1, len(curr)))


def extract_features_from_segments(prefix_segments: list[dict[str, object]]) -> dict[str, float]:
    texts = [str(seg["text"]) for seg in prefix_segments]
    prefix_text = "\n\n".join(texts)
    current_text = texts[-1] if texts else ""
    before_current = "\n\n".join(texts[:-1]) if len(texts) > 1 else ""

    prefix_tokens = float(len(tokenize(prefix_text)))
    current_tokens = float(len(tokenize(current_text)))

    backtrack_hits, _ = pattern_density(prefix_text, BACKTRACK_PATTERNS)
    _, correction_density = pattern_density(prefix_text, SELF_CORRECTION_PATTERNS)
    _, certainty_density = pattern_density(prefix_text, CERTAINTY_PATTERNS)
    _, hedge_density = pattern_density(prefix_text, HEDGE_PATTERNS)

    return {
        "step_index": float(len(prefix_segments)),
        "prefix_segments_count": float(len(prefix_segments)),
        "prefix_tokens": prefix_tokens,
        "current_segment_tokens": current_tokens,
        "prefix_text_tokens": prefix_tokens,
        "backtracking_flag": float(int(backtrack_hits > 0)),
        "backtracking_mentions": float(backtrack_hits),
        "self_correction_cue_density": correction_density,
        "certainty_density": certainty_density,
        "commitment_score": certainty_density - correction_density,
        "hedge_density": hedge_density,
        "semantic_drift_score": semantic_drift(before_current, current_text),
    }


def step_bucket(step_index: int) -> str:
    if step_index <= 1:
        return "1"
    if step_index == 2:
        return "2"
    if step_index == 3:
        return "3"
    return "4+"


def find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        p
        for p in run_dir.glob("*.json")
        if p.name != "summary.json" and "per_prefix_rows" not in p.name
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one main result json for {run_name}, found {candidates}")
    return candidates[0]


def load_task_rows() -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    for run_name, benchmark, small_family, large_family in RUN_SPECS:
        data = json.loads(find_result_json(run_name).read_text(encoding="utf-8"))
        for task in data:
            task = dict(task)
            task["_run_name"] = run_name
            task["_benchmark"] = benchmark
            task["_small_family"] = small_family
            task["_large_family"] = large_family
            out[(run_name, str(task["task_id"]))] = task
    return out


def build_recomputable_dataset(
    task_rows: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (run_name, task_id), task in task_rows.items():
        label = int(bool(task.get("full_trace_correct", False)))
        for step in task.get("prefix_oracle_steps", []):
            prefix_segments = step.get("prefix_segments", [])
            features = extract_features_from_segments(prefix_segments)
            row = {
                "run_name": run_name,
                "task_id": task_id,
                "benchmark": task["_benchmark"],
                "small_family": task["_small_family"],
                "large_family": task["_large_family"],
                "small_full_trace_success": label,
                "step_bucket": step_bucket(int(features["step_index"])),
                **features,
            }
            rows.append(row)
    return rows


def build_matrix(
    rows: list[dict[str, object]],
    feature_names: list[str],
    include_family: bool,
    include_benchmark: bool,
):
    train_rows = [row for row in rows if not stable_test_split(str(row["task_id"]))]
    test_rows = [row for row in rows if stable_test_split(str(row["task_id"]))]

    small_families = sorted({str(row["small_family"]) for row in rows})
    large_families = sorted({str(row["large_family"]) for row in rows})
    benchmarks = sorted({str(row["benchmark"]) for row in rows})

    expanded_feature_names = list(feature_names)
    if include_family:
        expanded_feature_names += [f"small_family={name}" for name in small_families]
        expanded_feature_names += [f"large_family={name}" for name in large_families]
    if include_benchmark:
        expanded_feature_names += [f"benchmark={name}" for name in benchmarks]

    def encode(selected_rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(selected_rows), len(expanded_feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0
            offset = 1
            for feat in feature_names:
                x[i, offset] = float(row[feat])
                offset += 1
            if include_family:
                for name in small_families:
                    x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                    offset += 1
                for name in large_families:
                    x[i, offset] = 1.0 if row["large_family"] == name else 0.0
                    offset += 1
            if include_benchmark:
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

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "feature_names": expanded_feature_names,
        "means": means,
        "stds": stds,
        "small_families": small_families,
        "large_families": large_families,
        "benchmarks": benchmarks,
        "base_feature_names": feature_names,
    }


def encode_counterfactual(
    row: dict[str, object],
    *,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    small_families: list[str],
    large_families: list[str],
    benchmarks: list[str],
    include_family: bool,
    include_benchmark: bool,
) -> np.ndarray:
    x = np.zeros((1, len(feature_names) + 1), dtype=np.float64)
    x[0, 0] = 1.0
    offset = 1
    for feat in STRUCTURAL_FEATURES + RECOMPUTABLE_TEXT_FEATURES:
        x[0, offset] = float(row[feat])
        offset += 1
    if include_family:
        for name in small_families:
            x[0, offset] = 1.0 if row["small_family"] == name else 0.0
            offset += 1
        for name in large_families:
            x[0, offset] = 1.0 if row["large_family"] == name else 0.0
            offset += 1
    if include_benchmark:
        for name in benchmarks:
            x[0, offset] = 1.0 if row["benchmark"] == name else 0.0
            offset += 1
    x[:, 1:] = (x[:, 1:] - means) / stds
    return x


def test_report(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    acc = float(np.mean(preds == y_true))
    return {
        "auroc": roc_auc_score(y_true, probs),
        "accuracy@0.5": acc,
        "base_rate": float(np.mean(y_true)),
        "rows": int(len(y_true)),
    }


def select_step(task: dict[str, object], step_index: int) -> dict[str, object]:
    for step in task.get("prefix_oracle_steps", []):
        if int(step["step_index"]) == int(step_index):
            return step
    raise KeyError(f"Step {step_index} not found")


def build_counterfactual_row(
    task: dict[str, object], kept_segments: list[dict[str, object]]
) -> dict[str, object]:
    features = extract_features_from_segments(kept_segments)
    return {
        "benchmark": task["_benchmark"],
        "small_family": task["_small_family"],
        "large_family": task["_large_family"],
        **features,
    }


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> None:
    task_rows = load_task_rows()
    dataset_rows = build_recomputable_dataset(task_rows)
    feature_names = STRUCTURAL_FEATURES + RECOMPUTABLE_TEXT_FEATURES
    split = build_matrix(dataset_rows, feature_names, include_family=True, include_benchmark=True)
    weights = fit_logistic_regression(split["x_train"], split["y_train"])
    test_probs = sigmoid(split["x_test"] @ weights)

    bottleneck = json.loads(
        (EARLY_DIR / "early_decision_bottlenecks.json").read_text(encoding="utf-8")
    )
    task_groups = bottleneck["task_groups"]

    evaluated = []
    decisive_drops = []
    control_drops = []
    decisive_cross70_failures = 0
    control_cross70_failures = 0

    for row in task_groups:
        cross70 = row.get("first_cross_70")
        if cross70 is None or int(cross70) < 2:
            continue
        key = (row["run_name"], row["task_id"])
        task = task_rows[key]
        decisive_step = select_step(task, int(cross70))
        prefix_segments = list(decisive_step["prefix_segments"])
        original_cf = build_counterfactual_row(task, prefix_segments)
        original_x = encode_counterfactual(
            original_cf,
            feature_names=split["feature_names"],
            means=split["means"],
            stds=split["stds"],
            small_families=split["small_families"],
            large_families=split["large_families"],
            benchmarks=split["benchmarks"],
            include_family=True,
            include_benchmark=True,
        )
        original_prob = float(sigmoid(original_x @ weights)[0])
        label = int(row["label"])
        original_correct_prob = original_prob if label == 1 else 1.0 - original_prob

        # Remove decisive segment from the decisive prefix.
        decisive_removed_segments = prefix_segments[:-1]
        decisive_cf = build_counterfactual_row(task, decisive_removed_segments)
        decisive_x = encode_counterfactual(
            decisive_cf,
            feature_names=split["feature_names"],
            means=split["means"],
            stds=split["stds"],
            small_families=split["small_families"],
            large_families=split["large_families"],
            benchmarks=split["benchmarks"],
            include_family=True,
            include_benchmark=True,
        )
        decisive_prob = float(sigmoid(decisive_x @ weights)[0])
        decisive_correct_prob = decisive_prob if label == 1 else 1.0 - decisive_prob
        decisive_drop = original_correct_prob - decisive_correct_prob

        # Control: remove the first segment but keep the decisive one.
        control_removed_segments = prefix_segments[1:]
        control_cf = build_counterfactual_row(task, control_removed_segments)
        control_x = encode_counterfactual(
            control_cf,
            feature_names=split["feature_names"],
            means=split["means"],
            stds=split["stds"],
            small_families=split["small_families"],
            large_families=split["large_families"],
            benchmarks=split["benchmarks"],
            include_family=True,
            include_benchmark=True,
        )
        control_prob = float(sigmoid(control_x @ weights)[0])
        control_correct_prob = control_prob if label == 1 else 1.0 - control_prob
        control_drop = original_correct_prob - control_correct_prob

        decisive_drops.append(decisive_drop)
        control_drops.append(control_drop)
        decisive_cross70_failures += int(decisive_correct_prob < 0.7)
        control_cross70_failures += int(control_correct_prob < 0.7)

        evaluated.append(
            {
                "run_name": row["run_name"],
                "task_id": row["task_id"],
                "benchmark": row["benchmark"],
                "small_family": row["small_family"],
                "label": label,
                "cross70_step": int(cross70),
                "original_correct_prob": original_correct_prob,
                "decisive_removed_correct_prob": decisive_correct_prob,
                "control_removed_correct_prob": control_correct_prob,
                "decisive_drop": decisive_drop,
                "control_drop": control_drop,
                "decisive_below_70": int(decisive_correct_prob < 0.7),
                "control_below_70": int(control_correct_prob < 0.7),
            }
        )

    evaluated.sort(key=lambda item: item["decisive_drop"] - item["control_drop"], reverse=True)
    output = {
        "notes": [
            "This is a first-pass segment removal analysis using a recomputable Early Decision probe.",
            "The probe uses structural plus recomputable text features with family and benchmark one-hot features.",
            "For each task with cross70 >= 2, we remove the decisive segment from the decisive prefix and compare against a control removal that drops the first segment instead.",
        ],
        "probe_features": feature_names,
        "probe_test_metrics": test_report(split["y_test"], test_probs),
        "evaluated_tasks": len(evaluated),
        "overall": {
            "mean_decisive_drop": avg(decisive_drops),
            "mean_control_drop": avg(control_drops),
            "mean_drop_gap": avg(
                [d - c for d, c in zip(decisive_drops, control_drops, strict=True)]
            ),
            "decisive_below_70_rate": decisive_cross70_failures / len(evaluated)
            if evaluated
            else None,
            "control_below_70_rate": control_cross70_failures / len(evaluated)
            if evaluated
            else None,
        },
        "by_benchmark": {},
        "by_small_family": {},
        "representative_large_effects": evaluated[:5],
        "rows": evaluated,
    }

    for key in ("benchmark", "small_family"):
        buckets: dict[str, list[dict[str, object]]] = {}
        for row in evaluated:
            buckets.setdefault(str(row[key]), []).append(row)
        section = output["by_benchmark"] if key == "benchmark" else output["by_small_family"]
        for bucket_name, rows in buckets.items():
            section[bucket_name] = {
                "count": len(rows),
                "mean_decisive_drop": avg([float(r["decisive_drop"]) for r in rows]),
                "mean_control_drop": avg([float(r["control_drop"]) for r in rows]),
                "mean_drop_gap": avg(
                    [float(r["decisive_drop"]) - float(r["control_drop"]) for r in rows]
                ),
                "decisive_below_70_rate": sum(int(r["decisive_below_70"]) for r in rows)
                / len(rows),
                "control_below_70_rate": sum(int(r["control_below_70"]) for r in rows) / len(rows),
            }

    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Segment Removal Bottlenecks",
        "",
        "這份分析用第一版可重算 probe 檢查：拿掉 `cross70` 對應的 decisive segment 之後，Early Decision 的信心會不會明顯崩掉。",
        "",
        "## Overall",
        "",
        f"- evaluated tasks: `{output['evaluated_tasks']}`",
        f"- probe test AUROC: `{output['probe_test_metrics']['auroc']:.3f}`",
        f"- mean decisive drop: `{output['overall']['mean_decisive_drop']:.3f}`",
        f"- mean control drop: `{output['overall']['mean_control_drop']:.3f}`",
        f"- mean drop gap: `{output['overall']['mean_drop_gap']:.3f}`",
        f"- decisive below 0.7 rate: `{output['overall']['decisive_below_70_rate']:.3f}`",
        f"- control below 0.7 rate: `{output['overall']['control_below_70_rate']:.3f}`",
        "",
        "## By Benchmark",
        "",
    ]
    for name, stats in output["by_benchmark"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- count: `{stats['count']}`",
                f"- mean decisive drop: `{stats['mean_decisive_drop']:.3f}`",
                f"- mean control drop: `{stats['mean_control_drop']:.3f}`",
                f"- mean drop gap: `{stats['mean_drop_gap']:.3f}`",
                f"- decisive below 0.7 rate: `{stats['decisive_below_70_rate']:.3f}`",
                f"- control below 0.7 rate: `{stats['control_below_70_rate']:.3f}`",
                "",
            ]
        )
    lines.extend(["## Representative Large Effects", ""])
    for row in output["representative_large_effects"]:
        lines.extend(
            [
                f"### {row['run_name']} :: {row['task_id']}",
                "",
                f"- benchmark: `{row['benchmark']}`",
                f"- small_family: `{row['small_family']}`",
                f"- cross70_step: `{row['cross70_step']}`",
                f"- original correct-label prob: `{row['original_correct_prob']:.3f}`",
                f"- decisive-removed prob: `{row['decisive_removed_correct_prob']:.3f}`",
                f"- control-removed prob: `{row['control_removed_correct_prob']:.3f}`",
                f"- decisive drop: `{row['decisive_drop']:.3f}`",
                f"- control drop: `{row['control_drop']:.3f}`",
                "",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
