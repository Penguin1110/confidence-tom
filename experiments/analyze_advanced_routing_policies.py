from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
PREFIX_DIR = RESULTS_DIR / "_prefix_predictor_v1"

EARLY_CSV = EARLY_DIR / "early_decision_rows.csv"
PREFIX_CSV = PREFIX_DIR / "prefix_predictor_rows.csv"

OUTPUT_JSON = EARLY_DIR / "advanced_routing_policies.json"
OUTPUT_MD = ROOT / "docs" / "advanced_routing_policies.md"


EARLY_STATE_FEATURES = [
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

PREFIX_STATE_FEATURES = [
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


def stable_partition(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.2:
        return "test"
    if value < 0.35:
        return "val"
    return "train"


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


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        p
        for p in run_dir.glob("*.json")
        if p.name != "summary.json" and "per_prefix_rows" not in p.name
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one result json for {run_name}, found {candidates}")
    return candidates[0]


def load_oracle_index() -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for run_name, benchmark, small_family, large_family in RUN_SPECS:
        data = json.loads(find_result_json(run_name).read_text(encoding="utf-8"))
        for task in data:
            steps = {}
            for step in task["prefix_oracle_steps"]:
                steps[int(step["step_index"])] = {
                    "prefix_id": step["prefix_id"],
                    "large_takeover_correct": int(bool(step["large_takeover_correct"])),
                    "small_continue_correct": int(bool(step["small_continue_correct"])),
                    "delta_correctness": float(step["delta_correctness"]),
                    "large_total_tokens": step.get("large_takeover_cost", {}).get("total_tokens"),
                }
            index[(run_name, str(task["task_id"]))] = {
                "benchmark": benchmark,
                "small_family": small_family,
                "large_family": large_family,
                "full_trace_correct": int(bool(task["full_trace_correct"])),
                "steps": steps,
            }
    return index


def build_early_model(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    small_families = sorted({r["small_family"] for r in rows})
    feature_names = list(EARLY_STATE_FEATURES) + [f"small_family={name}" for name in small_families]
    train_rows = [r for r in rows if stable_partition(r["task_id"]) == "train"]
    val_rows = [r for r in rows if stable_partition(r["task_id"]) == "val"]
    test_rows = [r for r in rows if stable_partition(r["task_id"]) == "test"]

    def encode(selected_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(selected_rows), len(feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0
            offset = 1
            for feat in EARLY_STATE_FEATURES:
                x[i, offset] = float(row[feat])
                offset += 1
            for name in small_families:
                x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                offset += 1
            y[i] = float(row["small_full_trace_success"])
        return x, y

    x_train, y_train = encode(train_rows)
    x_val, y_val = encode(val_rows)
    x_test, y_test = encode(test_rows)

    means = np.mean(x_train[:, 1:], axis=0)
    stds = np.std(x_train[:, 1:], axis=0)
    stds[stds == 0.0] = 1.0
    for arr in (x_train, x_val, x_test):
        arr[:, 1:] = (arr[:, 1:] - means) / stds

    weights = fit_logistic_regression(x_train, y_train)
    all_rows = train_rows + val_rows + test_rows
    x_all, _ = encode(all_rows)
    x_all[:, 1:] = (x_all[:, 1:] - means) / stds
    probs = sigmoid(x_all @ weights)
    return (
        probs,
        ["train"] * len(train_rows) + ["val"] * len(val_rows) + ["test"] * len(test_rows),
        np.array(all_rows, dtype=object),
        y_test,
    )


def build_gain_models(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    small_families = sorted({r["small_family"] for r in rows})
    large_families = sorted({r["large_family"] for r in rows})
    feature_names = (
        list(PREFIX_STATE_FEATURES)
        + [f"small_family={name}" for name in small_families]
        + [f"large_family={name}" for name in large_families]
    )
    train_rows = [r for r in rows if stable_partition(r["task_id"]) == "train"]
    val_rows = [r for r in rows if stable_partition(r["task_id"]) == "val"]
    test_rows = [r for r in rows if stable_partition(r["task_id"]) == "test"]

    def encode(
        selected_rows: list[dict[str, str]], label_key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(selected_rows), len(feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0
            offset = 1
            for feat in PREFIX_STATE_FEATURES:
                x[i, offset] = float(row[feat])
                offset += 1
            for name in small_families:
                x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                offset += 1
            for name in large_families:
                x[i, offset] = 1.0 if row["large_family"] == name else 0.0
                offset += 1
            y[i] = float(row[label_key])
        return x, y

    x_train_pos, y_train_pos = encode(train_rows, "delta_positive")
    x_val_pos, _ = encode(val_rows, "delta_positive")
    x_test_pos, _ = encode(test_rows, "delta_positive")

    x_train_neg, y_train_neg = encode(train_rows, "delta_negative")
    x_val_neg, _ = encode(val_rows, "delta_negative")
    x_test_neg, _ = encode(test_rows, "delta_negative")

    means = np.mean(x_train_pos[:, 1:], axis=0)
    stds = np.std(x_train_pos[:, 1:], axis=0)
    stds[stds == 0.0] = 1.0
    for arr in (x_train_pos, x_val_pos, x_test_pos, x_train_neg, x_val_neg, x_test_neg):
        arr[:, 1:] = (arr[:, 1:] - means) / stds

    pos_weights = fit_logistic_regression(x_train_pos, y_train_pos)
    neg_weights = fit_logistic_regression(x_train_neg, y_train_neg)

    all_rows = train_rows + val_rows + test_rows
    x_all, _ = encode(all_rows, "delta_positive")
    x_all[:, 1:] = (x_all[:, 1:] - means) / stds
    pos_probs = sigmoid(x_all @ pos_weights)
    neg_probs = sigmoid(x_all @ neg_weights)

    return (
        pos_probs,
        neg_probs,
        ["train"] * len(train_rows) + ["val"] * len(val_rows) + ["test"] * len(test_rows),
        np.array(all_rows, dtype=object),
    )


def join_rows(
    early_probs: np.ndarray,
    early_splits: list[str],
    early_rows: np.ndarray,
    pos_probs: np.ndarray,
    neg_probs: np.ndarray,
    gain_splits: list[str],
    gain_rows: np.ndarray,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    early_index = {}
    for prob, split, row in zip(early_probs, early_splits, early_rows, strict=True):
        key = (row["run_name"], row["task_id"], row["prefix_id"])
        early_index[key] = {
            "pred_success": float(prob),
            "split": split,
            "benchmark": row["benchmark"],
            "small_family": row["small_family"],
            "large_family": row["large_family"],
        }

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for pos_prob, neg_prob, split, row in zip(
        pos_probs, neg_probs, gain_splits, gain_rows, strict=True
    ):
        key = (row["run_name"], row["task_id"], row["prefix_id"])
        if key not in early_index:
            continue
        merged = dict(row)
        merged.update(early_index[key])
        merged["pred_positive"] = float(pos_prob)
        merged["pred_negative"] = float(neg_prob)
        merged["split"] = split
        grouped.setdefault((row["run_name"], row["task_id"]), []).append(merged)
    return grouped


def route_decision(row: dict[str, object], config: dict[str, object]) -> bool:
    failure_prob = 1.0 - float(row["pred_success"])
    if failure_prob < float(config["fail_threshold"]):
        return False
    if config["policy_type"] in {"dual_signal", "risk_aware", "benchmark_aware_risk_aware"}:
        if float(row["pred_positive"]) < float(config["positive_threshold"]):
            return False
    if config["policy_type"] in {"risk_aware", "benchmark_aware_risk_aware"}:
        neg_threshold = float(config["negative_threshold"])
        if float(row["pred_negative"]) > neg_threshold:
            return False
    return True


def route_decision_benchmark_aware(row: dict[str, object], config: dict[str, object]) -> bool:
    benchmark_cfg = config["per_benchmark"][str(row["benchmark"])]
    failure_prob = 1.0 - float(row["pred_success"])
    if failure_prob < float(benchmark_cfg["fail_threshold"]):
        return False
    if float(row["pred_positive"]) < float(benchmark_cfg["positive_threshold"]):
        return False
    if float(row["pred_negative"]) > float(benchmark_cfg["negative_threshold"]):
        return False
    return True


def simulate_policy(
    *,
    grouped_rows: dict[tuple[str, str], list[dict[str, object]]],
    oracle_index: dict[tuple[str, str], dict[str, object]],
    config: dict[str, object],
    split_name: str,
) -> dict[str, object]:
    records = []
    total = 0
    routed = 0
    wins = 0
    oracle_routed = 0
    oracle_wins = 0
    trigger_steps: list[int] = []

    for key, rows in grouped_rows.items():
        task_split = str(rows[0]["split"])
        if task_split != split_name:
            continue
        task_meta = oracle_index[key]
        baseline = int(task_meta["full_trace_correct"])
        selected_step = None
        for row in sorted(rows, key=lambda r: int(float(r["step_index"]))):
            step = int(float(row["step_index"]))
            if step > int(config["max_step"]):
                continue
            if config["policy_type"] == "benchmark_aware_risk_aware":
                route = route_decision_benchmark_aware(row, config)
            else:
                route = route_decision(row, config)
            if route:
                selected_step = step
                break

        if selected_step is None:
            final_correct = baseline
            route_used = 0
        else:
            final_correct = int(task_meta["steps"][selected_step]["large_takeover_correct"])
            route_used = 1
            routed += 1
            trigger_steps.append(selected_step)

        oracle_choice = None
        for step in sorted(task_meta["steps"]):
            if step > int(config["max_step"]):
                continue
            if task_meta["steps"][step]["large_takeover_correct"] > baseline:
                oracle_choice = step
                break
        if oracle_choice is None:
            oracle_final = baseline
        else:
            oracle_final = int(task_meta["steps"][oracle_choice]["large_takeover_correct"])
            oracle_routed += 1
        oracle_wins += oracle_final

        wins += final_correct
        total += 1
        records.append(
            {
                "run_name": key[0],
                "task_id": key[1],
                "benchmark": task_meta["benchmark"],
                "small_family": task_meta["small_family"],
                "baseline_small_correct": baseline,
                "policy_final_correct": final_correct,
                "route_used": route_used,
                "trigger_step": selected_step,
            }
        )

    baseline_accuracy = avg([float(r["baseline_small_correct"]) for r in records])
    policy_accuracy = avg([float(r["policy_final_correct"]) for r in records])
    oracle_accuracy = oracle_wins / total if total else None
    return {
        "split": split_name,
        "tasks": total,
        "baseline_small_accuracy": baseline_accuracy,
        "policy_accuracy": policy_accuracy,
        "policy_gain_over_small": None
        if policy_accuracy is None or baseline_accuracy is None
        else policy_accuracy - baseline_accuracy,
        "oracle_accuracy_same_budget": oracle_accuracy,
        "gap_to_oracle": None
        if oracle_accuracy is None or policy_accuracy is None
        else oracle_accuracy - policy_accuracy,
        "route_rate": routed / total if total else None,
        "oracle_route_rate": oracle_routed / total if total else None,
        "mean_trigger_step": avg([float(x) for x in trigger_steps]),
        "records": records,
    }


def summarize_by_group(sim: dict[str, object], group_key: str) -> dict[str, object]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in sim["records"]:
        buckets.setdefault(str(row[group_key]), []).append(row)
    out = {}
    for name, rows in buckets.items():
        small_acc = avg([float(r["baseline_small_correct"]) for r in rows])
        policy_acc = avg([float(r["policy_final_correct"]) for r in rows])
        out[name] = {
            "tasks": len(rows),
            "baseline_small_accuracy": small_acc,
            "policy_accuracy": policy_acc,
            "policy_gain_over_small": None
            if small_acc is None or policy_acc is None
            else policy_acc - small_acc,
            "route_rate": avg([float(r["route_used"]) for r in rows]),
            "mean_trigger_step": avg(
                [float(r["trigger_step"]) for r in rows if r["trigger_step"] is not None]
            ),
        }
    return out


def choose_best_config(configs: list[dict[str, object]]) -> dict[str, object]:
    ranked = sorted(
        configs,
        key=lambda x: (
            x["val"]["policy_gain_over_small"],
            -(x["val"]["gap_to_oracle"] or 999.0),
            -(x["val"]["route_rate"] or 999.0),
        ),
        reverse=True,
    )
    return ranked[0]


def main() -> None:
    early_rows = load_csv_rows(EARLY_CSV)
    prefix_rows = load_csv_rows(PREFIX_CSV)
    for row in prefix_rows:
        row["delta_negative"] = "1" if row["delta_sign"] == "negative" else "0"

    early_probs, early_splits, early_rows_arr, _ = build_early_model(early_rows)
    pos_probs, neg_probs, gain_splits, gain_rows_arr = build_gain_models(prefix_rows)
    grouped_rows = join_rows(
        early_probs,
        early_splits,
        early_rows_arr,
        pos_probs,
        neg_probs,
        gain_splits,
        gain_rows_arr,
    )
    oracle_index = load_oracle_index()

    max_steps = [1, 2, 3]
    fail_thresholds = [0.4, 0.5, 0.6, 0.7]
    positive_thresholds = [0.2, 0.3, 0.4, 0.5]
    negative_thresholds = [0.2, 0.3, 0.4, 0.5]

    all_results: dict[str, dict[str, object]] = {}

    failure_only = []
    for max_step, fail_threshold in product(max_steps, fail_thresholds):
        cfg = {
            "policy_type": "failure_only",
            "max_step": max_step,
            "fail_threshold": fail_threshold,
        }
        failure_only.append(
            {
                "config": cfg,
                "val": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="val",
                ),
                "test": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="test",
                ),
            }
        )
    all_results["failure_only"] = choose_best_config(failure_only)

    dual_signal = []
    for max_step, fail_threshold, positive_threshold in product(
        max_steps, fail_thresholds, positive_thresholds
    ):
        cfg = {
            "policy_type": "dual_signal",
            "max_step": max_step,
            "fail_threshold": fail_threshold,
            "positive_threshold": positive_threshold,
        }
        dual_signal.append(
            {
                "config": cfg,
                "val": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="val",
                ),
                "test": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="test",
                ),
            }
        )
    all_results["dual_signal"] = choose_best_config(dual_signal)

    risk_aware = []
    for max_step, fail_threshold, positive_threshold, negative_threshold in product(
        max_steps,
        fail_thresholds,
        positive_thresholds,
        negative_thresholds,
    ):
        cfg = {
            "policy_type": "risk_aware",
            "max_step": max_step,
            "fail_threshold": fail_threshold,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
        }
        risk_aware.append(
            {
                "config": cfg,
                "val": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="val",
                ),
                "test": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="test",
                ),
            }
        )
    all_results["risk_aware"] = choose_best_config(risk_aware)

    benchmarks = sorted({rows[0]["benchmark"] for rows in grouped_rows.values()})
    benchmark_aware = []
    benchmark_cfg_space = list(product(fail_thresholds, positive_thresholds, negative_thresholds))
    for max_step in max_steps:
        per_benchmark = {}
        for benchmark in benchmarks:
            best_local = None
            best_gain = None
            for fail_threshold, positive_threshold, negative_threshold in benchmark_cfg_space:
                cfg = {
                    "policy_type": "benchmark_aware_risk_aware",
                    "max_step": max_step,
                    "per_benchmark": {
                        benchmark: {
                            "fail_threshold": fail_threshold,
                            "positive_threshold": positive_threshold,
                            "negative_threshold": negative_threshold,
                        }
                    },
                }
                subset_grouped = {
                    key: value
                    for key, value in grouped_rows.items()
                    if str(value[0]["benchmark"]) == benchmark
                }
                sim_val = simulate_policy(
                    grouped_rows=subset_grouped,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="val",
                )
                gain = sim_val["policy_gain_over_small"]
                if best_gain is None or (gain is not None and gain > best_gain):
                    best_gain = gain
                    best_local = cfg["per_benchmark"][benchmark]
            per_benchmark[benchmark] = best_local
        cfg = {
            "policy_type": "benchmark_aware_risk_aware",
            "max_step": max_step,
            "per_benchmark": per_benchmark,
        }
        benchmark_aware.append(
            {
                "config": cfg,
                "val": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="val",
                ),
                "test": simulate_policy(
                    grouped_rows=grouped_rows,
                    oracle_index=oracle_index,
                    config=cfg,
                    split_name="test",
                ),
            }
        )
    all_results["benchmark_aware_risk_aware"] = choose_best_config(benchmark_aware)

    for block in all_results.values():
        block["test"]["by_benchmark"] = summarize_by_group(block["test"], "benchmark")
        block["test"]["by_small_family"] = summarize_by_group(block["test"], "small_family")

    summary = {
        "notes": [
            "Advanced routing upgrades the failure-only policy with positive-takeover and negative-risk predictors trained on the prefix predictor dataset.",
            "Thresholds are chosen on validation tasks only, then reported on held-out test tasks.",
            "Benchmark-aware policy picks a separate risk-aware threshold triple for each benchmark on validation.",
        ],
        "policies": all_results,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Advanced Routing Policies",
        "",
        "這份分析把原本只看 `P(small failure)` 的 routing baseline，升級成三種更強的 policy：",
        "",
        "- `dual_signal`: failure + positive takeover",
        "- `risk_aware`: failure + positive takeover + negative risk",
        "- `benchmark_aware_risk_aware`: 為每個 benchmark 分開選 risk-aware threshold",
        "",
        "| policy | max_step | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, block in all_results.items():
        cfg = block["config"]
        test = block["test"]
        lines.append(
            "| {name} | {max_step} | {small:.3f} | {policy:.3f} | {gain:.3f} | {oracle:.3f} | {gap:.3f} | {route:.3f} |".format(
                name=name,
                max_step=cfg["max_step"],
                small=test["baseline_small_accuracy"],
                policy=test["policy_accuracy"],
                gain=test["policy_gain_over_small"],
                oracle=test["oracle_accuracy_same_budget"],
                gap=test["gap_to_oracle"],
                route=test["route_rate"],
            )
        )

    for name, block in all_results.items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- validation-selected config: `{json.dumps(block['config'], ensure_ascii=False)}`",
                f"- test small baseline accuracy: `{block['test']['baseline_small_accuracy']:.3f}`",
                f"- test policy accuracy: `{block['test']['policy_accuracy']:.3f}`",
                f"- test gain over small: `{block['test']['policy_gain_over_small']:.3f}`",
                f"- test oracle accuracy: `{block['test']['oracle_accuracy_same_budget']:.3f}`",
                f"- test route rate: `{block['test']['route_rate']:.3f}`",
                "",
                "### By Benchmark",
                "",
            ]
        )
        for benchmark, stats in block["test"]["by_benchmark"].items():
            lines.extend(
                [
                    f"#### {benchmark}",
                    "",
                    f"- tasks: `{stats['tasks']}`",
                    f"- small baseline accuracy: `{stats['baseline_small_accuracy']:.3f}`",
                    f"- policy accuracy: `{stats['policy_accuracy']:.3f}`",
                    f"- gain over small: `{stats['policy_gain_over_small']:.3f}`",
                    f"- route rate: `{stats['route_rate']:.3f}`",
                    "",
                ]
            )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
