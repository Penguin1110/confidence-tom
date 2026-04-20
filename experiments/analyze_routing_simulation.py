from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
DATASET_CSV = EARLY_DIR / "early_decision_rows.csv"
OUTPUT_JSON = EARLY_DIR / "routing_simulation.json"
OUTPUT_MD = ROOT / "docs" / "routing_simulation.md"


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


def load_early_rows() -> list[dict[str, str]]:
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
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
                    "small_total_tokens": step.get("small_continue_cost", {}).get("total_tokens"),
                }
            index[(run_name, str(task["task_id"]))] = {
                "benchmark": benchmark,
                "small_family": small_family,
                "large_family": large_family,
                "full_trace_correct": int(bool(task["full_trace_correct"])),
                "steps": steps,
            }
    return index


def build_matrix(rows: list[dict[str, str]]):
    train_rows = [r for r in rows if not stable_test_split(r["task_id"])]
    test_rows = [r for r in rows if stable_test_split(r["task_id"])]

    small_families = sorted({r["small_family"] for r in rows})
    large_families = sorted({r["large_family"] for r in rows})

    feature_names = list(STATE_FEATURES)
    feature_names += [f"small_family={name}" for name in small_families]
    feature_names += [f"large_family={name}" for name in large_families]

    def encode(selected_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(selected_rows), len(feature_names) + 1), dtype=np.float64)
        y = np.zeros(len(selected_rows), dtype=np.float64)
        for i, row in enumerate(selected_rows):
            x[i, 0] = 1.0
            offset = 1
            for feat in STATE_FEATURES:
                x[i, offset] = float(row[feat])
                offset += 1
            for name in small_families:
                x[i, offset] = 1.0 if row["small_family"] == name else 0.0
                offset += 1
            for name in large_families:
                x[i, offset] = 1.0 if row["large_family"] == name else 0.0
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
        "feature_names": ["bias", *feature_names],
    }


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def simulate_policy(
    *,
    task_rows: dict[tuple[str, str], list[dict[str, object]]],
    oracle_index: dict[tuple[str, str], dict[str, object]],
    max_step: int,
    failure_threshold: float,
) -> dict[str, object]:
    records = []
    routed = 0
    wins = 0
    total = 0
    trigger_steps: list[int] = []
    route_token_costs: list[float] = []
    oracle_routed = 0
    oracle_wins = 0

    for key, rows in task_rows.items():
        task_meta = oracle_index[key]
        baseline = int(task_meta["full_trace_correct"])
        selected = None
        for row in sorted(rows, key=lambda r: int(float(r["step_index"]))):
            step = int(float(row["step_index"]))
            if step > max_step:
                continue
            failure_prob = 1.0 - float(row["pred_success"])
            if failure_prob >= failure_threshold:
                selected = step
                break

        if selected is None:
            final_correct = baseline
            route_used = 0
            trigger_step = None
            large_correct = None
        else:
            step_meta = task_meta["steps"][selected]
            final_correct = int(step_meta["large_takeover_correct"])
            route_used = 1
            trigger_step = selected
            large_correct = int(step_meta["large_takeover_correct"])
            routed += 1
            trigger_steps.append(selected)
            if step_meta["large_total_tokens"] is not None:
                route_token_costs.append(float(step_meta["large_total_tokens"]))
        total += 1
        wins += final_correct

        # Oracle comparator under same max_step: route iff some positive step exists within max_step.
        oracle_choice = None
        for step in sorted(task_meta["steps"]):
            if step > max_step:
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

        records.append(
            {
                "run_name": key[0],
                "task_id": key[1],
                "benchmark": task_meta["benchmark"],
                "small_family": task_meta["small_family"],
                "large_family": task_meta["large_family"],
                "baseline_small_correct": baseline,
                "route_used": route_used,
                "trigger_step": trigger_step,
                "policy_final_correct": final_correct,
                "selected_large_correct": large_correct,
            }
        )

    baseline_accuracy = avg([float(oracle_index[k]["full_trace_correct"]) for k in task_rows])
    policy_accuracy = wins / total if total else None
    oracle_accuracy = oracle_wins / total if total else None
    return {
        "max_step": max_step,
        "failure_threshold": failure_threshold,
        "tasks": total,
        "baseline_small_accuracy": baseline_accuracy,
        "policy_accuracy": policy_accuracy,
        "oracle_accuracy_same_budget": oracle_accuracy,
        "policy_gain_over_small": None
        if policy_accuracy is None or baseline_accuracy is None
        else policy_accuracy - baseline_accuracy,
        "gap_to_oracle": None
        if oracle_accuracy is None or policy_accuracy is None
        else oracle_accuracy - policy_accuracy,
        "route_rate": routed / total if total else None,
        "oracle_route_rate": oracle_routed / total if total else None,
        "mean_trigger_step": avg([float(x) for x in trigger_steps]),
        "mean_large_total_tokens_when_routed": avg(route_token_costs),
        "records": records,
    }


def summarize_by_group(sim: dict[str, object], group_key: str) -> dict[str, object]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in sim["records"]:
        buckets.setdefault(str(row[group_key]), []).append(row)
    out = {}
    for name, rows in buckets.items():
        total = len(rows)
        small_acc = avg([float(r["baseline_small_correct"]) for r in rows])
        policy_acc = avg([float(r["policy_final_correct"]) for r in rows])
        out[name] = {
            "tasks": total,
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


def main() -> None:
    early_rows = load_early_rows()
    oracle_index = load_oracle_index()
    split = build_matrix(early_rows)
    weights = fit_logistic_regression(split["x_train"], split["y_train"])
    test_probs = sigmoid(split["x_test"] @ weights)

    test_task_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row, prob in zip(split["test_rows"], test_probs, strict=True):
        key = (row["run_name"], row["task_id"])
        enriched = dict(row)
        enriched["pred_success"] = float(prob)
        test_task_rows.setdefault(key, []).append(enriched)

    configs = []
    for max_step in [1, 2, 3]:
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            sim = simulate_policy(
                task_rows=test_task_rows,
                oracle_index=oracle_index,
                max_step=max_step,
                failure_threshold=threshold,
            )
            sim["by_benchmark"] = summarize_by_group(sim, "benchmark")
            sim["by_small_family"] = summarize_by_group(sim, "small_family")
            configs.append(sim)

    best_by_gain = sorted(
        configs,
        key=lambda x: (
            x["policy_gain_over_small"],
            -x["gap_to_oracle"] if x["gap_to_oracle"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    output = {
        "notes": [
            "Policy uses Early Decision state_plus_family predictor trained on training tasks only.",
            "For each test task, route at the earliest prefix with failure probability >= threshold and step <= max_step.",
            "If routed, final correctness is the observed large takeover correctness at that prefix; otherwise the task keeps the small full-trace outcome.",
            "Oracle comparator uses the same max_step budget and routes only when a positive takeover exists within budget.",
        ],
        "test_tasks": len(test_task_rows),
        "configs": configs,
        "top_configs_by_gain": [
            {k: v for k, v in cfg.items() if k != "records"} for cfg in best_by_gain[:5]
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Routing Simulation Baseline",
        "",
        "這份模擬把 Early Decision predictor 直接接到 prefix-level routing policy：當 `P(small failure)` 超過門檻時，就在最早可用 prefix 交給 large takeover。",
        "",
        f"- test tasks: `{output['test_tasks']}`",
        "",
        "## Top Configs By Gain",
        "",
        "| max_step | fail threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate | mean trigger step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for cfg in output["top_configs_by_gain"]:
        lines.append(
            "| {max_step} | {failure_threshold:.1f} | {baseline_small_accuracy:.3f} | {policy_accuracy:.3f} | {policy_gain_over_small:.3f} | {oracle_accuracy_same_budget:.3f} | {gap_to_oracle:.3f} | {route_rate:.3f} | {mean_trigger_step_val} |".format(
                max_step=cfg["max_step"],
                failure_threshold=cfg["failure_threshold"],
                baseline_small_accuracy=cfg["baseline_small_accuracy"],
                policy_accuracy=cfg["policy_accuracy"],
                policy_gain_over_small=cfg["policy_gain_over_small"],
                oracle_accuracy_same_budget=cfg["oracle_accuracy_same_budget"],
                gap_to_oracle=cfg["gap_to_oracle"],
                route_rate=cfg["route_rate"],
                mean_trigger_step_val=f"{cfg['mean_trigger_step']:.3f}"
                if cfg["mean_trigger_step"] is not None
                else "-",
            )
        )

    best = output["top_configs_by_gain"][0]
    lines.extend(
        [
            "",
            "## Best Config Breakdown",
            "",
            f"- best config: `max_step={best['max_step']}`, `failure_threshold={best['failure_threshold']}`",
            f"- small baseline accuracy: `{best['baseline_small_accuracy']:.3f}`",
            f"- policy accuracy: `{best['policy_accuracy']:.3f}`",
            f"- gain over small: `{best['policy_gain_over_small']:.3f}`",
            f"- oracle accuracy under same budget: `{best['oracle_accuracy_same_budget']:.3f}`",
            f"- route rate: `{best['route_rate']:.3f}`",
            "",
            "### By Benchmark",
            "",
        ]
    )

    for name, stats in best["by_benchmark"].items():
        lines.extend(
            [
                f"#### {name}",
                "",
                f"- tasks: `{stats['tasks']}`",
                f"- small baseline accuracy: `{stats['baseline_small_accuracy']:.3f}`",
                f"- policy accuracy: `{stats['policy_accuracy']:.3f}`",
                f"- policy gain over small: `{stats['policy_gain_over_small']:.3f}`",
                f"- route rate: `{stats['route_rate']:.3f}`",
                "",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
