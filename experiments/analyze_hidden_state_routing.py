from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
HIDDEN_DIR = RESULTS_DIR / "_early_decision_hidden_states_v1"
DATASET_CSV = EARLY_DIR / "early_decision_rows.csv"
OUTPUT_JSON = EARLY_DIR / "hidden_state_routing.json"
OUTPUT_MD = ROOT / "docs" / "hidden_state_routing.md"


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
    ("mistral_to_openai_50", "olympiadbench", "mistral", "openai"),
    ("mistral_to_anthropic_50", "olympiadbench", "mistral", "anthropic"),
    ("livebench_qwen_to_openai_30", "livebench_reasoning", "qwen", "openai"),
    ("livebench_qwen_to_anthropic_30", "livebench_reasoning", "qwen", "anthropic"),
    ("livebench_mistral_to_openai_30", "livebench_reasoning", "mistral", "openai"),
    ("livebench_mistral_to_anthropic_30", "livebench_reasoning", "mistral", "anthropic"),
]

BEST_LAYERS = {
    "qwen": 9,
    "mistral": 9,
}


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


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


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


def load_text_rows() -> list[dict[str, str]]:
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["small_family"] in {"qwen", "mistral"}]


def encode_text_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    small_families = sorted({row["small_family"] for row in rows})
    large_families = sorted({row["large_family"] for row in rows})

    feature_names = list(STATE_FEATURES)
    feature_names += [f"small_family={name}" for name in small_families]
    feature_names += [f"large_family={name}" for name in large_families]

    x = np.zeros((len(rows), len(feature_names) + 1), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
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
    return x, y, ["bias", *feature_names]


def load_hidden_rows_and_features() -> dict[str, tuple[list[dict[str, object]], np.ndarray]]:
    out: dict[str, tuple[list[dict[str, object]], np.ndarray]] = {}
    family_dirs = {
        "qwen": HIDDEN_DIR,
        "mistral": HIDDEN_DIR / "mistral",
    }
    for family, path in family_dirs.items():
        with (path / "rows.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        arr = np.load(path / "hidden_states.npz")
        mean_pooled = arr["mean_pooled"]
        layer = BEST_LAYERS[family]
        if layer >= mean_pooled.shape[1]:
            raise ValueError(f"Layer {layer} out of bounds for {family}")
        out[family] = (rows, mean_pooled[:, layer, :].astype(np.float64))
    return out


def normalize_train_test(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x_out = x.copy()
    means = np.mean(x_out[train_mask], axis=0)
    stds = np.std(x_out[train_mask], axis=0)
    stds[stds == 0.0] = 1.0
    x_out = (x_out - means) / stds
    bias = np.ones((x_out.shape[0], 1), dtype=np.float64)
    return np.concatenate([bias, x_out], axis=1)


def build_pred_rows(
    rows: list[dict[str, object]],
    probs_success: np.ndarray,
) -> list[dict[str, object]]:
    out = []
    for row, prob in zip(rows, probs_success, strict=True):
        out.append(
            {
                "run_name": row["run_name"],
                "task_id": str(row["task_id"]),
                "benchmark": row["benchmark"],
                "small_family": row["small_family"],
                "large_family": row["large_family"],
                "step_index": int(float(row["step_index"])),
                "pred_success": float(prob),
            }
        )
    return out


def group_test_rows(
    pred_rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in pred_rows:
        key = (str(row["run_name"]), str(row["task_id"]))
        if not stable_test_split(key[1]):
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


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
    oracle_routed = 0
    oracle_wins = 0

    for key, rows in task_rows.items():
        task_meta = oracle_index[key]
        baseline = int(task_meta["full_trace_correct"])
        selected = None
        for row in sorted(rows, key=lambda r: int(r["step_index"])):
            step = int(row["step_index"])
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
        else:
            step_meta = task_meta["steps"][selected]
            final_correct = int(step_meta["large_takeover_correct"])
            route_used = 1
            trigger_step = selected
            routed += 1
            trigger_steps.append(selected)

        total += 1
        wins += final_correct

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
        "policy_gain_over_small": None
        if policy_accuracy is None or baseline_accuracy is None
        else policy_accuracy - baseline_accuracy,
        "oracle_accuracy_same_budget": oracle_accuracy,
        "gap_to_oracle": None
        if oracle_accuracy is None or policy_accuracy is None
        else oracle_accuracy - policy_accuracy,
        "route_rate": routed / total if total else None,
        "mean_trigger_step": avg([float(x) for x in trigger_steps]),
        "oracle_route_rate": oracle_routed / total if total else None,
        "task_records": records,
    }


def summarize_by_benchmark(task_records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in task_records:
        grouped.setdefault(str(row["benchmark"]), []).append(row)
    out: dict[str, dict[str, float]] = {}
    for benchmark, rows in grouped.items():
        baseline = avg([float(r["baseline_small_correct"]) for r in rows])
        policy = avg([float(r["policy_final_correct"]) for r in rows])
        route = avg([float(r["route_used"]) for r in rows])
        out[benchmark] = {
            "tasks": len(rows),
            "baseline_small_accuracy": baseline,
            "policy_accuracy": policy,
            "policy_gain_over_small": None
            if baseline is None or policy is None
            else policy - baseline,
            "route_rate": route,
        }
    return out


def train_text_predictor(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    x, y, _ = encode_text_rows(rows)
    train_mask = np.array([not stable_test_split(r["task_id"]) for r in rows], dtype=bool)
    x = normalize_train_test(x[:, 1:], train_mask)
    weights = fit_logistic_regression(x[train_mask], y[train_mask])
    probs = sigmoid(x @ weights)
    return build_pred_rows(rows, probs)


def train_hidden_predictor() -> list[dict[str, object]]:
    pred_rows: list[dict[str, object]] = []
    for family, (rows, x) in load_hidden_rows_and_features().items():
        y = np.array([float(r["small_full_trace_success"]) for r in rows], dtype=np.float64)
        train_mask = np.array([not stable_test_split(str(r["task_id"])) for r in rows], dtype=bool)
        x = normalize_train_test(x, train_mask)
        weights = fit_logistic_regression(x[train_mask], y[train_mask], lr=0.03, steps=3000)
        probs = sigmoid(x @ weights)
        pred_rows.extend(build_pred_rows(rows, probs))
    return pred_rows


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Hidden-State Routing Baseline",
        "",
        "這份分析把 `Qwen + Mistral` 的 prefix hidden states 直接接到 routing simulator，做最小可跑版比較：",
        "",
        "- `text_state_plus_family`: 現有 text/state-feature predictor",
        "- `hidden_state_mean_pool_layer9`: 用各 family 的 layer-9 mean-pooled hidden state 預測 `P(small failure)`",
        "",
        "兩個版本都只在 `qwen + mistral` 子集上訓練與評估，並使用相同的 stable task split、相同的 routing simulator、相同的 oracle comparator。",
        "",
    ]
    for model_name, block in summary["models"].items():
        best = block["best"]
        lines.extend(
            [
                f"## {model_name}",
                "",
                "| max_step | fail threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cfg in block["results"]:
            lines.append(
                "| {max_step} | {failure_threshold:.1f} | {baseline_small_accuracy:.3f} | {policy_accuracy:.3f} | {policy_gain_over_small:.3f} | {oracle_accuracy_same_budget:.3f} | {gap_to_oracle:.3f} | {route_rate:.3f} |".format(
                    **cfg
                )
            )
        lines.extend(
            [
                "",
                f"- best config: `max_step={best['max_step']}`, `failure_threshold={best['failure_threshold']}`",
                f"- small baseline accuracy: `{best['baseline_small_accuracy']:.3f}`",
                f"- policy accuracy: `{best['policy_accuracy']:.3f}`",
                f"- gain over small: `{best['policy_gain_over_small']:.3f}`",
                f"- oracle accuracy under same budget: `{best['oracle_accuracy_same_budget']:.3f}`",
                f"- gap to oracle: `{best['gap_to_oracle']:.3f}`",
                f"- route rate: `{best['route_rate']:.3f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    oracle_index = load_oracle_index()
    text_pred_rows = train_text_predictor(load_text_rows())
    hidden_pred_rows = train_hidden_predictor()

    models = {
        "text_state_plus_family": group_test_rows(text_pred_rows),
        "hidden_state_mean_pool_layer9": group_test_rows(hidden_pred_rows),
    }

    summary: dict[str, object] = {
        "notes": [
            "Both policies predict small full-trace success at prefix level and route when predicted failure exceeds threshold.",
            "Evaluation is restricted to qwen + mistral tasks because hidden states are currently available for those families only.",
            "Hidden-state features use mean-pooled layer 9 for both qwen and mistral.",
        ],
        "models": {},
    }

    for model_name, grouped_rows in models.items():
        results = []
        for max_step in [1, 2, 3]:
            for threshold in [0.5, 0.6, 0.7]:
                sim = simulate_policy(
                    task_rows=grouped_rows,
                    oracle_index=oracle_index,
                    max_step=max_step,
                    failure_threshold=threshold,
                )
                sim["benchmark_breakdown"] = summarize_by_benchmark(sim["task_records"])
                results.append(sim)
        best = max(results, key=lambda r: (r["policy_accuracy"], -abs(r["route_rate"] - 0.5)))
        summary["models"][model_name] = {
            "results": results,
            "best": {k: v for k, v in best.items() if k != "task_records"},
        }

    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    for model_name, block in summary["models"].items():
        best = block["best"]
        print(
            f"{model_name}: max_step={best['max_step']} fail_threshold={best['failure_threshold']:.1f} "
            f"policy_acc={best['policy_accuracy']:.3f} gain={best['policy_gain_over_small']:.3f} "
            f"oracle={best['oracle_accuracy_same_budget']:.3f} route_rate={best['route_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
