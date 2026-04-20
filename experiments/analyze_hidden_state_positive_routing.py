from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_hidden_state_routing import (
    HIDDEN_DIR,
    RESULTS_DIR,
    RUN_SPECS,
    STATE_FEATURES,
    avg,
    find_result_json,
    fit_logistic_regression,
    normalize_train_test,
    sigmoid,
    stable_test_split,
    summarize_by_benchmark,
)

EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
PREFIX_CSV = RESULTS_DIR / "_prefix_predictor_v1" / "prefix_predictor_rows.csv"
OUTPUT_JSON = EARLY_DIR / "hidden_state_positive_routing.json"
OUTPUT_MD = ROOT / "docs" / "hidden_state_positive_routing.md"

BEST_LAYERS = {
    "qwen": 19,
    "mistral": 9,
}


def load_oracle_index() -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for run_name, benchmark, small_family, large_family in RUN_SPECS:
        if small_family not in {"qwen", "mistral"}:
            continue
        data = json.loads(find_result_json(run_name).read_text(encoding="utf-8"))
        for task in data:
            steps = {}
            for step in task["prefix_oracle_steps"]:
                steps[int(step["step_index"])] = {
                    "large_takeover_correct": int(bool(step["large_takeover_correct"])),
                    "small_continue_correct": int(bool(step["small_continue_correct"])),
                    "delta_correctness": float(step["delta_correctness"]),
                }
            index[(run_name, str(task["task_id"]))] = {
                "benchmark": benchmark,
                "small_family": small_family,
                "large_family": large_family,
                "full_trace_correct": int(bool(task["full_trace_correct"])),
                "steps": steps,
            }
    return index


def load_prefix_rows() -> list[dict[str, str]]:
    with PREFIX_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["small_family"] in {"qwen", "mistral"}]


def encode_text_positive(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
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
        y[i] = float(row["delta_positive"])
    return x, y


def load_hidden_rows_and_labels() -> dict[
    str, tuple[list[dict[str, object]], np.ndarray, np.ndarray]
]:
    prefix_rows = load_prefix_rows()
    label_index = {
        (r["run_name"], r["task_id"], r["prefix_id"]): int(r["delta_positive"]) for r in prefix_rows
    }
    out: dict[str, tuple[list[dict[str, object]], np.ndarray, np.ndarray]] = {}
    family_dirs = {
        "qwen": HIDDEN_DIR,
        "mistral": HIDDEN_DIR / "mistral",
    }
    for family, path in family_dirs.items():
        with (path / "rows.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        arr = np.load(path / "hidden_states.npz")
        x = arr["mean_pooled"][:, BEST_LAYERS[family], :].astype(np.float64)
        y = np.array(
            [label_index[(r["run_name"], str(r["task_id"]), r["prefix_id"])] for r in rows],
            dtype=np.float64,
        )
        out[family] = (rows, x, y)
    return out


def train_text_positive_predictor(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    x, y = encode_text_positive(rows)
    train_mask = np.array([not stable_test_split(r["task_id"]) for r in rows], dtype=bool)
    x = normalize_train_test(x[:, 1:], train_mask)
    weights = fit_logistic_regression(x[train_mask], y[train_mask])
    probs = sigmoid(x @ weights)
    return [
        {
            "run_name": row["run_name"],
            "task_id": str(row["task_id"]),
            "benchmark": row["benchmark"],
            "small_family": row["small_family"],
            "large_family": row["large_family"],
            "step_index": int(float(row["step_index"])),
            "pred_positive": float(prob),
        }
        for row, prob in zip(rows, probs, strict=True)
    ]


def train_hidden_positive_predictor() -> list[dict[str, object]]:
    pred_rows: list[dict[str, object]] = []
    for family, (rows, x, y) in load_hidden_rows_and_labels().items():
        train_mask = np.array([not stable_test_split(str(r["task_id"])) for r in rows], dtype=bool)
        x = normalize_train_test(x, train_mask)
        weights = fit_logistic_regression(x[train_mask], y[train_mask], lr=0.03, steps=3000)
        probs = sigmoid(x @ weights)
        pred_rows.extend(
            [
                {
                    "run_name": row["run_name"],
                    "task_id": str(row["task_id"]),
                    "benchmark": row["benchmark"],
                    "small_family": row["small_family"],
                    "large_family": row["large_family"],
                    "step_index": int(float(row["step_index"])),
                    "pred_positive": float(prob),
                }
                for row, prob in zip(rows, probs, strict=True)
            ]
        )
    return pred_rows


def group_positive_rows(
    pred_rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in pred_rows:
        key = (str(row["run_name"]), str(row["task_id"]))
        if not stable_test_split(key[1]):
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def simulate_positive_policy(
    *,
    task_rows: dict[tuple[str, str], list[dict[str, object]]],
    oracle_index: dict[tuple[str, str], dict[str, object]],
    max_step: int,
    positive_threshold: float,
) -> dict[str, object]:
    records = []
    routed = 0
    wins = 0
    total = 0
    oracle_routed = 0
    oracle_wins = 0
    trigger_steps: list[int] = []

    for key, rows in task_rows.items():
        task_meta = oracle_index[key]
        baseline = int(task_meta["full_trace_correct"])
        selected = None
        for row in sorted(rows, key=lambda r: int(r["step_index"])):
            step = int(row["step_index"])
            if step > max_step:
                continue
            if float(row["pred_positive"]) >= positive_threshold:
                selected = step
                break
        if selected is None:
            final_correct = baseline
            route_used = 0
            trigger_step = None
        else:
            final_correct = int(task_meta["steps"][selected]["large_takeover_correct"])
            route_used = 1
            trigger_step = selected
            routed += 1
            trigger_steps.append(selected)
        oracle_choice = None
        for step in sorted(task_meta["steps"]):
            if step > max_step:
                continue
            if task_meta["steps"][step]["delta_correctness"] > 0:
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
                "trigger_step": trigger_step,
            }
        )

    baseline_accuracy = avg([float(r["baseline_small_correct"]) for r in records])
    policy_accuracy = avg([float(r["policy_final_correct"]) for r in records])
    oracle_accuracy = oracle_wins / total if total else None
    return {
        "max_step": max_step,
        "positive_threshold": positive_threshold,
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
        "task_records": records,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Hidden-State Positive-Only Routing",
        "",
        "這份分析直接用 `positive takeover` 當 routing label，比較：",
        "",
        "- `text_positive_only`: 用 prefix predictor state features 預測 `delta_positive`",
        "- `hidden_state_positive_only`: 用 family-specific hidden state 預測 `delta_positive`",
        "",
        "兩個版本都只在 `qwen + mistral` 子集上訓練與評估，並使用相同的 test task split 與 oracle comparator。",
        "",
    ]
    for name, block in summary["models"].items():
        best = block["best"]
        lines.extend(
            [
                f"## {name}",
                "",
                "| max_step | positive threshold | small acc | policy acc | gain | oracle acc | gap to oracle | route rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cfg in block["results"]:
            lines.append(
                "| {max_step} | {positive_threshold:.1f} | {baseline_small_accuracy:.3f} | {policy_accuracy:.3f} | {policy_gain_over_small:.3f} | {oracle_accuracy_same_budget:.3f} | {gap_to_oracle:.3f} | {route_rate:.3f} |".format(
                    **cfg
                )
            )
        lines.extend(
            [
                "",
                f"- best config: `max_step={best['max_step']}`, `positive_threshold={best['positive_threshold']}`",
                f"- policy accuracy: `{best['policy_accuracy']:.3f}`",
                f"- gain over small: `{best['policy_gain_over_small']:.3f}`",
                f"- oracle accuracy: `{best['oracle_accuracy_same_budget']:.3f}`",
                f"- gap to oracle: `{best['gap_to_oracle']:.3f}`",
                f"- route rate: `{best['route_rate']:.3f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    oracle_index = load_oracle_index()
    text_rows = load_prefix_rows()
    text_pred_rows = train_text_positive_predictor(text_rows)
    hidden_pred_rows = train_hidden_positive_predictor()

    models = {
        "text_positive_only": group_positive_rows(text_pred_rows),
        "hidden_state_positive_only": group_positive_rows(hidden_pred_rows),
    }
    summary: dict[str, object] = {"models": {}}
    for model_name, grouped_rows in models.items():
        results = []
        for max_step in [1, 2, 3]:
            for threshold in [0.2, 0.3, 0.4, 0.5]:
                sim = simulate_positive_policy(
                    task_rows=grouped_rows,
                    oracle_index=oracle_index,
                    max_step=max_step,
                    positive_threshold=threshold,
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
            f"{model_name}: max_step={best['max_step']} positive_threshold={best['positive_threshold']:.1f} "
            f"policy_acc={best['policy_accuracy']:.3f} gain={best['policy_gain_over_small']:.3f} "
            f"oracle={best['oracle_accuracy_same_budget']:.3f} route_rate={best['route_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
