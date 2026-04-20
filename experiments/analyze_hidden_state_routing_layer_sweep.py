from __future__ import annotations

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
    build_pred_rows,
    find_result_json,
    fit_logistic_regression,
    group_test_rows,
    load_text_rows,
    normalize_train_test,
    sigmoid,
    simulate_policy,
    stable_test_split,
    summarize_by_benchmark,
    train_text_predictor,
)

EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
OUTPUT_JSON = EARLY_DIR / "hidden_state_routing_layer_sweep.json"
OUTPUT_MD = ROOT / "docs" / "hidden_state_routing_layer_sweep.md"

CANDIDATE_LAYERS = {
    "qwen": [0, 9, 19, 29, 39],
    "mistral": [0, 9, 19, 29, 35],
}


def load_oracle_index() -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for run_name, benchmark, small_family, large_family in RUN_SPECS:
        data = json.loads(find_result_json(run_name).read_text(encoding="utf-8"))
        for task in data:
            steps = {}
            for step in task["prefix_oracle_steps"]:
                steps[int(step["step_index"])] = {
                    "large_takeover_correct": int(bool(step["large_takeover_correct"])),
                }
            index[(run_name, str(task["task_id"]))] = {
                "benchmark": benchmark,
                "small_family": small_family,
                "large_family": large_family,
                "full_trace_correct": int(bool(task["full_trace_correct"])),
                "steps": steps,
            }
    return index


def load_hidden_family_data() -> dict[str, tuple[list[dict[str, object]], np.ndarray]]:
    out: dict[str, tuple[list[dict[str, object]], np.ndarray]] = {}
    family_dirs = {
        "qwen": HIDDEN_DIR,
        "mistral": HIDDEN_DIR / "mistral",
    }
    for family, path in family_dirs.items():
        with (path / "rows.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        arr = np.load(path / "hidden_states.npz")
        out[family] = (rows, arr["mean_pooled"].astype(np.float64))
    return out


def train_hidden_predictor_for_layers(
    family_data: dict[str, tuple[list[dict[str, object]], np.ndarray]],
    family_layers: dict[str, int],
) -> list[dict[str, object]]:
    pred_rows: list[dict[str, object]] = []
    for family, layer in family_layers.items():
        rows, all_layers = family_data[family]
        x = all_layers[:, layer, :]
        y = np.array([float(r["small_full_trace_success"]) for r in rows], dtype=np.float64)
        train_mask = np.array([not stable_test_split(str(r["task_id"])) for r in rows], dtype=bool)
        x = normalize_train_test(x, train_mask)
        weights = fit_logistic_regression(x[train_mask], y[train_mask], lr=0.03, steps=3000)
        probs = sigmoid(x @ weights)
        pred_rows.extend(build_pred_rows(rows, probs))
    return pred_rows


def render_markdown(summary: dict[str, object]) -> str:
    best = summary["best_hidden_state"]
    lines = [
        "# Hidden-State Routing Layer Sweep",
        "",
        "這份分析在 `qwen + mistral` 子集上，對 hidden-state routing 做 family-specific layer sweep。",
        "",
        "- hidden state feature: `mean_pooled`",
        "- qwen candidate layers: `0, 9, 19, 29, 39`",
        "- mistral candidate layers: `0, 9, 19, 29, 35`",
        "- routing search: `max_step in {1,2,3}`, `failure_threshold in {0.5, 0.6, 0.7}`",
        "",
        "## Text Baseline",
        "",
        f"- best config: `max_step={summary['text_baseline']['max_step']}`, `failure_threshold={summary['text_baseline']['failure_threshold']}`",
        f"- policy accuracy: `{summary['text_baseline']['policy_accuracy']:.3f}`",
        f"- gain over small: `{summary['text_baseline']['policy_gain_over_small']:.3f}`",
        f"- route rate: `{summary['text_baseline']['route_rate']:.3f}`",
        "",
        "## Best Hidden-State Policy",
        "",
        f"- qwen layer: `{best['layers']['qwen']}`",
        f"- mistral layer: `{best['layers']['mistral']}`",
        f"- best config: `max_step={best['max_step']}`, `failure_threshold={best['failure_threshold']}`",
        f"- policy accuracy: `{best['policy_accuracy']:.3f}`",
        f"- gain over small: `{best['policy_gain_over_small']:.3f}`",
        f"- route rate: `{best['route_rate']:.3f}`",
        f"- oracle accuracy under same budget: `{best['oracle_accuracy_same_budget']:.3f}`",
        f"- gap to oracle: `{best['gap_to_oracle']:.3f}`",
        "",
        "## Top Hidden-State Combinations",
        "",
        "| qwen layer | mistral layer | max_step | fail threshold | policy acc | gain | oracle acc | gap | route rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["top_hidden_state_results"]:
        lines.append(
            "| {qwen} | {mistral} | {max_step} | {failure_threshold:.1f} | {policy_accuracy:.3f} | {policy_gain_over_small:.3f} | {oracle_accuracy_same_budget:.3f} | {gap_to_oracle:.3f} | {route_rate:.3f} |".format(
                qwen=row["layers"]["qwen"],
                mistral=row["layers"]["mistral"],
                **row,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    oracle_index = load_oracle_index()
    family_data = load_hidden_family_data()

    text_pred_rows = train_text_predictor(load_text_rows())
    text_task_rows = group_test_rows(text_pred_rows)
    text_results = []
    for max_step in [1, 2, 3]:
        for threshold in [0.5, 0.6, 0.7]:
            sim = simulate_policy(
                task_rows=text_task_rows,
                oracle_index=oracle_index,
                max_step=max_step,
                failure_threshold=threshold,
            )
            sim["benchmark_breakdown"] = summarize_by_benchmark(sim["task_records"])
            text_results.append(sim)
    text_best = max(text_results, key=lambda r: (r["policy_accuracy"], -abs(r["route_rate"] - 0.5)))

    hidden_results = []
    for q_layer in CANDIDATE_LAYERS["qwen"]:
        for m_layer in CANDIDATE_LAYERS["mistral"]:
            pred_rows = train_hidden_predictor_for_layers(
                family_data,
                {"qwen": q_layer, "mistral": m_layer},
            )
            task_rows = group_test_rows(pred_rows)
            for max_step in [1, 2, 3]:
                for threshold in [0.5, 0.6, 0.7]:
                    sim = simulate_policy(
                        task_rows=task_rows,
                        oracle_index=oracle_index,
                        max_step=max_step,
                        failure_threshold=threshold,
                    )
                    sim["benchmark_breakdown"] = summarize_by_benchmark(sim["task_records"])
                    sim["layers"] = {"qwen": q_layer, "mistral": m_layer}
                    hidden_results.append(sim)

    hidden_best = max(
        hidden_results, key=lambda r: (r["policy_accuracy"], -abs(r["route_rate"] - 0.5))
    )
    top_hidden = sorted(
        hidden_results,
        key=lambda r: (
            r["policy_accuracy"],
            r["policy_gain_over_small"],
            -r["gap_to_oracle"],
            -abs(r["route_rate"] - 0.5),
        ),
        reverse=True,
    )[:10]

    summary = {
        "notes": [
            "Layer sweep uses family-specific hidden-state predictors because qwen and mistral live in different hidden spaces.",
            "All routing results are evaluated on the same qwen+mistral task subset and use the same oracle comparator.",
        ],
        "text_baseline": {k: v for k, v in text_best.items() if k != "task_records"},
        "best_hidden_state": {k: v for k, v in hidden_best.items() if k != "task_records"},
        "top_hidden_state_results": [
            {k: v for k, v in row.items() if k != "task_records"} for row in top_hidden
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        "text_baseline:",
        {
            k: text_best[k]
            for k in [
                "max_step",
                "failure_threshold",
                "policy_accuracy",
                "policy_gain_over_small",
                "route_rate",
            ]
        },
    )
    print(
        "best_hidden_state:",
        {
            "layers": hidden_best["layers"],
            **{
                k: hidden_best[k]
                for k in [
                    "max_step",
                    "failure_threshold",
                    "policy_accuracy",
                    "policy_gain_over_small",
                    "route_rate",
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
