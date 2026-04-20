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
    build_pred_rows,
    fit_logistic_regression,
    group_test_rows,
    load_oracle_index,
    load_text_rows,
    normalize_train_test,
    sigmoid,
    simulate_policy,
    stable_test_split,
    summarize_by_benchmark,
    train_text_predictor,
)

EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
OUTPUT_JSON = EARLY_DIR / "hidden_state_pooling_compare.json"
OUTPUT_MD = ROOT / "docs" / "hidden_state_pooling_compare.md"

BEST_LAYERS = {
    "qwen": 19,
    "mistral": 9,
}


def load_hidden_rows_and_features(
    pooling: str,
) -> dict[str, tuple[list[dict[str, object]], np.ndarray]]:
    out: dict[str, tuple[list[dict[str, object]], np.ndarray]] = {}
    family_dirs = {
        "qwen": HIDDEN_DIR,
        "mistral": HIDDEN_DIR / "mistral",
    }
    for family, path in family_dirs.items():
        with (path / "rows.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        arr = np.load(path / "hidden_states.npz")
        features = arr[pooling]
        layer = BEST_LAYERS[family]
        out[family] = (rows, features[:, layer, :].astype(np.float64))
    return out


def train_hidden_predictor(pooling: str) -> list[dict[str, object]]:
    pred_rows: list[dict[str, object]] = []
    for family, (rows, x) in load_hidden_rows_and_features(pooling).items():
        y = np.array([float(r["small_full_trace_success"]) for r in rows], dtype=np.float64)
        train_mask = np.array([not stable_test_split(str(r["task_id"])) for r in rows], dtype=bool)
        x = normalize_train_test(x, train_mask)
        weights = fit_logistic_regression(x[train_mask], y[train_mask], lr=0.03, steps=3000)
        probs = sigmoid(x @ weights)
        pred_rows.extend(build_pred_rows(rows, probs))
    return pred_rows


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Hidden-State Pooling Comparison",
        "",
        "這份分析固定 family-specific best layers，直接比較 hidden-state routing 用 `mean_pooled` 還是 `last_token` 更有效。",
        "",
        f"- qwen layer: `{BEST_LAYERS['qwen']}`",
        f"- mistral layer: `{BEST_LAYERS['mistral']}`",
        "",
    ]
    for name, block in summary["models"].items():
        best = block["best"]
        lines.extend(
            [
                f"## {name}",
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
    text_pred_rows = train_text_predictor(load_text_rows())
    models = {
        "text_state_plus_family": group_test_rows(text_pred_rows),
        "hidden_state_mean_pooled": group_test_rows(train_hidden_predictor("mean_pooled")),
        "hidden_state_last_token": group_test_rows(train_hidden_predictor("last_token")),
    }

    summary: dict[str, object] = {"models": {}}
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
