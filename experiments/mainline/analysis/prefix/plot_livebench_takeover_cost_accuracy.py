from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
RESULT_DIRS = [
    "results/livebench_llama_to_openai_30",
    "results/livebench_llama_to_anthropic_30",
    "results/livebench_mistral_to_openai_30",
    "results/livebench_mistral_to_anthropic_30",
]
BEST_STEPS = {
    "results/livebench_llama_to_openai_30": 1,
    "results/livebench_llama_to_anthropic_30": 4,
    "results/livebench_mistral_to_openai_30": 1,
    "results/livebench_mistral_to_anthropic_30": 1,
}
OUT_DIR = ROOT / "outputs" / "analysis" / "prefix_leverage"
USD_RATES_PER_1M = {
    "meta-llama/llama-4-scout": {"input": 0.10, "output": 0.30},
    "mistralai/ministral-8b-2512": {"input": 0.15, "output": 0.15},
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "anthropic/claude-opus-4.6": {"input": 5.00, "output": 25.00},
}


@dataclass
class Point:
    label: str
    avg_cost: float
    accuracy: float
    note: str


def _token_count(text: str) -> int:
    return len((text or "").split())


def _load_task_rows(dir_path: Path) -> list[dict]:
    json_paths = [
        p
        for p in dir_path.glob("*.json")
        if p.name
        not in {"summary.json", "dataset_meta.json", "baseline_results.json", "_run_status.json"}
        and not p.name.endswith(".bak")
    ]
    if len(json_paths) != 1:
        raise ValueError(
            f"expected exactly one task json in {dir_path}, got {[p.name for p in json_paths]}"
        )
    return json.loads(json_paths[0].read_text())


def _estimate_small_prefix_cost_tokens(task_row: dict, step_index: int) -> float:
    api = task_row["metadata"]["full_trace_api_trace"]
    segments = task_row.get("segments", [])
    total_segment_words = sum(_token_count(seg.get("text", "")) for seg in segments)
    prefix_segments = [seg for seg in segments if int(seg.get("index", 0)) <= step_index]
    prefix_words = sum(_token_count(seg.get("text", "")) for seg in prefix_segments)
    if total_segment_words <= 0:
        fraction = step_index / max(1, len(segments))
    else:
        fraction = prefix_words / total_segment_words
    generated_total = int(api.get("completion_tokens", 0)) + int(api.get("reasoning_tokens", 0))
    return float(int(api.get("prompt_tokens", 0)) + fraction * generated_total)


def _estimate_small_prefix_cost_usd(task_row: dict, step_index: int, small_model: str) -> float:
    api = task_row["metadata"]["full_trace_api_trace"]
    rates = USD_RATES_PER_1M[small_model]
    segments = task_row.get("segments", [])
    total_segment_words = sum(_token_count(seg.get("text", "")) for seg in segments)
    prefix_segments = [seg for seg in segments if int(seg.get("index", 0)) <= step_index]
    prefix_words = sum(_token_count(seg.get("text", "")) for seg in prefix_segments)
    if total_segment_words <= 0:
        fraction = step_index / max(1, len(segments))
    else:
        fraction = prefix_words / total_segment_words
    prompt_tokens = int(api.get("prompt_tokens", 0))
    completion_tokens = int(api.get("completion_tokens", 0))
    return (
        prompt_tokens * rates["input"] / 1_000_000
        + fraction * completion_tokens * rates["output"] / 1_000_000
    )


def _step_cost_usd_from_cost(cost: dict, model: str) -> float:
    rates = USD_RATES_PER_1M[model]
    input_tokens = float(cost.get("input_tokens", 0))
    output_tokens = float(cost.get("output_tokens", 0))
    reasoning_tokens = float(cost.get("reasoning_tokens", 0))
    billable_output = output_tokens + reasoning_tokens
    return input_tokens * rates["input"] / 1_000_000 + billable_output * rates["output"] / 1_000_000


def _summarize_combo(
    result_dir: str,
) -> tuple[
    list[dict[str, float | int]], dict[str, Point], list[dict[str, float | int]], dict[str, Point]
]:
    dir_path = ROOT / result_dir
    task_rows = _load_task_rows(dir_path)
    small_model = task_rows[0]["small_model"].split(":")[0]
    large_model = task_rows[0]["large_model"]

    frontier_tokens: list[dict[str, float | int]] = []
    frontier_usd: list[dict[str, float | int]] = []
    all_steps = sorted(
        {
            int(step["step_index"])
            for row in task_rows
            for step in row.get("prefix_oracle_steps", [])
        }
    )
    by_step: dict[int, list[tuple[dict, dict]]] = {step: [] for step in all_steps}
    for task_row in task_rows:
        for step in task_row.get("prefix_oracle_steps", []):
            by_step[int(step["step_index"])].append((task_row, step))

    for step in all_steps:
        step_rows = by_step[step]
        token_costs = []
        usd_costs = []
        correct = 0
        for task_row, step_row in step_rows:
            prefix_tokens = _estimate_small_prefix_cost_tokens(task_row, step)
            takeover_tokens = prefix_tokens + float(
                (step_row.get("large_takeover_cost") or {}).get("total_tokens", 0)
            )
            token_costs.append(takeover_tokens)
            prefix_usd = _estimate_small_prefix_cost_usd(task_row, step, small_model)
            takeover_usd = prefix_usd + _step_cost_usd_from_cost(
                step_row.get("large_takeover_cost") or {}, large_model
            )
            usd_costs.append(takeover_usd)
            correct += int(bool(step_row.get("large_takeover_correct")))
        frontier_tokens.append(
            {
                "step": step,
                "support": len(step_rows),
                "support_rate": len(step_rows) / len(task_rows),
                "avg_cost": mean(token_costs),
                "accuracy": correct / len(step_rows),
            }
        )
        frontier_usd.append(
            {
                "step": step,
                "support": len(step_rows),
                "support_rate": len(step_rows) / len(task_rows),
                "avg_cost": mean(usd_costs),
                "accuracy": correct / len(step_rows),
            }
        )

    small_costs_tokens = []
    small_costs_usd = []
    small_correct = 0
    for row in task_rows:
        api = row["metadata"]["full_trace_api_trace"]
        small_costs_tokens.append(float(int(api.get("total_tokens", 0))))
        rates = USD_RATES_PER_1M[small_model]
        small_costs_usd.append(
            int(api.get("prompt_tokens", 0)) * rates["input"] / 1_000_000
            + int(api.get("completion_tokens", 0)) * rates["output"] / 1_000_000
        )
        small_correct += int(bool(row.get("full_trace_correct")))
    all_small_tokens = Point(
        label="all_small",
        avg_cost=mean(small_costs_tokens),
        accuracy=small_correct / len(task_rows),
        note="direct small full-trace",
    )
    all_small_usd = Point(
        label="all_small",
        avg_cost=mean(small_costs_usd),
        accuracy=small_correct / len(task_rows),
        note="direct small full-trace",
    )

    step1_rows = [step_row for _, step_row in by_step[1]]
    large_proxy_tokens = Point(
        label="all_large_proxy",
        avg_cost=mean(
            float((row.get("large_takeover_cost") or {}).get("total_tokens", 0))
            for row in step1_rows
        ),
        accuracy=sum(int(bool(row.get("large_takeover_correct"))) for row in step1_rows)
        / len(step1_rows),
        note="earliest takeover proxy, not from-scratch large",
    )
    large_proxy_usd = Point(
        label="all_large_proxy",
        avg_cost=mean(
            _step_cost_usd_from_cost(row.get("large_takeover_cost") or {}, large_model)
            for row in step1_rows
        ),
        accuracy=sum(int(bool(row.get("large_takeover_correct"))) for row in step1_rows)
        / len(step1_rows),
        note="earliest takeover proxy, not from-scratch large",
    )

    best_step = BEST_STEPS[result_dir]
    best_row_tokens = next(item for item in frontier_tokens if int(item["step"]) == best_step)
    best_row_usd = next(item for item in frontier_usd if int(item["step"]) == best_step)
    takeover_tokens = Point(
        label="takeover",
        avg_cost=float(best_row_tokens["avg_cost"]),
        accuracy=float(best_row_tokens["accuracy"]),
        note=f"best supported step={best_step}",
    )
    takeover_usd = Point(
        label="takeover",
        avg_cost=float(best_row_usd["avg_cost"]),
        accuracy=float(best_row_usd["accuracy"]),
        note=f"best supported step={best_step}",
    )

    denom_tokens = large_proxy_tokens.avg_cost - all_small_tokens.avg_cost
    if abs(denom_tokens) < 1e-9:
        mix_prob_large_tokens = 0.0
    else:
        mix_prob_large_tokens = (
            takeover_tokens.avg_cost - all_small_tokens.avg_cost
        ) / denom_tokens
    mix_prob_large_tokens = min(1.0, max(0.0, mix_prob_large_tokens))
    random_mix_tokens = Point(
        label="random_mix_cost_matched",
        avg_cost=all_small_tokens.avg_cost + mix_prob_large_tokens * denom_tokens,
        accuracy=all_small_tokens.accuracy
        + mix_prob_large_tokens * (large_proxy_tokens.accuracy - all_small_tokens.accuracy),
        note=f"mix_prob_large={mix_prob_large_tokens:.3f}",
    )

    denom_usd = large_proxy_usd.avg_cost - all_small_usd.avg_cost
    if abs(denom_usd) < 1e-12:
        mix_prob_large_usd = 0.0
    else:
        mix_prob_large_usd = (takeover_usd.avg_cost - all_small_usd.avg_cost) / denom_usd
    mix_prob_large_usd = min(1.0, max(0.0, mix_prob_large_usd))
    random_mix_usd = Point(
        label="random_mix_cost_matched",
        avg_cost=all_small_usd.avg_cost + mix_prob_large_usd * denom_usd,
        accuracy=all_small_usd.accuracy
        + mix_prob_large_usd * (large_proxy_usd.accuracy - all_small_usd.accuracy),
        note=f"mix_prob_large={mix_prob_large_usd:.3f}",
    )

    return (
        frontier_tokens,
        {
            "all_small": all_small_tokens,
            "all_large_proxy": large_proxy_tokens,
            "takeover": takeover_tokens,
            "random_mix_cost_matched": random_mix_tokens,
        },
        frontier_usd,
        {
            "all_small": all_small_usd,
            "all_large_proxy": large_proxy_usd,
            "takeover": takeover_usd,
            "random_mix_cost_matched": random_mix_usd,
        },
    )


def _pretty_name(result_dir: str) -> str:
    return result_dir.split("/")[-1].replace("_30", "")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combo_rows: list[dict[str, str | float | int]] = []
    anchor_rows: list[dict[str, str | float]] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, result_dir in zip(axes_flat, RESULT_DIRS):
        _, _, frontier, anchors = _summarize_combo(result_dir)
        combo_name = _pretty_name(result_dir)

        frontier_sorted = sorted(frontier, key=lambda row: float(row["avg_cost"]))
        ax.plot(
            [float(row["avg_cost"]) for row in frontier_sorted],
            [float(row["accuracy"]) for row in frontier_sorted],
            color="#4472c4",
            marker="o",
            linewidth=1.5,
            markersize=4,
            alpha=0.75,
            label="takeover step frontier",
        )

        colors = {
            "all_small": "#2ca02c",
            "all_large_proxy": "#d62728",
            "takeover": "#ff7f0e",
            "random_mix_cost_matched": "#9467bd",
        }
        markers = {
            "all_small": "s",
            "all_large_proxy": "X",
            "takeover": "D",
            "random_mix_cost_matched": "^",
        }
        labels = {
            "all_small": "all small",
            "all_large_proxy": "all large proxy",
            "takeover": "takeover",
            "random_mix_cost_matched": "random mix baseline",
        }

        for key, point in anchors.items():
            ax.scatter(
                [point.avg_cost],
                [point.accuracy],
                color=colors[key],
                marker=markers[key],
                s=90,
                zorder=3,
                label=labels[key],
            )
            ax.annotate(
                labels[key],
                (point.avg_cost, point.accuracy),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
            anchor_rows.append(
                {
                    "combo": combo_name,
                    "point": key,
                    "avg_cost": round(point.avg_cost, 9),
                    "accuracy": round(point.accuracy, 6),
                    "note": point.note,
                }
            )

        for row in frontier:
            combo_rows.append(
                {
                    "combo": combo_name,
                    "step": int(row["step"]),
                    "support": int(row["support"]),
                    "support_rate": round(float(row["support_rate"]), 6),
                    "avg_cost": round(float(row["avg_cost"]), 9),
                    "accuracy": round(float(row["accuracy"]), 6),
                }
            )

        ax.set_title(combo_name)
        ax.set_xlabel("Average Cost (USD)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))

    plot_path = OUT_DIR / "livebench_takeover_cost_accuracy_usd.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (OUT_DIR / "livebench_takeover_frontier_usd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["combo", "step", "support", "support_rate", "avg_cost", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(combo_rows)

    with (OUT_DIR / "livebench_takeover_anchor_points_usd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["combo", "point", "avg_cost", "accuracy", "note"],
        )
        writer.writeheader()
        writer.writerows(anchor_rows)

    md_lines = [
        "# Livebench Takeover Cost-Accuracy",
        "",
        "Caveat:",
        "- `all_large_proxy` uses the earliest available takeover point at step 1.",
        "- The repo does not currently contain a clean from-scratch large-model baseline for these four combos.",
        "- Prices use current OpenRouter list prices for these model slugs.",
        "- `takeover` total cost is estimated as `small prefix partial generation + large takeover call`.",
        "- The small prefix partial output cost is approximated by allocating full small-trace completion cost proportionally to prefix segment length.",
        "",
        "| combo | point | avg_cost | accuracy | note |",
        "|---|---|---:|---:|---|",
    ]
    for row in anchor_rows:
        md_lines.append(
            f"| {row['combo']} | {row['point']} | {row['avg_cost']:.6f} | {row['accuracy']:.3f} | {row['note']} |"
        )
    (OUT_DIR / "livebench_takeover_cost_accuracy_usd.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
