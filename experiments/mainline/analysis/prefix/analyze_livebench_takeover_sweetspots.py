from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]
RESULT_DIRS = [
    "results/livebench_llama_to_openai_30",
    "results/livebench_llama_to_anthropic_30",
    "results/livebench_mistral_to_openai_30",
    "results/livebench_mistral_to_anthropic_30",
]
BASELINE_FILES = {
    "openai/gpt-5.4": ROOT
    / "outputs"
    / "results"
    / "livebench_large_baselines"
    / "gpt54_livebench30.json",
    "anthropic/claude-opus-4.6": ROOT
    / "outputs"
    / "results"
    / "livebench_large_baselines"
    / "opus46_livebench30.json",
}
OUT_DIR = ROOT / "outputs" / "analysis" / "prefix_leverage"
BOOTSTRAP_DRAWS = 500
MIN_SUPPORT_RATE = 0.5

USD_RATES_PER_1M = {
    "meta-llama/llama-4-scout": {"input": 0.10, "output": 0.30},
    "mistralai/ministral-8b-2512": {"input": 0.15, "output": 0.15},
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "anthropic/claude-opus-4.6": {"input": 5.00, "output": 25.00},
}


def _token_count(text: str) -> int:
    return len((text or "").split())


def _estimate_small_prefix_cost_usd(task_row: dict, step_index: int, small_model: str) -> float:
    api = task_row["metadata"]["full_trace_api_trace"]
    rates = USD_RATES_PER_1M[small_model]
    segments = task_row.get("segments", [])
    total_segment_words = sum(_token_count(seg.get("text", "")) for seg in segments)
    prefix_segments = [seg for seg in segments if int(seg.get("index", 0)) <= step_index]
    prefix_words = sum(_token_count(seg.get("text", "")) for seg in prefix_segments)
    fraction = (
        (prefix_words / total_segment_words)
        if total_segment_words > 0
        else step_index / max(1, len(segments))
    )
    prompt_tokens = int(api.get("prompt_tokens", 0))
    completion_tokens = int(api.get("completion_tokens", 0))
    return (
        prompt_tokens * rates["input"] / 1_000_000
        + fraction * completion_tokens * rates["output"] / 1_000_000
    )


def _cost_usd(cost: dict, model: str) -> float:
    rates = USD_RATES_PER_1M[model]
    return (
        float(cost.get("input_tokens", 0)) * rates["input"] / 1_000_000
        + (float(cost.get("output_tokens", 0)) + float(cost.get("reasoning_tokens", 0)))
        * rates["output"]
        / 1_000_000
    )


def _load_combo(result_dir: str) -> tuple[list[dict], dict[str, dict]]:
    path = ROOT / result_dir
    task_file = [
        p
        for p in path.glob("*.json")
        if p.name
        not in {"summary.json", "dataset_meta.json", "baseline_results.json", "_run_status.json"}
        and not p.name.endswith(".bak")
    ][0]
    tasks = json.loads(task_file.read_text())
    return tasks, {row["task_id"]: row for row in tasks}


def _load_baseline(model: str) -> dict[str, dict]:
    rows = json.loads(BASELINE_FILES[model].read_text())
    return {row["task_id"]: row for row in rows}


def _combo_records(result_dir: str) -> tuple[str, list[dict]]:
    tasks, _ = _load_combo(result_dir)
    small_model = tasks[0]["small_model"].split(":")[0]
    large_model = tasks[0]["large_model"]
    baseline = _load_baseline(large_model)
    records = []
    for task in tasks:
        task_id = task["task_id"]
        base = baseline[task_id]
        full_api = task["metadata"]["full_trace_api_trace"]
        small_cost = (
            int(full_api.get("prompt_tokens", 0))
            * USD_RATES_PER_1M[small_model]["input"]
            / 1_000_000
            + int(full_api.get("completion_tokens", 0))
            * USD_RATES_PER_1M[small_model]["output"]
            / 1_000_000
        )
        rec = {
            "task_id": task_id,
            "small_correct": int(bool(task.get("full_trace_correct"))),
            "small_cost": small_cost,
            "large_correct": int(bool(base.get("correct"))),
            "large_cost": _cost_usd(base.get("cost") or {}, large_model),
            "steps": {},
        }
        for step in task.get("prefix_oracle_steps", []):
            step_index = int(step["step_index"])
            rec["steps"][step_index] = {
                "correct": int(bool(step.get("large_takeover_correct"))),
                "cost": _estimate_small_prefix_cost_usd(task, step_index, small_model)
                + _cost_usd(step.get("large_takeover_cost") or {}, large_model),
            }
        records.append(rec)
    return result_dir.split("/")[-1].replace("_30", ""), records


def _mix_accuracy(
    target_cost: float, small_cost: float, small_acc: float, large_cost: float, large_acc: float
) -> float:
    denom = large_cost - small_cost
    if abs(denom) < 1e-12:
        p = 1.0
    else:
        p = (target_cost - small_cost) / denom
    p = min(1.0, max(0.0, p))
    return small_acc + p * (large_acc - small_acc)


def _bootstrap_best_step(records: list[dict], rng: random.Random) -> dict:
    sample = [records[rng.randrange(len(records))] for _ in range(len(records))]
    small_cost = mean(r["small_cost"] for r in sample)
    small_acc = mean(r["small_correct"] for r in sample)
    large_cost = mean(r["large_cost"] for r in sample)
    large_acc = mean(r["large_correct"] for r in sample)
    step_pool = sorted({step for r in sample for step in r["steps"].keys()})
    candidates = []
    for step in step_pool:
        support_rows = [r for r in sample if step in r["steps"]]
        support_rate = len(support_rows) / len(sample)
        if support_rate < MIN_SUPPORT_RATE:
            continue
        takeover_cost = mean(r["steps"][step]["cost"] for r in support_rows)
        takeover_acc = mean(r["steps"][step]["correct"] for r in support_rows)
        mix_acc = _mix_accuracy(takeover_cost, small_cost, small_acc, large_cost, large_acc)
        sweet_gain = takeover_acc - mix_acc
        candidates.append(
            {
                "step": step,
                "support_rate": support_rate,
                "takeover_cost": takeover_cost,
                "takeover_acc": takeover_acc,
                "mix_acc": mix_acc,
                "sweet_gain": sweet_gain,
            }
        )
    if not candidates:
        return {"best_step": None, "best_gain": None}
    best = max(candidates, key=lambda row: (row["sweet_gain"], -row["takeover_cost"], -row["step"]))
    return {
        "best_step": best["step"],
        "best_gain": best["sweet_gain"],
        "small_cost": small_cost,
        "large_cost": large_cost,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    summary_rows = []
    bootstrap_rows = []
    for result_dir in RESULT_DIRS:
        combo_name, records = _combo_records(result_dir)
        draws = [_bootstrap_best_step(records, rng) for _ in range(BOOTSTRAP_DRAWS)]
        step_counter = Counter(draw["best_step"] for draw in draws if draw["best_step"] is not None)
        gain_values = [draw["best_gain"] for draw in draws if draw["best_gain"] is not None]
        top_step, top_count = step_counter.most_common(1)[0] if step_counter else (None, 0)
        summary_rows.append(
            {
                "combo": combo_name,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mode_best_step": top_step,
                "mode_best_step_rate": round(top_count / BOOTSTRAP_DRAWS, 6),
                "interior_step_win_rate": round(
                    sum(count for step, count in step_counter.items() if step and step > 1)
                    / BOOTSTRAP_DRAWS,
                    6,
                ),
                "mean_best_gain_vs_mix": round(mean(gain_values), 6) if gain_values else None,
            }
        )
        for step, count in sorted(step_counter.items()):
            bootstrap_rows.append(
                {
                    "combo": combo_name,
                    "best_step": step,
                    "count": count,
                    "rate": round(count / BOOTSTRAP_DRAWS, 6),
                }
            )

    with (OUT_DIR / "livebench_sweetspot_bootstrap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "combo",
                "bootstrap_draws",
                "mode_best_step",
                "mode_best_step_rate",
                "interior_step_win_rate",
                "mean_best_gain_vs_mix",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUT_DIR / "livebench_sweetspot_bootstrap_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["combo", "best_step", "count", "rate"])
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    lines = [
        "# Livebench Sweetspot Bootstrap",
        "",
        "Bootstrap definition:",
        f"- draws = {BOOTSTRAP_DRAWS}",
        f"- step candidates require support >= {MIN_SUPPORT_RATE:.0%} of sampled tasks",
        "- sweet_gain = takeover_accuracy - random_mix_accuracy_at_same_cost",
        "",
        "| combo | mode_best_step | mode_rate | interior_step_win_rate | mean_best_gain_vs_mix |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['combo']} | {row['mode_best_step']} | {row['mode_best_step_rate']:.3f} | {row['interior_step_win_rate']:.3f} | {row['mean_best_gain_vs_mix']:.3f} |"
        )
    (OUT_DIR / "livebench_sweetspot_bootstrap.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
