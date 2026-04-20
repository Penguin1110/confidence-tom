from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RUNS = {
    "qwen_to_openai_50": "results/qwen_to_openai_50/Qwen_3_14B_to_GPT_5_4.json",
    "qwen_to_anthropic_50": "results/qwen_to_anthropic_50/Qwen_3_14B_to_Claude_Opus_4_6.json",
    "llama_to_openai_50": "results/llama_to_openai_50/Llama_4_Scout_to_GPT_5_4.json",
    "llama_to_anthropic_50": "results/llama_to_anthropic_50/Llama_4_Scout_to_Claude_Opus_4_6.json",
    "mistral_to_openai_50": "results/mistral_to_openai_50/Ministral_8B_2512_to_GPT_5_4.json",
    "mistral_to_anthropic_50": "results/mistral_to_anthropic_50/Ministral_8B_2512_to_Claude_Opus_4_6.json",
}


def build_summary(data: list[dict]) -> dict:
    tasks = len(data)
    full_trace_correct_tasks = sum(1 for row in data if row.get("full_trace_correct"))
    prefix_steps = 0
    positive = zero = negative = 0
    tasks_with_any_positive_gain = 0
    tasks_with_any_negative_gain = 0
    small_correct_total = 0
    large_correct_total = 0
    per_step = defaultdict(list)
    per_step_pos = defaultdict(int)
    per_step_neg = defaultdict(int)
    zero_both_correct = 0
    zero_both_wrong = 0
    zero_mixed = 0
    per_task_summary = {}
    for row in data:
        any_pos = False
        any_neg = False
        tpos = tzero = tneg = 0
        for step in row.get("prefix_oracle_steps", []):
            prefix_steps += 1
            sc = bool(step.get("small_continue_correct"))
            lc = bool(step.get("large_takeover_correct"))
            delta = (1 if lc else 0) - (1 if sc else 0)
            step_index = int(step.get("step_index", 0))
            per_step[step_index].append(delta)
            if delta > 0:
                positive += 1
                tpos += 1
                any_pos = True
                per_step_pos[step_index] += 1
            elif delta < 0:
                negative += 1
                tneg += 1
                any_neg = True
                per_step_neg[step_index] += 1
            else:
                zero += 1
                tzero += 1
                if sc and lc:
                    zero_both_correct += 1
                elif (not sc) and (not lc):
                    zero_both_wrong += 1
                else:
                    zero_mixed += 1
            small_correct_total += 1 if sc else 0
            large_correct_total += 1 if lc else 0
        if any_pos:
            tasks_with_any_positive_gain += 1
        if any_neg:
            tasks_with_any_negative_gain += 1
        per_task_summary[row["task_id"]] = {"positive": tpos, "zero": tzero, "negative": tneg}
    avg_delta_by_step = {str(k): sum(v) / len(v) for k, v in sorted(per_step.items())}
    per_step_counts = {
        str(k): {"count": len(v), "positive": per_step_pos[k], "negative": per_step_neg[k]}
        for k, v in sorted(per_step.items())
    }
    return {
        "tasks": tasks,
        "full_trace_correct_tasks": full_trace_correct_tasks,
        "prefix_steps": prefix_steps,
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "tasks_with_any_positive_gain": tasks_with_any_positive_gain,
        "tasks_with_any_negative_gain": tasks_with_any_negative_gain,
        "small_success_rate": (small_correct_total / prefix_steps) if prefix_steps else 0.0,
        "large_success_rate": (large_correct_total / prefix_steps) if prefix_steps else 0.0,
        "avg_delta_by_step": avg_delta_by_step,
        "per_step_counts": per_step_counts,
        "zero_subtypes": {
            "both_correct": zero_both_correct,
            "both_wrong": zero_both_wrong,
            "mixed": zero_mixed,
        },
        "per_task_summary": per_task_summary,
    }


def main() -> None:
    for run, fp in RUNS.items():
        path = Path(fp)
        data = json.loads(path.read_text())
        summary = build_summary(data)
        out = path.parent / "summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
