from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.eval.static_evaluators import build_static_evaluator


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return cast(list[dict[str, Any]], data) if isinstance(data, list) else []


def _bucket(delta: float, eps: float) -> str:
    if delta > eps:
        return "positive"
    if delta < -eps:
        return "negative"
    return "zero"


def _question_normalized_delta(deltas: list[float], delta: float) -> float:
    max_positive = max([d for d in deltas if d > 0.0], default=0.0)
    if max_positive <= 0.0:
        return 0.0
    return delta / max_positive


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _bow(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_distance(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine = dot / (norm_a * norm_b)
    return 1.0 - cosine


_HEDGE_PATTERNS = [
    "i think",
    "maybe",
    "perhaps",
    "it seems",
    "likely",
    "possibly",
    "probably",
    "not sure",
    "unclear",
]


def _hedge_density(text: str) -> float:
    lowered = (text or "").lower()
    hits = sum(lowered.count(pattern) for pattern in _HEDGE_PATTERNS)
    token_count = max(1, len(_tokenize(text)))
    return hits / token_count


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _load_tasks(cfg: DictConfig) -> dict[str, StaticTask]:
    benchmark_name = str(cfg.dataset.benchmark)
    if benchmark_name == "olympiadbench":
        questions = load_olympiadbench(num_samples=int(cfg.dataset.olympiadbench))
    elif benchmark_name == "livebench_reasoning":
        livebench_count = int(
            cfg.dataset.get("livebench_reasoning", cfg.dataset.get("livebench", 0))
        )
        questions = load_livebench_reasoning(num_samples=livebench_count)
    else:
        raise ValueError(f"Unsupported benchmark for analysis: {benchmark_name}")
    return {q.id: q for q in questions}


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="prefix_oracle_gain_mapping",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    files = sorted(output_dir.glob("*.json"))
    if not files:
        print(f"No prefix oracle-gain result files found under {output_dir}")
        return

    analysis_cfg = cfg.get("analysis", {})
    eps = float(analysis_cfg.get("delta_epsilon", 1e-6))
    hist_bins = [
        float(x) for x in analysis_cfg.get("histogram_bins", [-1.0, -0.5, -1e-6, 1e-6, 0.5, 1.0])
    ]
    prefix_token_bin_size = int(analysis_cfg.get("prefix_token_bin_size", 100))
    summary_path_raw = analysis_cfg.get("summary_json")
    summary_path = Path(to_absolute_path(str(summary_path_raw))) if summary_path_raw else None
    per_prefix_rows_csv_raw = analysis_cfg.get("per_prefix_rows_csv")
    per_prefix_rows_csv = (
        Path(to_absolute_path(str(per_prefix_rows_csv_raw))) if per_prefix_rows_csv_raw else None
    )
    tasks = _load_tasks(cfg)

    total_tasks = 0
    total_steps = 0
    full_trace_correct = 0
    positive_tasks: set[str] = set()
    negative_tasks: set[str] = set()
    delta_counter: Counter[str] = Counter()
    zero_subtypes: Counter[str] = Counter()
    per_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_task_pattern: dict[str, list[float]] = {}
    per_step_deltas: dict[int, list[float]] = defaultdict(list)
    per_step_positive: Counter[int] = Counter()
    per_step_negative: Counter[int] = Counter()
    prefix_len_to_deltas: dict[int, list[float]] = defaultdict(list)
    prefix_tokens_to_deltas: dict[int, list[float]] = defaultdict(list)
    prefix_token_bins_to_deltas: dict[str, list[float]] = defaultdict(list)
    all_deltas: list[float] = []
    normalized_deltas: list[float] = []
    large_correct_total = 0
    small_correct_total = 0
    hist_counter: Counter[str] = Counter()
    per_prefix_rows: list[dict[str, float | int | str | bool]] = []
    per_task_roi: dict[str, dict[str, float | int]] = {}

    for file_path in files:
        rows = _load_rows(file_path)
        if not rows:
            continue
        print(f"\n== {file_path.name} ==")
        print(f"tasks: {len(rows)}")

        for row in rows:
            total_tasks += 1
            task_id = cast(str, row["task_id"])
            task = tasks[task_id]
            evaluator = build_static_evaluator(task)
            if row.get("full_trace_correct"):
                full_trace_correct += 1

            steps = cast(list[dict[str, Any]], row.get("prefix_oracle_steps", []))
            reevaluated_task_deltas: list[float] = []
            total_steps += len(steps)
            positive_steps = 0
            beneficial_and_cheaper_steps = 0
            total_provider_token_cost_delta = 0
            positive_only_provider_token_cost_delta = 0
            total_visible_token_cost_delta = 0
            positive_only_visible_token_cost_delta = 0
            beneficial_and_cheaper_visible_steps = 0

            for step in steps:
                small_answer = step.get("small_continue_answer") or ""
                large_answer = step.get("large_takeover_answer") or ""
                small_correct = evaluator(small_answer, task).is_correct
                large_correct = evaluator(large_answer, task).is_correct
                delta = float((1.0 if large_correct else 0.0) - (1.0 if small_correct else 0.0))
                reevaluated_task_deltas.append(delta)
                bucket = _bucket(delta, eps)
                step_index = int(step.get("step_index", 0))
                prefix_segments = cast(list[dict[str, Any]], step.get("prefix_segments", []))
                prefix_len = len(prefix_segments)
                prefix_text = step.get("prefix_text") or ""
                prefix_tokens = len(_tokenize(prefix_text))
                current_segment_text = str(prefix_segments[-1]["text"]) if prefix_segments else ""
                prev_segment_text = (
                    str(prefix_segments[-2]["text"]) if len(prefix_segments) >= 2 else ""
                )
                current_segment_tokens = len(_tokenize(current_segment_text))
                semantic_drift_score = (
                    _cosine_distance(_bow(prev_segment_text), _bow(current_segment_text))
                    if prev_segment_text
                    else 0.0
                )
                hedge_density = _hedge_density(current_segment_text)
                confidence_proxy = max(0.0, 1.0 - hedge_density)
                small_total_tokens = int(
                    (step.get("small_continue_cost") or {}).get("total_tokens", 0)
                )
                large_total_tokens = int(
                    (step.get("large_takeover_cost") or {}).get("total_tokens", 0)
                )
                small_input_tokens = int(
                    (step.get("small_continue_cost") or {}).get("input_tokens", 0)
                )
                small_output_tokens = int(
                    (step.get("small_continue_cost") or {}).get("output_tokens", 0)
                )
                large_input_tokens = int(
                    (step.get("large_takeover_cost") or {}).get("input_tokens", 0)
                )
                large_output_tokens = int(
                    (step.get("large_takeover_cost") or {}).get("output_tokens", 0)
                )
                small_visible_tokens = small_input_tokens + small_output_tokens
                large_visible_tokens = large_input_tokens + large_output_tokens
                token_cost_delta = large_total_tokens - small_total_tokens
                token_cost_ratio = (
                    (large_total_tokens / small_total_tokens) if small_total_tokens > 0 else 0.0
                )
                visible_token_cost_delta = large_visible_tokens - small_visible_tokens
                visible_token_cost_ratio = (
                    (large_visible_tokens / small_visible_tokens)
                    if small_visible_tokens > 0
                    else 0.0
                )
                prefix_token_bin_left = (
                    prefix_tokens // prefix_token_bin_size
                ) * prefix_token_bin_size
                prefix_token_bin = (
                    f"{prefix_token_bin_left}-{prefix_token_bin_left + prefix_token_bin_size - 1}"
                )

                delta_counter[bucket] += 1
                per_task_counts[task_id][bucket] += 1
                all_deltas.append(delta)

                per_step_deltas[step_index].append(delta)
                prefix_len_to_deltas[prefix_len].append(delta)
                prefix_tokens_to_deltas[prefix_tokens].append(delta)
                prefix_token_bins_to_deltas[prefix_token_bin].append(delta)

                if bucket == "positive":
                    positive_tasks.add(task_id)
                    per_step_positive[step_index] += 1
                    positive_steps += 1
                    positive_only_provider_token_cost_delta += token_cost_delta
                    positive_only_visible_token_cost_delta += visible_token_cost_delta
                    if token_cost_delta < 0:
                        beneficial_and_cheaper_steps += 1
                    if visible_token_cost_delta < 0:
                        beneficial_and_cheaper_visible_steps += 1
                elif bucket == "negative":
                    negative_tasks.add(task_id)
                    per_step_negative[step_index] += 1
                else:
                    if small_correct and large_correct:
                        zero_subtypes["both_correct"] += 1
                    elif (not small_correct) and (not large_correct):
                        zero_subtypes["both_wrong"] += 1
                    else:
                        zero_subtypes["mixed"] += 1

                if small_correct:
                    small_correct_total += 1
                if large_correct:
                    large_correct_total += 1
                total_provider_token_cost_delta += token_cost_delta
                total_visible_token_cost_delta += visible_token_cost_delta

                per_prefix_rows.append(
                    {
                        "task_id": task_id,
                        "prefix_id": step.get("prefix_id", ""),
                        "step_index": step_index,
                        "prefix_segments_count": prefix_len,
                        "prefix_tokens": prefix_tokens,
                        "prefix_token_bin": prefix_token_bin,
                        "current_segment_tokens": current_segment_tokens,
                        "semantic_drift_score": semantic_drift_score,
                        "hedge_density": hedge_density,
                        "confidence_proxy": confidence_proxy,
                        "small_correct": int(small_correct),
                        "large_correct": int(large_correct),
                        "delta_correctness": delta,
                        "small_total_tokens": small_total_tokens,
                        "large_total_tokens": large_total_tokens,
                        "small_visible_tokens": small_visible_tokens,
                        "large_visible_tokens": large_visible_tokens,
                        "token_cost_delta": token_cost_delta,
                        "token_cost_ratio": token_cost_ratio,
                        "visible_token_cost_delta": visible_token_cost_delta,
                        "visible_token_cost_ratio": visible_token_cost_ratio,
                        "positive_gain": int(delta > eps),
                    }
                )

                for idx in range(len(hist_bins) - 1):
                    left = hist_bins[idx]
                    right = hist_bins[idx + 1]
                    include_right = idx == len(hist_bins) - 2
                    if (left <= delta < right) or (include_right and left <= delta <= right):
                        hist_counter[f"[{left}, {right}{']' if include_right else ')'}"] += 1
                        break

            per_task_pattern[task_id] = reevaluated_task_deltas
            for delta in reevaluated_task_deltas:
                normalized_deltas.append(_question_normalized_delta(reevaluated_task_deltas, delta))

            per_task_roi[task_id] = {
                "positive_steps": positive_steps,
                "total_steps": len(steps),
                "provider_total_token_cost_delta": total_provider_token_cost_delta,
                "provider_positive_only_token_cost_delta": positive_only_provider_token_cost_delta,
                "provider_beneficial_and_cheaper_steps": beneficial_and_cheaper_steps,
                "provider_roi_gain_per_1k_extra_tokens": (
                    positive_steps / max(1e-9, positive_only_provider_token_cost_delta / 1000.0)
                    if positive_only_provider_token_cost_delta > 0
                    else float(positive_steps)
                    if positive_steps > 0
                    else 0.0
                ),
                "visible_total_token_cost_delta": total_visible_token_cost_delta,
                "visible_positive_only_token_cost_delta": positive_only_visible_token_cost_delta,
                "visible_beneficial_and_cheaper_steps": beneficial_and_cheaper_visible_steps,
                "visible_roi_gain_per_1k_extra_tokens": (
                    positive_steps / max(1e-9, positive_only_visible_token_cost_delta / 1000.0)
                    if positive_only_visible_token_cost_delta > 0
                    else float(positive_steps)
                    if positive_steps > 0
                    else 0.0
                ),
            }

        file_step_counter: Counter[str] = Counter()
        for row in rows:
            task = tasks[cast(str, row["task_id"])]
            evaluator = build_static_evaluator(task)
            for step in cast(list[dict[str, Any]], row.get("prefix_oracle_steps", [])):
                small_answer = step.get("small_continue_answer") or ""
                large_answer = step.get("large_takeover_answer") or ""
                small_correct = evaluator(small_answer, task).is_correct
                large_correct = evaluator(large_answer, task).is_correct
                delta = float((1.0 if large_correct else 0.0) - (1.0 if small_correct else 0.0))
                file_step_counter[_bucket(delta, eps)] += 1
        print(
            "step deltas:",
            dict(
                positive=file_step_counter["positive"],
                zero=file_step_counter["zero"],
                negative=file_step_counter["negative"],
            ),
        )

    avg_delta_vs_step = {
        str(step): _safe_mean(values) for step, values in sorted(per_step_deltas.items())
    }
    avg_delta_vs_prefix_len = {
        str(length): _safe_mean(values) for length, values in sorted(prefix_len_to_deltas.items())
    }
    avg_delta_vs_prefix_tokens = {
        str(tokens): _safe_mean(values)
        for tokens, values in sorted(prefix_tokens_to_deltas.items())
    }
    avg_delta_vs_prefix_token_bin = {
        token_bin: _safe_mean(values)
        for token_bin, values in sorted(
            prefix_token_bins_to_deltas.items(),
            key=lambda kv: int(kv[0].split("-")[0]),
        )
    }
    per_step_counts = {
        str(step): {
            "count": len(per_step_deltas[step]),
            "positive": per_step_positive[step],
            "negative": per_step_negative[step],
        }
        for step in sorted(per_step_deltas)
    }
    numeric_fields = [
        "delta_correctness",
        "step_index",
        "prefix_segments_count",
        "prefix_tokens",
        "current_segment_tokens",
        "semantic_drift_score",
        "hedge_density",
        "confidence_proxy",
        "small_total_tokens",
        "large_total_tokens",
        "small_visible_tokens",
        "large_visible_tokens",
        "token_cost_delta",
        "token_cost_ratio",
        "visible_token_cost_delta",
        "visible_token_cost_ratio",
    ]
    correlation_matrix = {
        field_a: {
            field_b: _pearson(
                [float(row[field_a]) for row in per_prefix_rows],
                [float(row[field_b]) for row in per_prefix_rows],
            )
            for field_b in numeric_fields
        }
        for field_a in numeric_fields
    }
    delta_correlations = {
        field: correlation_matrix["delta_correctness"][field]
        for field in numeric_fields
        if field != "delta_correctness"
    }

    summary: dict[str, Any] = {
        "tasks": total_tasks,
        "full_trace_correct": {
            "count": full_trace_correct,
            "rate": (full_trace_correct / total_tasks) if total_tasks else 0.0,
        },
        "prefix_steps": total_steps,
        "delta_buckets": {
            "positive": delta_counter["positive"],
            "zero": delta_counter["zero"],
            "negative": delta_counter["negative"],
        },
        "zero_subtypes": {
            "both_correct": zero_subtypes["both_correct"],
            "both_wrong": zero_subtypes["both_wrong"],
            "mixed": zero_subtypes["mixed"],
        },
        "tasks_with_positive_gain": {
            "count": len(positive_tasks),
            "rate": (len(positive_tasks) / total_tasks) if total_tasks else 0.0,
        },
        "tasks_with_negative_gain": {
            "count": len(negative_tasks),
            "rate": (len(negative_tasks) / total_tasks) if total_tasks else 0.0,
        },
        "small_success_rate": (small_correct_total / total_steps) if total_steps else 0.0,
        "large_success_rate": (large_correct_total / total_steps) if total_steps else 0.0,
        "avg_delta_overall": _safe_mean(all_deltas),
        "avg_normalized_delta_overall": _safe_mean(normalized_deltas),
        "avg_delta_vs_step": avg_delta_vs_step,
        "per_step_counts": per_step_counts,
        "avg_delta_vs_prefix_len": avg_delta_vs_prefix_len,
        "avg_delta_vs_prefix_tokens": avg_delta_vs_prefix_tokens,
        "avg_delta_vs_prefix_token_bin": avg_delta_vs_prefix_token_bin,
        "delta_histogram": dict(hist_counter),
        "correlation_matrix": correlation_matrix,
        "delta_correlations": delta_correlations,
        "per_task_roi": per_task_roi,
        "per_task_patterns": per_task_pattern,
        "per_task_summary": {
            task_id: {
                "positive": counts["positive"],
                "zero": counts["zero"],
                "negative": counts["negative"],
            }
            for task_id, counts in sorted(
                per_task_counts.items(), key=lambda kv: (-kv[1]["positive"], kv[0])
            )
        },
    }

    # Backward-compatible flat aliases used by downstream quick summaries.
    full_trace_correct_summary = cast(dict[str, Any], summary["full_trace_correct"])
    delta_bucket_summary = cast(dict[str, Any], summary["delta_buckets"])
    positive_gain_summary = cast(dict[str, Any], summary["tasks_with_positive_gain"])
    negative_gain_summary = cast(dict[str, Any], summary["tasks_with_negative_gain"])
    summary["full_trace_correct_tasks"] = full_trace_correct_summary["count"]
    summary["positive"] = delta_bucket_summary["positive"]
    summary["zero"] = delta_bucket_summary["zero"]
    summary["negative"] = delta_bucket_summary["negative"]
    summary["tasks_with_any_positive_gain"] = positive_gain_summary["count"]
    summary["tasks_with_any_negative_gain"] = negative_gain_summary["count"]
    summary["avg_delta_by_step"] = summary["avg_delta_vs_step"]

    # Persist outputs before verbose logging so partial runs still leave usable summaries.
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if per_prefix_rows_csv and per_prefix_rows:
        per_prefix_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        with per_prefix_rows_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_prefix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_prefix_rows)

    print("\n== Overall ==")
    print(f"tasks: {summary['tasks']}")
    print(f"full_trace_correct: {summary['full_trace_correct']['count']}/{summary['tasks']}")
    print(f"prefix_steps: {summary['prefix_steps']}")
    print(f"delta_buckets: {summary['delta_buckets']}")
    print(f"zero_subtypes: {summary['zero_subtypes']}")
    print(
        "tasks_with_any_positive_gain:"
        f" {summary['tasks_with_positive_gain']['count']}/{summary['tasks']}"
    )
    print(
        "tasks_with_any_negative_gain:"
        f" {summary['tasks_with_negative_gain']['count']}/{summary['tasks']}"
    )
    print(
        "success_rates:"
        f" small={summary['small_success_rate']:.3f}"
        f" large={summary['large_success_rate']:.3f}"
    )
    print(f"avg_delta_overall: {summary['avg_delta_overall']:.3f}")
    print(f"avg_normalized_delta_overall: {summary['avg_normalized_delta_overall']:.3f}")

    print("\n== Avg Delta vs Step ==")
    for step_label, value in avg_delta_vs_step.items():
        counts = per_step_counts[step_label]
        print(
            f"step {step_label}: avg_delta={value:.3f}"
            f" count={counts['count']}"
            f" positive={counts['positive']}"
            f" negative={counts['negative']}"
        )

    print("\n== Avg Delta vs Prefix Length ==")
    for length, value in avg_delta_vs_prefix_len.items():
        print(f"segments {length}: avg_delta={value:.3f}")

    print("\n== Avg Delta vs Prefix Tokens ==")
    for token_bin, value in avg_delta_vs_prefix_token_bin.items():
        print(f"tokens {token_bin}: avg_delta={value:.3f}")

    print("\n== Delta Histogram ==")
    for bucket in sorted(hist_counter):
        print(f"{bucket}: {hist_counter[bucket]}")

    print("\n== Delta Correlations ==")
    for field, corr in sorted(delta_correlations.items(), key=lambda kv: -abs(kv[1])):
        print(f"{field}: {corr:.3f}")

    print("\n== Per-task ROI ==")
    for task_id, roi in sorted(
        per_task_roi.items(),
        key=lambda kv: (-int(kv[1]["positive_steps"]), kv[0]),
    ):
        print(
            f"{task_id}: positive_steps={roi['positive_steps']}/{roi['total_steps']}"
            f" provider_cheaper={roi['provider_beneficial_and_cheaper_steps']}"
            f" provider_delta={roi['provider_total_token_cost_delta']}"
            f" provider_positive_delta={roi['provider_positive_only_token_cost_delta']}"
            f" provider_roi_per_1k_extra={roi['provider_roi_gain_per_1k_extra_tokens']:.3f}"
            f" | visible_cheaper={roi['visible_beneficial_and_cheaper_steps']}"
            f" visible_delta={roi['visible_total_token_cost_delta']}"
            f" visible_positive_delta={roi['visible_positive_only_token_cost_delta']}"
            f" visible_roi_per_1k_extra={roi['visible_roi_gain_per_1k_extra_tokens']:.3f}"
        )

    print("\n== Per-task Patterns ==")
    for task_id, pattern in sorted(
        per_task_pattern.items(),
        key=lambda kv: (-per_task_counts[kv[0]]["positive"], kv[0]),
    ):
        counts = per_task_counts[task_id]
        print(
            f"{task_id}: deltas={pattern}"
            f" | +{counts['positive']} / 0={counts['zero']} / -{counts['negative']}"
        )

    if summary_path:
        print(f"\nWrote summary JSON to {summary_path}")

    if per_prefix_rows_csv and per_prefix_rows:
        print(f"Wrote per-prefix rows CSV to {per_prefix_rows_csv}")


if __name__ == "__main__":
    main()
