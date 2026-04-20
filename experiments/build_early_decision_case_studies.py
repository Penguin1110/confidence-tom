#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EARLY_DIR = RESULTS_DIR / "_early_decision_v1"
DOCS_DIR = ROOT / "docs"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def short_text(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def find_run_file(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        p
        for p in run_dir.glob("*.json")
        if p.name != "summary.json" and not p.name.startswith("per_prefix")
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected 1 run json for {run_name}, found {candidates}")
    return candidates[0]


def load_run_task(run_name: str, task_id: str) -> dict[str, Any]:
    run_path = find_run_file(run_name)
    data = load_json(run_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {run_path}")
    for row in data:
        if row.get("task_id") == task_id:
            return row
    raise KeyError(f"Task {task_id} not found in {run_path}")


def make_task_index(
    rows: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row.get(k) for k in key_fields): row for row in rows}


def build_step_table(
    task_row: dict[str, Any],
    msp_row: dict[str, Any] | None,
    bottleneck_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    label = msp_row["label"] if msp_row else None
    prob_by_step: dict[int, float] = {}
    pred_by_step: dict[int, int] = {}
    if msp_row:
        for item in msp_row["trajectory"]:
            prob = float(item["prob"])
            pred = int(item["pred"])
            if label == 0:
                prob = 1.0 - prob
            prob_by_step[int(item["step_index"])] = prob
            pred_by_step[int(item["step_index"])] = pred

    jump_by_step: dict[int, float] = {}
    if bottleneck_row:
        for item in bottleneck_row["trajectory"]:
            jump_by_step[int(item["step_index"])] = float(item["jump_from_prev"])

    table = []
    for step in task_row.get("prefix_oracle_steps", []):
        idx = int(step["step_index"])
        table.append(
            {
                "step_index": idx,
                "prefix_prob_correct_label": prob_by_step.get(idx),
                "prefix_pred": pred_by_step.get(idx),
                "prob_jump": jump_by_step.get(idx),
                "small_continue_correct": bool(step.get("small_continue_correct")),
                "large_takeover_correct": bool(step.get("large_takeover_correct")),
                "delta_correctness": int(step.get("delta_correctness", 0)),
                "prefix_snippet": short_text(step.get("prefix_text", ""), 120),
            }
        )
    return table


def build_segment_outline(task_row: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    outline = []
    for seg in task_row.get("segments", [])[:limit]:
        outline.append(
            {
                "index": int(seg["index"]),
                "snippet": short_text(seg.get("text", ""), 180),
            }
        )
    return outline


def build_case(
    title: str,
    rationale: str,
    alignment_row: dict[str, Any] | None,
    msp_row: dict[str, Any] | None,
    bottleneck_row: dict[str, Any] | None,
) -> dict[str, Any]:
    base = alignment_row or msp_row or bottleneck_row
    if base is None:
        raise ValueError("At least one row is required")

    run_name = base["run_name"]
    task_id = base["task_id"]
    task_row = load_run_task(run_name, task_id)

    summary = {
        "title": title,
        "rationale": rationale,
        "run_name": run_name,
        "task_id": task_id,
        "benchmark": base.get("benchmark"),
        "small_family": base.get("small_family"),
        "large_family": alignment_row.get("large_family") if alignment_row else None,
        "label": msp_row.get("label") if msp_row else None,
        "full_trace_correct": bool(task_row.get("full_trace_correct")),
        "minimal_sufficient_step": msp_row.get("minimal_sufficient_step") if msp_row else None,
        "first_cross_60": bottleneck_row.get("first_cross_60") if bottleneck_row else None,
        "first_cross_70": bottleneck_row.get("first_cross_70") if bottleneck_row else None,
        "first_cross_80": bottleneck_row.get("first_cross_80") if bottleneck_row else None,
        "earliest_positive_step": alignment_row.get("earliest_positive_step")
        if alignment_row
        else None,
        "earliest_negative_step": alignment_row.get("earliest_negative_step")
        if alignment_row
        else None,
        "positive_steps": alignment_row.get("positive_steps") if alignment_row else None,
        "negative_steps": alignment_row.get("negative_steps") if alignment_row else None,
        "total_steps": alignment_row.get("total_steps")
        if alignment_row
        else len(task_row.get("prefix_oracle_steps", [])),
        "segment_outline": build_segment_outline(task_row),
        "step_table": build_step_table(task_row, msp_row, bottleneck_row),
        "final_answer": task_row.get("full_trace_answer"),
    }
    return summary


def md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Early Decision Case Studies")
    lines.append("")
    lines.append("這份筆記把 aggregate 指標拆回題目層級，目標是看清楚三件事：")
    lines.append("")
    lines.append("- 哪些題的 early diagnosis 和 early takeover 很早就對齊")
    lines.append("- 哪些題的 takeover 機會出現得比穩定診斷更早")
    lines.append("- 哪些題根本沒有穩定的 MSP")
    lines.append("")
    for case in cases:
        lines.append(f"## {case['title']}")
        lines.append("")
        lines.append(case["rationale"])
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        fields = [
            ("run_name", case["run_name"]),
            ("task_id", case["task_id"]),
            ("benchmark", case["benchmark"]),
            ("small_family", case["small_family"]),
            ("large_family", case["large_family"]),
            ("full_trace_correct", case["full_trace_correct"]),
            ("minimal_sufficient_step", case["minimal_sufficient_step"]),
            ("first_cross_60", case["first_cross_60"]),
            ("first_cross_70", case["first_cross_70"]),
            ("earliest_positive_step", case["earliest_positive_step"]),
            ("earliest_negative_step", case["earliest_negative_step"]),
            ("positive_steps", case["positive_steps"]),
            ("negative_steps", case["negative_steps"]),
            ("total_steps", case["total_steps"]),
        ]
        for key, value in fields:
            lines.append(f"| `{key}` | {md_value(value)} |")
        lines.append("")
        lines.append("### Segment Outline")
        lines.append("")
        for seg in case["segment_outline"]:
            lines.append(f"- step {seg['index']}: {seg['snippet']}")
        lines.append("")
        lines.append("### Step-Level View")
        lines.append("")
        lines.append("| step | p(correct label) | pred | jump | small | large | delta |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in case["step_table"]:
            lines.append(
                "| {step} | {prob} | {pred} | {jump} | {small} | {large} | {delta} |".format(
                    step=row["step_index"],
                    prob=md_value(row["prefix_prob_correct_label"]),
                    pred=md_value(row["prefix_pred"]),
                    jump=md_value(row["prob_jump"]),
                    small=int(row["small_continue_correct"]),
                    large=int(row["large_takeover_correct"]),
                    delta=row["delta_correctness"],
                )
            )
        lines.append("")
        lines.append("### Reading")
        lines.append("")
        step_rows = case["step_table"]
        notes = []
        if (
            case["minimal_sufficient_step"] is not None
            and case["earliest_positive_step"] is not None
        ):
            if case["minimal_sufficient_step"] <= case["earliest_positive_step"]:
                notes.append(
                    "MSP 不晚於第一個 positive takeover step，代表穩定診斷與介入機會是同步或診斷更早成熟。"
                )
            else:
                notes.append(
                    "第一個 positive takeover step 早於 MSP，代表介入機會先出現，但穩定診斷稍後才成熟。"
                )
        elif case["minimal_sufficient_step"] is None:
            notes.append("整條 trajectory 沒有長出穩定 MSP，表示這題的 outcome 診斷一路都不夠穩。")
        if case["first_cross_70"] is not None:
            notes.append(
                f"`cross70` 出現在 step {case['first_cross_70']}，可視為較高信心判斷真正形成的時間。"
            )
        if any(row["delta_correctness"] > 0 for row in step_rows):
            positive_steps = [
                str(row["step_index"]) for row in step_rows if row["delta_correctness"] > 0
            ]
            notes.append("positive takeover 出現在 step " + ", ".join(positive_steps) + "。")
        if any(row["delta_correctness"] < 0 for row in step_rows):
            negative_steps = [
                str(row["step_index"]) for row in step_rows if row["delta_correctness"] < 0
            ]
            notes.append("negative takeover 出現在 step " + ", ".join(negative_steps) + "。")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    alignment = load_json(EARLY_DIR / "early_decision_takeover_alignment.json")
    msp = load_json(EARLY_DIR / "minimal_sufficient_prefix_analysis.json")
    bottleneck = load_json(EARLY_DIR / "early_decision_bottlenecks.json")

    align_index = make_task_index(alignment["rows_detail"], ("run_name", "task_id"))
    msp_index = make_task_index(msp["task_groups"], ("run_name", "task_id"))
    bottleneck_index = make_task_index(bottleneck["task_groups"], ("run_name", "task_id"))

    selections = [
        (
            "Case 1: Early Diagnosis and Early Takeover Align",
            "這題是乾淨的 aligned case。MSP、cross70 和第一個 positive takeover 幾乎同一步成熟，適合用來說明 early diagnosis 可以直接對應 intervention timing。",
            (
                "livebench_mistral_to_openai_30",
                "livebench_reasoning_3de6bc30b87b4698a32f4e705b8e1b7aeb4a37b06bc53da26ce9739720f13a62_0015",
            ),
        ),
        (
            "Case 2: Positive Takeover Appears Before Stable Diagnosis",
            "這題是 misaligned case。positive takeover 很早就出現，但 MSP 和 cross70 稍後才穩定，說明 intervention opportunity 可能比高信心診斷更早成熟。",
            (
                "livebench_llama_to_anthropic_30",
                "livebench_reasoning_7f1b41d1cdf3a3cf65c6107f8bb29f112137dc875cff80dc667c4cab83c4037a_0010",
            ),
        ),
        (
            "Case 3: No Stable MSP Despite Long Trace",
            "這題代表 long-tail OlympiadBench failure。prediction trajectory 會來回翻，最後沒有形成穩定 MSP，很適合說明 heterogeneous benchmark 的診斷困難。",
            ("llama_to_openai_50", "olympiadbench_2453_0027"),
        ),
        (
            "Case 4: Late High-Confidence Bottleneck",
            "這題的高信心 crossing 出現在最後一步，適合拿來說明有些題不是沒有訊號，而是 decisive information 的確偏晚才出現。",
            ("qwen_to_openai_50", "olympiadbench_2746_0038"),
        ),
    ]

    cases = []
    for title, rationale, key in selections:
        cases.append(
            build_case(
                title=title,
                rationale=rationale,
                alignment_row=align_index.get(key),
                msp_row=msp_index.get(key),
                bottleneck_row=bottleneck_index.get(key),
            )
        )

    output_json = {
        "notes": [
            "Cases were selected to cover aligned, misaligned, no-MSP, and late-bottleneck regimes.",
            "Case 4 may not have a takeover row if it was selected from bottleneck analysis only; the case is kept to illustrate late decisive information.",
        ],
        "cases": cases,
    }
    write_json(EARLY_DIR / "early_decision_case_studies.json", output_json)
    (DOCS_DIR / "early_decision_case_studies.md").write_text(
        render_markdown(cases),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
