from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from confidence_tom.infra.client import LLMClient

AttributionLabel = Literal["error_catch", "path_inconsistency", "ambiguous", "empty_or_unusable"]

_SYSTEM_PROMPT = """You are classifying FLAG responses from a reasoning-edit pilot.
Return exactly one JSON object and nothing else.

Schema:
{
  "label": "error_catch | path_inconsistency | ambiguous",
  "rationale": "brief justification"
}

Definitions:
- error_catch: the model mainly says the target step is mathematically/factually wrong, uses the wrong formula, has the wrong value, or contains a local mistake that can be checked on its own.
- path_inconsistency: the model mainly says the target step does not fit the earlier reasoning path, contradicts previous setup, switches routes, or is inconsistent with prior steps.
- ambiguous: both interpretations are present, or the response is too vague to tell.

Important:
- Prefer error_catch if the response explicitly focuses on a local mathematical mistake.
- Prefer path_inconsistency only when the response's core complaint depends on comparing against earlier steps or the prior route.
- Ignore whether the trial was intended to be type C; classify only what the FLAG response actually says.
"""


class AttributionOutput(BaseModel):
    label: Literal["error_catch", "path_inconsistency", "ambiguous"]
    rationale: str = ""


class AttributionRecord(BaseModel):
    task_id: str
    elicitation: str
    label: AttributionLabel
    rationale: str = ""
    model_response: str = ""
    edited_step_text: str = ""
    judge_note: str = ""


def _load_trials(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data.get("trials", []))


def _build_prompt(trial: dict) -> str:
    original_steps = trial.get("original_steps", [])
    edited_steps = trial.get("edited_steps", [])
    edit_index = int(trial.get("edit_index", 0))
    target_step_num = edit_index + 1
    original_target = original_steps[edit_index] if edit_index < len(original_steps) else ""
    edited_target = edited_steps[edit_index] if edit_index < len(edited_steps) else ""
    prior_steps = original_steps[:edit_index]

    lines = [
        f"Task ID: {trial.get('task_id', '')}",
        f"Elicitation: {trial.get('elicitation', '')}",
        "",
        "Prior steps:",
    ]
    for idx, step in enumerate(prior_steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.extend(
        [
            "",
            f"Original step {target_step_num}: {original_target}",
            f"Edited step {target_step_num}: {edited_target}",
            "",
            "Model FLAG response:",
            trial.get("model_response", ""),
        ]
    )
    return "\n".join(lines)


async def _classify_trial(client: LLMClient, trial: dict) -> AttributionRecord:
    raw_response = (trial.get("model_response") or "").strip()
    if not raw_response:
        return AttributionRecord(
            task_id=str(trial.get("task_id", "")),
            elicitation=str(trial.get("elicitation", "")),
            label="empty_or_unusable",
            model_response="",
            edited_step_text=str(trial.get("edited_step_text", "")),
            judge_note=str(trial.get("judge_note", "")),
            rationale="No readable model_response was stored for this FLAG trial.",
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(trial)},
    ]
    parsed, trace = await client.agenerate_with_trace(messages, AttributionOutput)
    if not isinstance(parsed, AttributionOutput):
        trace_text = getattr(trace, "response_content", "") if trace is not None else ""
        return AttributionRecord(
            task_id=str(trial.get("task_id", "")),
            elicitation=str(trial.get("elicitation", "")),
            label="ambiguous",
            model_response=raw_response,
            edited_step_text=str(trial.get("edited_step_text", "")),
            judge_note=str(trial.get("judge_note", "")),
            rationale=(
                "Classifier did not return a valid schema."
                + (f" Raw classifier output: {trace_text[:300]}" if trace_text else "")
            ),
        )

    return AttributionRecord(
        task_id=str(trial.get("task_id", "")),
        elicitation=str(trial.get("elicitation", "")),
        label=parsed.label,
        rationale=parsed.rationale,
        model_response=raw_response,
        edited_step_text=str(trial.get("edited_step_text", "")),
        judge_note=str(trial.get("judge_note", "")),
    )


def _share(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _render_markdown(
    *,
    source_path: Path,
    model: str,
    records: list[AttributionRecord],
) -> str:
    counts = Counter(record.label for record in records)
    nonempty = [record for record in records if record.label != "empty_or_unusable"]
    nonempty_counts = Counter(record.label for record in nonempty)

    lines = [
        "# C FLAG Attribution Analysis",
        "",
        f"- Source: `{source_path}`",
        f"- Classifier model: `{model}`",
        f"- Total C FLAG trials: {len(records)}",
        f"- Empty or unusable responses: {counts['empty_or_unusable']}",
        "",
        "## Counts",
        "",
        "| Label | Count | Share of all C FLAG | Share of nonempty C FLAG |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in ["error_catch", "path_inconsistency", "ambiguous", "empty_or_unusable"]:
        count = counts[label]
        nonempty_share = (
            f"{_share(nonempty_counts[label], len(nonempty)):.3f}"
            if label != "empty_or_unusable"
            else "-"
        )
        lines.append(
            f"| {label} | {count} | {_share(count, len(records)):.3f} | {nonempty_share} |"
        )

    def _add_examples(label: str, limit: int = 5) -> None:
        examples = [record for record in records if record.label == label][:limit]
        if not examples:
            return
        lines.extend(["", f"## Examples: {label}", ""])
        for record in examples:
            lines.append(f"- `{record.task_id}` / `{record.elicitation}`: {record.rationale}")

    _add_examples("error_catch")
    _add_examples("path_inconsistency")
    _add_examples("ambiguous")
    _add_examples("empty_or_unusable", limit=3)
    lines.append("")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    trials = _load_trials(input_path)
    flagged_c_trials = [
        trial
        for trial in trials
        if trial.get("edit_type") == "C_inconsistent" and trial.get("judge_label") == "FLAG"
    ]

    client = LLMClient(
        model=args.model,
        backend=args.backend,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    records: list[AttributionRecord] = []
    for trial in flagged_c_trials:
        records.append(await _classify_trial(client, trial))

    counts = Counter(record.label for record in records)
    nonempty = [record for record in records if record.label != "empty_or_unusable"]
    summary = {
        "input": str(input_path),
        "classifier_model": args.model,
        "backend": args.backend,
        "total_c_flag_trials": len(records),
        "nonempty_c_flag_trials": len(nonempty),
        "counts": dict(counts),
        "shares_all": {label: _share(count, len(records)) for label, count in counts.items()},
        "shares_nonempty": {
            label: _share(count, len(nonempty))
            for label, count in Counter(record.label for record in nonempty).items()
        },
        "records": [record.model_dump() for record in records],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    output_md.write_text(
        _render_markdown(source_path=input_path, model=args.model, records=records),
        encoding="utf-8",
    )

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(
        f"nonempty_error_share={summary['shares_nonempty'].get('error_catch', 0.0):.3f} "
        f"nonempty_path_share={summary['shares_nonempty'].get('path_inconsistency', 0.0):.3f}"
    )
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify why C trials were flagged.")
    parser.add_argument(
        "--input",
        default="outputs/results/cot_edit_introspection_pilot/full_math12_pool20.json",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/results/cot_edit_introspection_pilot/full_math12_pool20_c_flag_attribution.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/results/cot_edit_introspection_pilot/full_math12_pool20_c_flag_attribution.md",
    )
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
