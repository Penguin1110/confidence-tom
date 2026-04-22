from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import load_livebench_reasoning, load_olympiadbench
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.paths import project_root, results_root

ROOT = project_root()
RESULTS_DIR = results_root()
OUT_DIR = RESULTS_DIR / "_prefix_fragility_v1"
OUT_ROWS = OUT_DIR / "pilot_rows.jsonl"
OUT_SUMMARY = OUT_DIR / "pilot_summary.json"
OUT_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "prefix" / "prefix_fragility_pilot.md"
)
LOCK_PATH = OUT_DIR / ".pilot.lock"

MAX_PER_RUN = 2
CONCURRENCY = 4
REWRITE_MODEL = "openai/gpt-5.4"
REWRITE_MAX_TOKENS = 600
CONTINUE_MAX_TOKENS = 1024

RUN_SPECS = {
    "qwen_to_openai_50": ("olympiadbench", "qwen/qwen3-14b:nitro"),
    "qwen_to_anthropic_50": ("olympiadbench", "qwen/qwen3-14b:nitro"),
    "llama_to_openai_50": ("olympiadbench", "meta-llama/llama-4-scout"),
    "llama_to_anthropic_50": ("olympiadbench", "meta-llama/llama-4-scout"),
    "mistral_to_openai_50": ("olympiadbench", "mistralai/ministral-8b-2512"),
    "mistral_to_anthropic_50": ("olympiadbench", "mistralai/ministral-8b-2512"),
    "livebench_qwen_to_openai_30": ("livebench_reasoning", "qwen/qwen3-14b:nitro"),
    "livebench_qwen_to_anthropic_30": ("livebench_reasoning", "qwen/qwen3-14b:nitro"),
    "livebench_llama_to_openai_30": ("livebench_reasoning", "meta-llama/llama-4-scout"),
    "livebench_llama_to_anthropic_30": ("livebench_reasoning", "meta-llama/llama-4-scout"),
    "livebench_mistral_to_openai_30": ("livebench_reasoning", "mistralai/ministral-8b-2512"),
    "livebench_mistral_to_anthropic_30": ("livebench_reasoning", "mistralai/ministral-8b-2512"),
}

_REWRITE_SYSTEM = """You rewrite a reasoning prefix while preserving its meaning exactly.

Hard constraints:
- Do not add any new deduction, fact, assumption, or conclusion.
- Do not remove any deduction, fact, assumption, or conclusion.
- Do not strengthen or weaken certainty, commitment, or epistemic force.
- Do not convert an analysis goal into an established fact.
- Do not repair mistakes in the reasoning.
- Do not reorder steps unless the order is unchanged in meaning.
- Keep all mathematical and logical claims exactly equivalent.

Allowed edits:
- Minor wording substitutions.
- Light sentence smoothing.
- Small formatting cleanup.

Return only the rewritten prefix. If you cannot safely rewrite it under these
constraints, return the original prefix unchanged."""

_CONTINUE_SYSTEM = """You are the same small worker continuing an existing reasoning prefix.
Continue naturally from the given prefix and finish the problem.
Do not restart from scratch unless the prefix is unusable.
End with a final line exactly in the form:
Final Answer: <answer>"""


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""

    matches = re.findall(r"(?im)^(?:the\s+)?final answer:\s*(.+?)\s*$", text)
    if matches:
        return str(matches[-1]).strip()

    boxed = re.findall(r"\\boxed\{(.+?)\}", text, flags=re.DOTALL)
    if boxed:
        return str(boxed[-1]).strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    return (
        str(last)
        if re.search(r"(sqrt|\\sqrt|\\boxed|boxed|[0-9]|yes|no)", last, flags=re.I)
        else ""
    )


def _stable_score(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        path
        for path in run_dir.glob("*.json")
        if path.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
    ]
    json_candidates = [path for path in candidates if "per_prefix_rows" not in path.name]
    if not json_candidates:
        raise FileNotFoundError(f"Could not find main result JSON in {run_dir}")
    return json_candidates[0]


def _normalize_prefix_surface(text: str) -> str:
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _load_task_map(benchmark: str) -> dict[str, StaticTask]:
    if benchmark == "olympiadbench":
        tasks = load_olympiadbench(num_samples=50)
    elif benchmark == "livebench_reasoning":
        tasks = load_livebench_reasoning(num_samples=30)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return {task.id: task for task in tasks}


def _extract_answer_or_raw(text: str) -> str:
    answer = _extract_final_answer(text or "")
    return answer.strip() if answer.strip() else (text or "").strip()


def _load_existing_rows() -> set[tuple[str, str]]:
    if not OUT_ROWS.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in OUT_ROWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add((str(row["run_name"]), str(row["prefix_id"])))
    return seen


def _dedupe_rows() -> list[dict[str, object]]:
    if not OUT_ROWS.exists():
        return []
    keep: dict[tuple[str, str], dict[str, object]] = {}
    for line in OUT_ROWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["run_name"]), str(row["prefix_id"]))
        # Prefer successful rows over error rows when duplicates exist.
        if key not in keep or ("error" in keep[key] and "error" not in row):
            keep[key] = row
    rows = list(keep.values())
    OUT_ROWS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _sample_prefix_rows() -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for run_name in sorted(RUN_SPECS):
        benchmark, _small_model = RUN_SPECS[run_name]
        result_json = _find_result_json(run_name)
        data = json.loads(result_json.read_text(encoding="utf-8"))
        pool: list[dict[str, object]] = []
        for task_row in data:
            for step in task_row.get("prefix_oracle_steps", []):
                prefix_text = str(step.get("prefix_text", "")).strip()
                if not prefix_text:
                    continue
                pool.append(
                    {
                        "run_name": run_name,
                        "benchmark": benchmark,
                        "task_id": task_row["task_id"],
                        "prefix_id": step["prefix_id"],
                        "step_index": step["step_index"],
                        "prefix_text": prefix_text,
                        "original_small_continue_text": step.get("small_continue_text", ""),
                        "original_small_continue_correct": bool(
                            step.get("small_continue_correct", False)
                        ),
                        "delta_correctness": float(step.get("delta_correctness", 0.0)),
                    }
                )
        ordered = sorted(
            pool,
            key=lambda row: _stable_score(
                str(row["run_name"]),
                str(row["task_id"]),
                str(row["prefix_id"]),
            ),
        )
        selected.extend(ordered[:MAX_PER_RUN])
    return selected


async def _rewrite_prefix(client: LLMClient, question: str, prefix_text: str) -> str:
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\n"
                f"Reasoning prefix to rewrite:\n{prefix_text}\n\n"
                "Rewrite the prefix so it says exactly the same thing, with the "
                "same level of certainty, but with slightly cleaner surface "
                "wording. If there is any risk of changing meaning, "
                "return the original text unchanged."
            ),
        },
    ]
    text = await client.agenerate_text(messages, max_tokens=REWRITE_MAX_TOKENS, temperature=0.0)
    return text.strip() or prefix_text


async def _continue_from_prefix(client: LLMClient, question: str, prefix_text: str) -> str:
    messages = [
        {"role": "system", "content": _CONTINUE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\nReasoning prefix:\n{prefix_text}\n\nContinue and finish."
            ),
        },
    ]
    return await client.agenerate_text(
        messages,
        max_tokens=CONTINUE_MAX_TOKENS,
        temperature=0.0,
    )


async def _process_one(
    row: dict[str, Any],
    task_map: dict[str, StaticTask],
    rewrite_client: LLMClient,
    sem: asyncio.Semaphore,
) -> dict[str, object]:
    async with sem:
        benchmark, small_model = RUN_SPECS[str(row["run_name"])]
        task = task_map[str(row["task_id"])]
        evaluator = build_static_evaluator(task)
        small_client = LLMClient(model=small_model, max_tokens=CONTINUE_MAX_TOKENS)

        prefix_text = str(row["prefix_text"])
        normalized_prefix = _normalize_prefix_surface(prefix_text)
        rewritten_prefix = await _rewrite_prefix(rewrite_client, task.question, prefix_text)

        norm_text = await _continue_from_prefix(small_client, task.question, normalized_prefix)
        rewrite_text = await _continue_from_prefix(small_client, task.question, rewritten_prefix)

        norm_answer = _extract_answer_or_raw(norm_text)
        rewrite_answer = _extract_answer_or_raw(rewrite_text)
        norm_eval = evaluator(norm_answer, task)
        rewrite_eval = evaluator(rewrite_answer, task)

        result = dict(row)
        result.update(
            {
                "normalized_prefix": normalized_prefix,
                "rewritten_prefix": rewritten_prefix,
                "normalized_answer": norm_answer,
                "rewritten_answer": rewrite_answer,
                "normalized_correct": bool(norm_eval.is_correct),
                "rewritten_correct": bool(rewrite_eval.is_correct),
                "normalized_answer_changed": int(
                    norm_answer != _extract_answer_or_raw(str(row["original_small_continue_text"]))
                ),
                "rewritten_answer_changed": int(
                    rewrite_answer
                    != _extract_answer_or_raw(str(row["original_small_continue_text"]))
                ),
                "normalized_correctness_changed": int(
                    bool(norm_eval.is_correct) != bool(row["original_small_continue_correct"])
                ),
                "rewritten_correctness_changed": int(
                    bool(rewrite_eval.is_correct) != bool(row["original_small_continue_correct"])
                ),
            }
        )
        return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"rows": 0}
    return {
        "rows": total,
        "normalized_correctness_changed_rate": sum(
            int(r["normalized_correctness_changed"]) for r in rows
        )
        / total,
        "rewritten_correctness_changed_rate": sum(
            int(r["rewritten_correctness_changed"]) for r in rows
        )
        / total,
        "normalized_answer_changed_rate": sum(int(r["normalized_answer_changed"]) for r in rows)
        / total,
        "rewritten_answer_changed_rate": sum(int(r["rewritten_answer_changed"]) for r in rows)
        / total,
        "by_benchmark": {
            benchmark: {
                "rows": len(block),
                "rewritten_correctness_changed_rate": sum(
                    int(r["rewritten_correctness_changed"]) for r in block
                )
                / len(block),
                "normalized_correctness_changed_rate": sum(
                    int(r["normalized_correctness_changed"]) for r in block
                )
                / len(block),
            }
            for benchmark, block in _group_by(rows, "benchmark").items()
        },
        "by_small_family": {
            family: {
                "rows": len(block),
                "rewritten_correctness_changed_rate": sum(
                    int(r["rewritten_correctness_changed"]) for r in block
                )
                / len(block),
                "normalized_correctness_changed_rate": sum(
                    int(r["normalized_correctness_changed"]) for r in block
                )
                / len(block),
            }
            for family, block in _group_by(rows, "run_name", family_only=True).items()
        },
    }


def _group_by(
    rows: list[dict[str, Any]], key: str, family_only: bool = False
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if family_only:
            value = str(row["run_name"]).split("_")[0].replace("livebench_", "")
        else:
            value = str(row[key])
        grouped[value].append(row)
    return grouped


def _to_markdown(summary: dict[str, object]) -> str:
    by_benchmark = cast(dict[str, dict[str, Any]], summary["by_benchmark"])
    by_small_family = cast(dict[str, dict[str, Any]], summary["by_small_family"])
    lines = [
        "# Prefix Fragility Pilot",
        "",
        f"- rows: `{summary['rows']}`",
        (
            "- normalized correctness changed rate: "
            f"`{summary['normalized_correctness_changed_rate']:.3f}`"
        ),
        (
            "- rewritten correctness changed rate: "
            f"`{summary['rewritten_correctness_changed_rate']:.3f}`"
        ),
        f"- normalized answer changed rate: `{summary['normalized_answer_changed_rate']:.3f}`",
        f"- rewritten answer changed rate: `{summary['rewritten_answer_changed_rate']:.3f}`",
        "",
        "## By Benchmark",
        "",
    ]
    for benchmark, block in by_benchmark.items():
        lines.append(
            f"- `{benchmark}`: rows={block['rows']}, "
            f"rewrite_correctness_change={block['rewritten_correctness_changed_rate']:.3f}, "
            f"normalize_correctness_change={block['normalized_correctness_changed_rate']:.3f}"
        )
    lines += ["", "## By Small Family", ""]
    for family, block in by_small_family.items():
        lines.append(
            f"- `{family}`: rows={block['rows']}, "
            f"rewrite_correctness_change={block['rewritten_correctness_changed_rate']:.3f}, "
            f"normalize_correctness_change={block['normalized_correctness_changed_rate']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


async def amain() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another fragility pilot process is already running: {LOCK_PATH}")
        return

    existing_rows = _dedupe_rows()
    sampled = _sample_prefix_rows()
    existing = {(str(row["run_name"]), str(row["prefix_id"])) for row in existing_rows}
    todo = [row for row in sampled if (str(row["run_name"]), str(row["prefix_id"])) not in existing]

    task_maps = {
        "olympiadbench": _load_task_map("olympiadbench"),
        "livebench_reasoning": _load_task_map("livebench_reasoning"),
    }
    rewrite_client = LLMClient(model=REWRITE_MODEL, max_tokens=REWRITE_MAX_TOKENS)
    sem = asyncio.Semaphore(CONCURRENCY)

    with OUT_ROWS.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(todo, start=1):
            try:
                result = await _process_one(
                    row,
                    task_maps[str(row["benchmark"])],
                    rewrite_client,
                    sem,
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"processed {idx}/{len(todo)}")
            except Exception as exc:
                error_row = dict(row)
                error_row["error"] = repr(exc)
                f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"error {idx}/{len(todo)}: {exc}")

    rows = [row for row in _dedupe_rows() if "error" not in row]
    summary = _summarize(rows)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote rows to {OUT_ROWS}")
    print(f"Wrote summary to {OUT_SUMMARY}")
    print(f"Wrote markdown to {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(amain())
