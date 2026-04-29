from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from confidence_tom.data.dataset_models import StaticTask
from confidence_tom.data.scale_dataset import (
    load_aime_2024,
    load_gpqa_diamond,
    load_livebench_reasoning,
    load_math500,
    load_olympiadbench,
)
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient
from confidence_tom.infra.paths import project_root, results_root
from confidence_tom.intervention.voi import trace_to_cost
from experiments.mainline.run.core.run_prefix_oracle_gain_mapping import (
    annotate_segment_count_outliers,
)

ROOT = project_root()
RESULTS_DIR = results_root()
LEGACY_RESULTS_DIR = ROOT / "results"
RESULT_DIR_CANDIDATES = [RESULTS_DIR]
if LEGACY_RESULTS_DIR != RESULTS_DIR:
    RESULT_DIR_CANDIDATES.append(LEGACY_RESULTS_DIR)
DEFAULT_OUT_DIR = RESULTS_DIR / "_prefix_reentry_controls_v1"
DEFAULT_RUN_NAMES = [
    "qwen_to_openai_50",
    "qwen_to_anthropic_50",
    "llama_to_openai_50",
    "llama_to_anthropic_50",
    "mistral_to_openai_50",
    "mistral_to_anthropic_50",
    "livebench_qwen_to_openai_30",
    "livebench_qwen_to_anthropic_30",
    "livebench_llama_to_openai_30",
    "livebench_llama_to_anthropic_30",
    "livebench_mistral_to_openai_30",
    "livebench_mistral_to_anthropic_30",
]
TAXONOMY_PATH = RESULTS_DIR / "_trace_taxonomy_v1" / "trace_taxonomy_summary.json"
OLLAMA_LOCAL_MODEL_BY_FAMILY = {
    "qwen": "qwen3:14b",
    "qwen3": "qwen3:14b",
    "qwen25": "qwen2.5:14b",
    "mistral": "mistral-small3.2:24b",
    "mistralai": "mistral-small3.2:24b",
    "ministral": "mistral-small3.2:24b",
    "mistral7": "mistral:7b-instruct",
    "llama": "llama3.1:8b",
    "meta-llama": "llama3.1:8b",
    "google": "gemma3:4b",
    "gemma": "gemma3:4b",
    "gemma4": "gemma3:4b",
    "gemma3": "gemma3:4b",
    "olmo": "olmo-3.1:32b",
}
LOCAL_MODEL_BY_FAMILY = {
    "qwen": "Qwen/Qwen3-14B",
    "qwen3": "Qwen/Qwen3-14B",
    "qwen25": "Qwen/Qwen2.5-14B-Instruct",
    "mistral": "mistralai/Ministral-8B-Instruct-2410",
    "mistralai": "mistralai/Ministral-8B-Instruct-2410",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama": "meta-llama/Llama-3.1-8B-Instruct",
    "google": "google/gemma-3-4b-it",
    "gemma": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-E4B-it",
    "gemma3": "google/gemma-3-4b-it",
    "olmo": "allenai/OLMo-2-13B-Instruct",
}

FULL_TRACE_SYSTEM_PROMPT = """You are a careful reasoning assistant.
Solve the problem naturally and completely.
Do not use JSON.
End with a final line exactly in the form:
Final Answer: <answer>
"""

REENTRY_SYSTEM_PROMPT = """You are the same small worker continuing an existing reasoning prefix.
Continue naturally from the given prefix and finish the problem.
Do not restart from scratch unless the prefix is unusable.
Do not restate the full problem or rebuild the derivation from the beginning if
the prefix already contains useful structure.
Do not use JSON.
End with a final line exactly in the form:
Final Answer: <answer>
"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _stable_score(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


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


def _answer_or_raw(text: str) -> str:
    answer = _extract_final_answer(text or "")
    if answer.strip():
        return answer.strip()
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_prefix_surface(text: str) -> str:
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _family_from_run_name(run_name: str) -> str:
    if run_name.startswith("reentry_"):
        parts = run_name.split("_")
        if len(parts) >= 3:
            return parts[2]
    if run_name.startswith("livebench_"):
        return run_name.split("_")[1]
    return run_name.split("_")[0]


def _benchmark_from_task_id(task_id: str) -> str:
    if task_id.startswith("livebench_reasoning_"):
        return "livebench_reasoning"
    if task_id.startswith("aime_2024_"):
        return "aime_2024"
    if task_id.startswith("math500_"):
        return "math500"
    if task_id.startswith("gpqa_diamond_"):
        return "gpqa_diamond"
    return "olympiadbench"


def _find_result_json(run_name: str) -> Path:
    excluded_names = {
        "summary.json",
        "dataset_meta.json",
        "baseline_results.json",
        "_run_status.json",
    }
    searched_dirs: list[Path] = []
    for results_dir in RESULT_DIR_CANDIDATES:
        run_dir = results_dir / run_name
        searched_dirs.append(run_dir)
        if not run_dir.exists():
            continue
        candidates = [
            path
            for path in run_dir.glob("*.json")
            if path.name not in excluded_names
        ]
        json_candidates: list[Path] = [
            path
            for path in candidates
            if "per_prefix_rows" not in path.name and not path.name.startswith("_")
        ]
        if json_candidates:
            return json_candidates[0]
    searched = ", ".join(str(path) for path in searched_dirs)
    raise FileNotFoundError(f"Could not find main result JSON for {run_name} in: {searched}")


def _discover_run_names() -> list[str]:
    discovered: dict[str, None] = {}
    for results_dir in RESULT_DIR_CANDIDATES:
        if not results_dir.exists():
            continue
        for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            if run_dir.name.startswith("_"):
                continue
            try:
                _find_result_json(run_dir.name)
            except FileNotFoundError:
                continue
            discovered[run_dir.name] = None
    return list(discovered)


def _load_task_map(benchmark: str) -> dict[str, StaticTask]:
    if benchmark == "olympiadbench":
        tasks = load_olympiadbench(num_samples=50)
    elif benchmark == "livebench_reasoning":
        tasks = load_livebench_reasoning(num_samples=30)
    elif benchmark == "aime_2024":
        tasks = load_aime_2024(num_samples=30)
    elif benchmark == "math500":
        tasks = load_math500(num_samples=50)
    elif benchmark == "gpqa_diamond":
        tasks = load_gpqa_diamond(num_samples=40)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return {task.id: task for task in tasks}


def _load_prefix_rows(run_names: list[str], max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        result_json = _find_result_json(run_name)
        annotate_segment_count_outliers(result_json)
        data = json.loads(result_json.read_text(encoding="utf-8"))
        small_family = _family_from_run_name(run_name)
        for task in data:
            task_id = str(task["task_id"])
            benchmark = _benchmark_from_task_id(task_id)
            prefix_steps = task.get("prefix_oracle_steps", [])
            if not prefix_steps:
                segments = task.get("segments", [])
                prefix_steps = []
                for idx in range(1, len(segments) + 1):
                    prefix_segments = segments[:idx]
                    prefix_text = "\n\n".join(
                        str(segment.get("text", "")).strip()
                        for segment in prefix_segments
                        if str(segment.get("text", "")).strip()
                    ).strip()
                    if not prefix_text:
                        continue
                    prefix_steps.append(
                        {
                            "prefix_id": f"{task_id}_reentry_p{idx}",
                            "step_index": idx,
                            "prefix_text": prefix_text,
                            "small_continue_answer": "",
                            "small_continue_correct": False,
                            "small_continue_text": "",
                            "delta_correctness": 0.0,
                        }
                    )
            for step in prefix_steps:
                prefix_text = str(step.get("prefix_text", "")).strip()
                if not prefix_text:
                    continue
                metadata = task.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                rows.append(
                    {
                        "run_name": run_name,
                        "benchmark": benchmark,
                        "small_family": small_family,
                        "task_id": task_id,
                        "small_model": str(task["small_model"]),
                        "full_trace_answer": str(task.get("full_trace_answer", "")),
                        "full_trace_correct": int(bool(task.get("full_trace_correct", False))),
                        "prefix_id": str(step["prefix_id"]),
                        "step_index": int(step["step_index"]),
                        "prefix_text": prefix_text,
                        "small_continue_answer": str(step.get("small_continue_answer", "")),
                        "small_continue_correct": int(
                            bool(step.get("small_continue_correct", False))
                        ),
                        "small_continue_text": str(step.get("small_continue_text", "")),
                        "delta_correctness": float(step.get("delta_correctness", 0.0)),
                        "positive_gain": int(float(step.get("delta_correctness", 0.0)) > 0.0),
                        "prepare_mode": str(
                            metadata.get("prepare_mode", "oracle_steps")
                        ),
                        "segment_count": int(metadata.get("segment_count", len(segments))),
                        "segment_count_outlier": int(
                            bool(metadata.get("segment_count_outlier", False))
                        ),
                    }
                )
    ordered = sorted(
        rows,
        key=lambda row: _stable_score(
            str(row["run_name"]),
            str(row["task_id"]),
            str(row["prefix_id"]),
        ),
    )
    if max_rows is not None:
        return ordered[:max_rows]
    return ordered


def _slice_rows_by_task(
    rows: list[dict[str, Any]],
    task_start_index: int | None,
    task_limit: int | None,
) -> list[dict[str, Any]]:
    if task_start_index is None and task_limit is None:
        return rows

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["run_name"]), str(row["benchmark"]), str(row["task_id"]))
        grouped[key].append(row)

    ordered_keys = sorted(
        grouped,
        key=lambda item: _stable_score(item[0], item[1], item[2]),
    )
    start = max(0, int(task_start_index or 0))
    end = len(ordered_keys) if task_limit is None else start + max(0, int(task_limit))
    selected_keys = set(ordered_keys[start:end])

    return [
        row
        for row in rows
        if (str(row["run_name"]), str(row["benchmark"]), str(row["task_id"])) in selected_keys
    ]


def _load_taxonomy_categories() -> dict[tuple[str, str, str], str]:
    if not TAXONOMY_PATH.exists():
        return {}
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    records = data.get("records", [])
    categories: dict[tuple[str, str, str], str] = {}
    for record in records:
        key = (
            str(record.get("benchmark", "")),
            str(record.get("task_id", "")),
            str(record.get("family", "")).lower(),
        )
        categories[key] = str(record.get("category", ""))
    return categories


def _parse_family_model_map(entries: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in entries or []:
        item = raw.strip()
        if not item:
            continue
        family, sep, model_name = item.partition("=")
        if not sep or not family.strip() or not model_name.strip():
            raise ValueError(
                f"Invalid --small-local-model-map entry: {raw!r}. Expected FAMILY=MODEL."
            )
        mapping[family.strip().lower()] = model_name.strip()
    return mapping


def _resolve_local_model_name(
    small_family: str,
    explicit: str | None,
    backend: str,
    family_model_map: dict[str, str],
) -> str | None:
    if explicit:
        return explicit
    normalized_family = small_family.lower()
    if normalized_family in family_model_map:
        return family_model_map[normalized_family]
    if backend == "ollama":
        return OLLAMA_LOCAL_MODEL_BY_FAMILY.get(normalized_family)
    if backend == "local":
        return LOCAL_MODEL_BY_FAMILY.get(normalized_family)
    return None


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["run_name"]), str(row["prefix_id"])


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _dedupe_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    keep: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_existing_rows(path):
        key = _row_key(row)
        if key not in keep or ("error" in keep[key] and "error" not in row):
            keep[key] = row
    deduped = list(keep.values())
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in deduped),
        encoding="utf-8",
    )
    return deduped


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _safe_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(int(row[field]) for row in rows) / len(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}

    summary: dict[str, Any] = {
        "rows": len(rows),
        "full_rerun_match_rate": _safe_rate(rows, "full_rerun_matches_original_full"),
        "reentry_match_rate": _safe_rate(rows, "reentry_exact_matches_original_small"),
        "reentry_repeat_match_rate": _safe_rate(rows, "reentry_repeat_matches_first"),
        "marker_boundary_match_rate": _safe_rate(rows, "reentry_marker_matches_exact"),
        "fenced_boundary_match_rate": _safe_rate(rows, "reentry_fenced_matches_exact"),
        "full_trace_success_given_reentry_match": None,
        "full_trace_success_given_reentry_mismatch": None,
        "positive_takeover_given_reentry_match": None,
        "positive_takeover_given_reentry_mismatch": None,
        "by_benchmark": {},
        "by_small_family": {},
    }

    matched = [row for row in rows if int(row["reentry_exact_matches_original_small"]) == 1]
    mismatched = [row for row in rows if int(row["reentry_exact_matches_original_small"]) == 0]
    if matched:
        summary["full_trace_success_given_reentry_match"] = sum(
            int(r["full_trace_correct"]) for r in matched
        ) / len(matched)
        summary["positive_takeover_given_reentry_match"] = sum(
            int(r["positive_gain"]) for r in matched
        ) / len(matched)
    if mismatched:
        summary["full_trace_success_given_reentry_mismatch"] = sum(
            int(r["full_trace_correct"]) for r in mismatched
        ) / len(mismatched)
        summary["positive_takeover_given_reentry_mismatch"] = sum(
            int(r["positive_gain"]) for r in mismatched
        ) / len(mismatched)

    for benchmark, block in _group_rows(rows, "benchmark").items():
        summary["by_benchmark"][benchmark] = {
            "rows": len(block),
            "reentry_match_rate": _safe_rate(block, "reentry_exact_matches_original_small"),
            "full_rerun_match_rate": _safe_rate(block, "full_rerun_matches_original_full"),
            "positive_takeover_given_reentry_match": (
                sum(
                    int(r["positive_gain"])
                    for r in block
                    if int(r["reentry_exact_matches_original_small"]) == 1
                )
                / max(
                    1, sum(1 for r in block if int(r["reentry_exact_matches_original_small"]) == 1)
                )
            ),
            "positive_takeover_given_reentry_mismatch": (
                sum(
                    int(r["positive_gain"])
                    for r in block
                    if int(r["reentry_exact_matches_original_small"]) == 0
                )
                / max(
                    1, sum(1 for r in block if int(r["reentry_exact_matches_original_small"]) == 0)
                )
            ),
        }
    for family, block in _group_rows(rows, "small_family").items():
        summary["by_small_family"][family] = {
            "rows": len(block),
            "reentry_match_rate": _safe_rate(block, "reentry_exact_matches_original_small"),
            "full_rerun_match_rate": _safe_rate(block, "full_rerun_matches_original_full"),
            "positive_takeover_given_reentry_match": (
                sum(
                    int(r["positive_gain"])
                    for r in block
                    if int(r["reentry_exact_matches_original_small"]) == 1
                )
                / max(
                    1, sum(1 for r in block if int(r["reentry_exact_matches_original_small"]) == 1)
                )
            ),
            "positive_takeover_given_reentry_mismatch": (
                sum(
                    int(r["positive_gain"])
                    for r in block
                    if int(r["reentry_exact_matches_original_small"]) == 0
                )
                / max(
                    1, sum(1 for r in block if int(r["reentry_exact_matches_original_small"]) == 0)
                )
            ),
        }
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    if summary["rows"] == 0:
        return "# Prefix Re-entry Controls\n\nNo completed rows.\n"

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    lines = [
        "# Prefix Re-entry Controls",
        "",
        f"- rows: `{summary['rows']}`",
        f"- full rerun match rate: `{fmt(summary['full_rerun_match_rate'])}`",
        f"- re-entry match rate: `{fmt(summary['reentry_match_rate'])}`",
        f"- re-entry repeat match rate: `{fmt(summary['reentry_repeat_match_rate'])}`",
        f"- marker boundary match rate: `{fmt(summary['marker_boundary_match_rate'])}`",
        f"- fenced boundary match rate: `{fmt(summary['fenced_boundary_match_rate'])}`",
        (
            "- P(full-trace success | re-entry match): "
            f"`{fmt(summary['full_trace_success_given_reentry_match'])}`"
        ),
        (
            "- P(full-trace success | re-entry mismatch): "
            f"`{fmt(summary['full_trace_success_given_reentry_mismatch'])}`"
        ),
        (
            "- P(positive takeover | re-entry match): "
            f"`{fmt(summary['positive_takeover_given_reentry_match'])}`"
        ),
        (
            "- P(positive takeover | re-entry mismatch): "
            f"`{fmt(summary['positive_takeover_given_reentry_mismatch'])}`"
        ),
        "",
        "## By Benchmark",
        "",
    ]
    for benchmark, block in summary["by_benchmark"].items():
        lines.append(
            f"- `{benchmark}`: rows={block['rows']}, "
            f"reentry_match={fmt(block['reentry_match_rate'])}, "
            f"full_rerun_match={fmt(block['full_rerun_match_rate'])}, "
            f"p_pos|match={fmt(block['positive_takeover_given_reentry_match'])}, "
            f"p_pos|mismatch={fmt(block['positive_takeover_given_reentry_mismatch'])}"
        )
    lines += ["", "## By Small Family", ""]
    for family, block in summary["by_small_family"].items():
        lines.append(
            f"- `{family}`: rows={block['rows']}, "
            f"reentry_match={fmt(block['reentry_match_rate'])}, "
            f"full_rerun_match={fmt(block['full_rerun_match_rate'])}, "
            f"p_pos|match={fmt(block['positive_takeover_given_reentry_match'])}, "
            f"p_pos|mismatch={fmt(block['positive_takeover_given_reentry_mismatch'])}"
        )
    lines.append("")
    return "\n".join(lines)


def _build_full_trace_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FULL_TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}\n\nSolve it completely."},
    ]


def _build_reentry_messages(question: str, prefix_text: str, variant: str) -> list[dict[str, str]]:
    if variant == "exact":
        content = (
            f"Problem:\n{question}\n\nReasoning prefix:\n{prefix_text}\n\nContinue and finish."
        )
    elif variant == "marker":
        content = (
            f"Problem:\n{question}\n\nReasoning prefix:\n{prefix_text}\n\n"
            "<CONTINUE_FROM_PREFIX>\nContinue from the next step only and finish."
        )
    elif variant == "fenced":
        content = (
            f"Problem:\n{question}\n\n<prefix>\n{prefix_text}\n</prefix>\n\n"
            "Continue after the prefix and finish."
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return [
        {"role": "system", "content": REENTRY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def _generate_one(
    client: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    text, trace = await client.agenerate_text_with_trace(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return text, trace_to_cost(trace).model_dump()


async def _process_one(
    row: dict[str, Any],
    task_map: dict[str, StaticTask],
    client_cache: dict[str, LLMClient],
    *,
    max_tokens: int,
    full_rerun_temperature: float,
    reentry_temperature: float,
    small_backend: str,
    small_local_model_name: str | None,
    small_local_model_map: dict[str, str],
) -> dict[str, Any]:
    task = task_map[str(row["task_id"])]
    evaluator = build_static_evaluator(task)
    model_name = str(row["small_model"])
    small_family = str(row.get("small_family", ""))
    resolved_local_model_name = _resolve_local_model_name(
        small_family,
        small_local_model_name,
        small_backend,
        small_local_model_map,
    )
    if small_backend in {"ollama", "local"} and not resolved_local_model_name:
        raise ValueError(
            f"Missing local model mapping for small_family={small_family!r}, "
            f"backend={small_backend!r} (row small_model={model_name!r})."
        )
    client_key = f"{model_name}|{small_backend}|{small_local_model_name or ''}"
    client = client_cache.setdefault(
        client_key,
        LLMClient(
            model=model_name,
            max_tokens=max_tokens,
            backend=small_backend,
            local_model_name=resolved_local_model_name,
        ),
    )

    original_small_answer = _answer_or_raw(
        str(row["small_continue_text"]) or str(row["small_continue_answer"])
    )
    original_full_answer = _answer_or_raw(str(row["full_trace_answer"]))
    prefix_text = str(row["prefix_text"])

    full_rerun_text, full_rerun_cost = await _generate_one(
        client,
        _build_full_trace_messages(task.question),
        max_tokens=max_tokens,
        temperature=full_rerun_temperature,
    )
    full_rerun_answer = _answer_or_raw(full_rerun_text)
    full_rerun_eval = evaluator(full_rerun_answer, task)

    exact_text, exact_cost = await _generate_one(
        client,
        _build_reentry_messages(task.question, prefix_text, "exact"),
        max_tokens=max_tokens,
        temperature=reentry_temperature,
    )
    exact_answer = _answer_or_raw(exact_text)
    exact_eval = evaluator(exact_answer, task)

    repeat_text, repeat_cost = await _generate_one(
        client,
        _build_reentry_messages(task.question, prefix_text, "exact"),
        max_tokens=max_tokens,
        temperature=reentry_temperature,
    )
    repeat_answer = _answer_or_raw(repeat_text)
    repeat_eval = evaluator(repeat_answer, task)

    marker_text, marker_cost = await _generate_one(
        client,
        _build_reentry_messages(task.question, prefix_text, "marker"),
        max_tokens=max_tokens,
        temperature=reentry_temperature,
    )
    marker_answer = _answer_or_raw(marker_text)
    marker_eval = evaluator(marker_answer, task)

    fenced_text, fenced_cost = await _generate_one(
        client,
        _build_reentry_messages(task.question, _normalize_prefix_surface(prefix_text), "fenced"),
        max_tokens=max_tokens,
        temperature=reentry_temperature,
    )
    fenced_answer = _answer_or_raw(fenced_text)
    fenced_eval = evaluator(fenced_answer, task)

    return {
        **row,
        "execution_host": socket.gethostname(),
        "prefix_tokens_observed": len(_tokenize(prefix_text)),
        "original_small_answer_key": original_small_answer,
        "original_full_answer_key": original_full_answer,
        "full_rerun_answer_key": full_rerun_answer,
        "full_rerun_correct": int(bool(full_rerun_eval.is_correct)),
        "full_rerun_matches_original_full": int(full_rerun_answer == original_full_answer),
        "full_rerun_cost": full_rerun_cost,
        "reentry_exact_answer_key": exact_answer,
        "reentry_exact_correct": int(bool(exact_eval.is_correct)),
        "reentry_exact_matches_original_small": int(exact_answer == original_small_answer),
        "reentry_exact_cost": exact_cost,
        "reentry_repeat_answer_key": repeat_answer,
        "reentry_repeat_correct": int(bool(repeat_eval.is_correct)),
        "reentry_repeat_matches_first": int(repeat_answer == exact_answer),
        "reentry_repeat_cost": repeat_cost,
        "reentry_marker_answer_key": marker_answer,
        "reentry_marker_correct": int(bool(marker_eval.is_correct)),
        "reentry_marker_matches_exact": int(marker_answer == exact_answer),
        "reentry_marker_cost": marker_cost,
        "reentry_fenced_answer_key": fenced_answer,
        "reentry_fenced_correct": int(bool(fenced_eval.is_correct)),
        "reentry_fenced_matches_exact": int(fenced_answer == exact_answer),
        "reentry_fenced_cost": fenced_cost,
    }


async def amain(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows = output_dir / "reentry_rows.jsonl"
    out_summary = output_dir / "reentry_summary.json"
    out_md = (
        ROOT
        / "docs"
        / "mainline"
        / "generated"
        / "analysis"
        / "prefix"
        / "prefix_reentry_controls.md"
    )
    lock_path = output_dir / ".reentry.lock"

    lock_file = lock_path.open("w", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another re-entry control process is already running: {lock_path}")

    run_names = args.run_name or DEFAULT_RUN_NAMES
    if args.run_name is None:
        missing_defaults = [
            run_name
            for run_name in run_names
            if not any((results_dir / run_name).exists() for results_dir in RESULT_DIR_CANDIDATES)
        ]
        if missing_defaults:
            discovered = _discover_run_names()
            if discovered:
                run_names = discovered
    if not run_names:
        raise FileNotFoundError(
            "No result runs found under any configured results directory. "
            "Pass --run-name or create outputs/results/<run_name>."
        )

    wanted_prefixes = [prefix.strip() for prefix in args.run_name_prefix if prefix.strip()]
    if wanted_prefixes:
        run_names = [
            run_name
            for run_name in run_names
            if any(run_name.startswith(prefix) for prefix in wanted_prefixes)
        ]

    if not run_names:
        raise FileNotFoundError(
            "No result runs matched the requested re-entry filters. "
            "Adjust --run-name / --run-name-prefix."
        )

    existing_rows = _dedupe_rows(out_rows)
    done = {_row_key(row) for row in existing_rows if "error" not in row}
    all_rows = _load_prefix_rows(run_names, None)
    all_rows = _slice_rows_by_task(all_rows, args.task_start_index, args.task_limit)
    if args.max_rows is not None:
        all_rows = all_rows[: args.max_rows]
    pending = [row for row in all_rows if _row_key(row) not in done]

    if args.benchmark:
        wanted_benchmarks = {item.strip() for item in args.benchmark if item.strip()}
        pending = [row for row in pending if str(row["benchmark"]) in wanted_benchmarks]

    if args.small_family:
        wanted_families = {item.strip().lower() for item in args.small_family if item.strip()}
        pending = [row for row in pending if str(row["small_family"]).lower() in wanted_families]

    if args.category:
        taxonomy = _load_taxonomy_categories()
        wanted = {cat.strip() for cat in args.category if cat.strip()}
        pending = [
            row
            for row in pending
            if taxonomy.get(
                (str(row["benchmark"]), str(row["task_id"]), str(row["small_family"]).lower())
            )
            in wanted
        ]

    if args.exclude_segment_count_outliers:
        pending = [row for row in pending if int(row.get("segment_count_outlier", 0)) == 0]

    needed_benchmarks = {str(row["benchmark"]) for row in pending}
    task_maps = {benchmark: _load_task_map(benchmark) for benchmark in sorted(needed_benchmarks)}
    client_cache: dict[str, LLMClient] = {}
    sem = asyncio.Semaphore(args.concurrency)
    small_local_model_map = _parse_family_model_map(args.small_local_model_map)

    async def worker(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _process_one(
                row,
                task_maps[str(row["benchmark"])],
                client_cache,
                max_tokens=args.max_tokens,
                full_rerun_temperature=args.full_rerun_temperature,
                reentry_temperature=args.reentry_temperature,
                small_backend=args.small_backend,
                small_local_model_name=args.small_local_model_name,
                small_local_model_map=small_local_model_map,
            )

    with out_rows.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(pending, start=1):
            try:
                result = await worker(row)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"processed {idx}/{len(pending)} :: {row['run_name']} :: {row['prefix_id']}")
            except Exception as exc:
                error_row = dict(row)
                error_row["error"] = repr(exc)
                f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"error {idx}/{len(pending)} :: {row['run_name']} :: "
                    f"{row['prefix_id']} :: {exc}"
                )

    rows = [row for row in _dedupe_rows(out_rows) if "error" not in row]
    summary = _summarize(rows)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote rows to {out_rows}")
    print(f"Wrote summary to {out_summary}")
    print(f"Wrote markdown to {out_md}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prefix re-entry stability controls.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument(
        "--run-name-prefix",
        action="append",
        default=[],
        help="Filter discovered runs by prefix (repeatable).",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Filter rows by benchmark name (repeatable).",
    )
    parser.add_argument(
        "--small-family",
        action="append",
        default=[],
        help="Filter rows by small-model family (repeatable).",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Filter rows by taxonomy category (repeatable).",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--task-start-index",
        type=int,
        default=None,
        help="Start index into the ordered task list for re-entry processing.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Maximum number of tasks to process for re-entry after task-start-index.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--full-rerun-temperature", type=float, default=0.0)
    parser.add_argument("--reentry-temperature", type=float, default=0.0)
    parser.add_argument(
        "--small-backend", default="local", choices=["ollama", "local"]
    )
    parser.add_argument("--small-local-model-name", default=None)
    parser.add_argument(
        "--small-local-model-map",
        action="append",
        default=[],
        help="Per-family local model override in FAMILY=MODEL form (repeatable).",
    )
    parser.add_argument(
        "--exclude-segment-count-outliers",
        action="store_true",
        help="Skip prepare tasks whose segment count is a robust high outlier within its run.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(amain(args))
