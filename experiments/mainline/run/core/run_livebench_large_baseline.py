from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from confidence_tom.data.scale_dataset import load_livebench_reasoning
from confidence_tom.eval.static_evaluators import build_static_evaluator
from confidence_tom.infra.client import LLMClient
from confidence_tom.intervention import trace_to_cost
from experiments.mainline.run.core.run_prefix_oracle_gain_mapping import (
    _FULL_TRACE_SYSTEM_PROMPT,
    _extract_answer_with_fallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def _provider_for_model(model: str) -> dict[str, Any] | None:
    if model.startswith("openai/"):
        return {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    if model.startswith("anthropic/"):
        return {"only": ["anthropic"], "allow_fallbacks": False, "require_parameters": True}
    return None


def _sanitize_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    tmp.replace(path)


async def _run_one(
    task: Any, client: LLMClient, extract_client: LLMClient | None
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _FULL_TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{task.question}"},
    ]
    raw_text, trace = await client.agenerate_text_with_trace(messages)
    answer = await _extract_answer_with_fallback(raw_text, extract_client)
    evaluator = build_static_evaluator(task)
    eval_result = evaluator(answer, task)
    return {
        "task_id": task.id,
        "benchmark": "livebench_reasoning",
        "model": client.model,
        "question": task.question,
        "reference_answer": task.reference_answer,
        "response_text": raw_text,
        "parsed_answer": answer,
        "correct": bool(eval_result.is_correct),
        "api_trace": trace.model_dump(),
        "cost": trace_to_cost(trace, None).model_dump(),
    }


async def _main(args: argparse.Namespace) -> None:
    tasks = load_livebench_reasoning(num_samples=args.limit)
    if args.task_ids:
        wanted = set(args.task_ids.split(","))
        tasks = [task for task in tasks if task.id in wanted]
    if args.offset:
        tasks = tasks[args.offset :]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    provider = _provider_for_model(args.model) if args.provider_lock else None
    client = LLMClient(
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        backend="openrouter",
        provider=provider,
        seed=args.seed,
    )
    extract_client = None
    if args.extractor_model:
        extract_client = LLMClient(
            model=args.extractor_model,
            temperature=0.0,
            max_tokens=512,
            backend="openrouter",
            provider=_provider_for_model(args.extractor_model) if args.provider_lock else None,
        )

    output_path = Path(args.output)
    rows = _load_rows(output_path)
    done = {row["task_id"] for row in rows}

    for i, task in enumerate(tasks, start=1):
        if task.id in done:
            logger.info("skip completed task=%s", task.id)
            continue
        logger.info("run %s task %d/%d id=%s", args.model, i, len(tasks), task.id)
        row = await _run_one(task, client, extract_client)
        rows.append(row)
        _save_rows(output_path, rows)

    correct = sum(int(bool(row["correct"])) for row in rows)
    logger.info(
        "saved %s rows=%d correct=%d rate=%.3f",
        output_path,
        len(rows),
        correct,
        correct / len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run direct from-scratch large baseline on livebench."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--provider-lock", action="store_true")
    parser.add_argument("--extractor-model", default="google/gemini-3.1-flash-lite-preview")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
