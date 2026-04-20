from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from confidence_tom.client import LLMClient
from confidence_tom.scale_dataset import load_livebench_reasoning, load_olympiadbench


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _parse_provider(provider_only: str | None) -> dict[str, Any] | None:
    if not provider_only:
        return None
    return {
        "only": [provider_only],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def _load_question(
    benchmark: str | None, task_index: int | None, direct_question: str | None
) -> tuple[str, dict[str, Any]]:
    if direct_question:
        return direct_question, {"source": "direct"}
    if benchmark == "olympiadbench":
        idx = int(task_index or 0)
        task = load_olympiadbench(num_samples=idx + 1)[idx]
        return task.question, {"source": benchmark, "task_id": task.id}
    if benchmark == "livebench_reasoning":
        idx = int(task_index or 0)
        task = load_livebench_reasoning(num_samples=idx + 1)[idx]
        return task.question, {"source": benchmark, "task_id": task.id}
    raise ValueError("Provide either --question or a supported --benchmark with --task-index")


async def _run_once(
    *,
    client: LLMClient,
    messages: list[dict[str, str]],
    request_payload: dict[str, Any],
    run_index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    text, trace = await client.agenerate_text_with_trace(messages)
    elapsed = time.perf_counter() - start
    response_text = text or ""
    trace_payload = trace.model_dump()
    return {
        "run_index": run_index,
        "elapsed_sec": elapsed,
        "request_payload_hash": _sha256_text(_canonical_json(request_payload)),
        "response_text_hash": _sha256_text(response_text),
        "response_text": response_text,
        "request_id": trace_payload.get("request_id", ""),
        "model_id": trace_payload.get("model_id", ""),
        "prompt_tokens": trace_payload.get("prompt_tokens", 0),
        "completion_tokens": trace_payload.get("completion_tokens", 0),
        "total_tokens": trace_payload.get("total_tokens", 0),
        "reasoning_tokens": trace_payload.get("reasoning_tokens", 0),
        "cache_read_tokens": trace_payload.get("cache_read_tokens", 0),
        "cache_write_tokens": trace_payload.get("cache_write_tokens", 0),
        "api_trace": trace_payload,
    }


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    question, source_meta = _load_question(args.benchmark, args.task_index, args.question)
    system_prompt = args.system_prompt.strip()
    messages = _build_messages(question, system_prompt)
    client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        backend=args.backend,
        local_model_name=args.local_model_name,
        seed=args.seed,
        provider=_parse_provider(args.provider_only),
    )

    request_payload = {
        "model": args.local_model_name
        if args.backend == "ollama" and args.local_model_name
        else args.model,
        "backend": args.backend,
        "canonical_model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "provider_only": args.provider_only,
        "messages": messages,
    }

    rows: list[dict[str, Any]] = []
    for run_index in range(1, args.repeats + 1):
        row = await _run_once(
            client=client,
            messages=messages,
            request_payload=request_payload,
            run_index=run_index,
        )
        rows.append(row)

    response_hashes = [row["response_text_hash"] for row in rows]
    unique_response_hashes = sorted(set(response_hashes))
    request_ids = [row["request_id"] for row in rows if row["request_id"]]
    model_ids = [row["model_id"] for row in rows if row["model_id"]]

    return {
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed,
            "provider_only": args.provider_only,
            "repeats": args.repeats,
            "benchmark": args.benchmark,
            "task_index": args.task_index,
            "system_prompt": system_prompt,
        },
        "source": source_meta,
        "request_payload": request_payload,
        "request_payload_hash": _sha256_text(_canonical_json(request_payload)),
        "summary": {
            "repeat_count": len(rows),
            "unique_response_hash_count": len(unique_response_hashes),
            "all_request_payload_hash_same": len({row["request_payload_hash"] for row in rows})
            == 1,
            "all_response_hash_same": len(unique_response_hashes) == 1,
            "unique_response_hashes": unique_response_hashes,
            "unique_model_ids": sorted(set(model_ids)),
            "request_ids_present": len(request_ids),
        },
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether repeated identical API requests return identical outputs."
    )
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--local-model-name", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--provider-only", default=None, help="OpenRouter provider slug to pin to (e.g. openai)"
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--benchmark", choices=["olympiadbench", "livebench_reasoning"], default=None
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a careful reasoning assistant. "
            "Solve the problem and end with a final line exactly in the form: Final Answer: <answer>"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_main_async(args))
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
