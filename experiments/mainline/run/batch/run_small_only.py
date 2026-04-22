from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[3]
CORE_RUNNER_DIR = ROOT / "experiments" / "mainline" / "run" / "core"
for candidate in (ROOT / "src", ROOT, CORE_RUNNER_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from run_prefix_oracle_gain_mapping import (  # noqa: E402
    _SMALL_CONTINUE_SYSTEM_PROMPT,
    PartialTaskStore,
    ResultStore,
    _client_kwargs_from_cfg,
    _pricing_from_cfg,
    _run_continue,
    _run_full_trace,
    _segment_full_trace,
)

from confidence_tom.data.scale_dataset import (  # noqa: E402
    load_livebench_reasoning,
    load_olympiadbench,
)
from confidence_tom.eval.static_evaluators import build_static_evaluator  # noqa: E402
from confidence_tom.infra.client import LLMClient  # noqa: E402
from confidence_tom.intervention import (  # noqa: E402
    PrefixOracleGainStepResult,
    PrefixOracleGainTaskResult,
    PrefixSegment,
    trace_to_cost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    "qwen3_8b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:8b",
        "small_local_model_name": "qwen3:8b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen3_14b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:14b",
        "small_local_model_name": "qwen3:14b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen3_32b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3:32b",
        "small_local_model_name": "qwen3:32b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 3072,
    },
    "qwen35_9b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:9b",
        "small_local_model_name": "qwen3.5:9b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 4096,
    },
    "qwen35_4b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:4b",
        "small_local_model_name": "qwen3.5:4b",
        "small_max_tokens": 12288,
        "num_ctx": 16384,
        "num_predict": 3072,
    },
    "qwen35_4b_fast": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:4b_fast",
        "small_local_model_name": "qwen3.5:4b_fast",
        "small_max_tokens": 12288,
        "num_ctx": 16384,
        "num_predict": 3072,
    },
    "qwen35_27b": {
        "small_model": "qwen/qwen3-14b:nitro",
        "small_label": "Ollama-qwen3.5:27b",
        "small_local_model_name": "qwen3.5:27b",
        "small_max_tokens": 12288,
        "num_ctx": 32768,
        "num_predict": 3072,
    },
    "gemma3_12b": {
        "small_model": "google/gemma-3-4b-it",
        "small_label": "Ollama-gemma3:12b",
        "small_local_model_name": "gemma3:12b",
        "small_max_tokens": 8192,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "gemma3_4b": {
        "small_model": "google/gemma-3-4b-it",
        "small_label": "Ollama-gemma3:4b",
        "small_local_model_name": "gemma3:4b",
        "small_max_tokens": 8192,
        "num_ctx": 16384,
        "num_predict": 3072,
    },
    "gemma3_27b": {
        "small_model": "google/gemma-3-4b-it",
        "small_label": "Ollama-gemma3:27b",
        "small_local_model_name": "gemma3:27b",
        "small_max_tokens": 8192,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "mistral_small_24b": {
        "small_model": "mistralai/ministral-8b-2512",
        "small_label": "Ollama-mistral-small3.2:24b",
        "small_local_model_name": "mistral-small3.2:24b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "llama31_8b": {
        "small_model": "meta-llama/llama-4-scout",
        "small_label": "Ollama-llama3.1:8b",
        "small_local_model_name": "llama3.1:8b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
    "olmo31_32b": {
        "small_model": "allenai/olmo-2-13b-instruct",
        "small_label": "Ollama-olmo-3.1:32b",
        "small_local_model_name": "olmo-3.1:32b",
        "small_max_tokens": 12288,
        "num_ctx": 24576,
        "num_predict": 3072,
    },
}

LOCAL_MODEL_PRESETS: dict[str, dict[str, str | int]] = {
    "qwen25_7b": {
        "small_model": "Qwen/Qwen2.5-7B-Instruct",
        "small_label": "HF-Qwen2.5-7B-Instruct",
        "small_local_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "small_max_tokens": 8192,
        "num_ctx": 16384,
        "num_predict": 1024,
    },
    "qwen25_14b": {
        "small_model": "Qwen/Qwen2.5-14B-Instruct",
        "small_label": "HF-Qwen2.5-14B-Instruct",
        "small_local_model_name": "Qwen/Qwen2.5-14B-Instruct",
        "small_max_tokens": 8192,
        "num_ctx": 16384,
        "num_predict": 1024,
    },
    "gemma2_9b": {
        "small_model": "google/gemma-2-9b-it",
        "small_label": "HF-gemma-2-9b-it",
        "small_local_model_name": "google/gemma-2-9b-it",
        "small_max_tokens": 6144,
        "num_ctx": 12288,
        "num_predict": 1024,
    },
    "mistral7b_v03": {
        "small_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "small_label": "HF-Mistral-7B-Instruct-v0.3",
        "small_local_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "small_max_tokens": 6144,
        "num_ctx": 12288,
        "num_predict": 1024,
    },
}


def _run(cmd: list[str]) -> None:
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_optional(cmd: list[str]) -> int:
    print("$", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _build_cfg(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "output_dir": str(Path(args.output_dir).resolve()),
            "dataset": {
                "benchmark": args.benchmark,
                "limit": args.limit,
                "olympiadbench": args.limit if args.benchmark == "olympiadbench" else 0,
                "livebench": args.limit if args.benchmark == "livebench_reasoning" else 0,
                "livebench_reasoning": args.limit if args.benchmark == "livebench_reasoning" else 0,
            },
            "execution": {
                "task_concurrency": args.task_concurrency,
                "retry_attempts": args.retry_attempts,
                "retry_backoff_sec": args.retry_backoff_sec,
                "resume_from_partials": True,
                "retain_partials": True,
            },
            "timeouts": {
                "full_trace_sec": args.full_trace_sec,
                "small_worker_sec": args.small_worker_sec,
                "task_sec": args.task_sec,
            },
            "small_worker": {
                "model": args.small_model,
                "label": args.small_label,
                "max_tokens": args.small_max_tokens,
                "temperature": args.temperature,
                "backend": args.small_backend,
                "local_model_name": args.small_local_model_name,
                "reasoning_effort": args.reasoning_effort,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "enable_thinking": args.enable_thinking,
            },
            "large_worker": {
                "model": args.placeholder_large_model,
                "label": args.placeholder_large_label,
                "max_tokens": 1,
            },
            "extractor": {
                "enabled": bool(args.extractor_enabled),
                "model": args.extractor_model,
                "label": "Extractor",
                "max_tokens": args.extractor_max_tokens,
                "temperature": 0.0,
                "backend": args.extractor_backend,
            },
            "pricing": {},
        }
    )


def _trace_payload(trace: Any | None) -> Any:
    if trace is None:
        return None
    return cast(Any, trace).model_dump()


async def _map_task_small_only(task: Any, cfg: Any) -> PrefixOracleGainTaskResult:
    partial_store = PartialTaskStore(Path(str(cfg.output_dir)) / "partials")
    small_client = LLMClient(**_client_kwargs_from_cfg(cfg.small_worker))
    extract_client = None
    if bool(cfg.extractor.enabled):
        extract_client = LLMClient(**_client_kwargs_from_cfg(cfg.extractor))
    evaluator = build_static_evaluator(task)
    execution_cfg = cfg.execution
    retry_attempts = int(execution_cfg.retry_attempts)
    retry_backoff_sec = float(execution_cfg.retry_backoff_sec)

    trace_id = f"{task.id}_smallonly"
    resume_payload = partial_store.load(task.id)
    full_text = ""
    full_answer = ""
    full_trace_api = None
    parse_incomplete = False
    segments: list[dict[str, Any]] = []
    oracle_steps: list[PrefixOracleGainStepResult] = []

    if resume_payload and resume_payload.get("status") in {"full_trace_done", "prefix_step_done"}:
        trace_id = str(resume_payload.get("trace_id") or trace_id)
        full_text = str(resume_payload.get("full_trace_text") or "")
        full_answer = str(resume_payload.get("full_trace_answer") or "")
        parse_incomplete = bool(resume_payload.get("parse_incomplete", False))
        segments = resume_payload.get("segments", [])
        oracle_steps = [
            PrefixOracleGainStepResult.model_validate(s)
            for s in resume_payload.get("prefix_oracle_steps", [])
        ]
        full_trace_api = resume_payload.get("full_trace_api_trace")
        logger.info("partial.resume task=%s completed_steps=%d", task.id, len(oracle_steps))
    else:
        full_text, full_answer, full_trace_api = await _run_full_trace(
            task,
            small_client,
            extract_client,
            int(cfg.timeouts.full_trace_sec),
            retry_attempts,
            retry_backoff_sec,
        )
        segs, parsed_final_answer, parse_incomplete = await _segment_full_trace(
            full_text, extract_client, task.id
        )
        segments = [s.model_dump() for s in segs]
        if parsed_final_answer:
            full_answer = parsed_final_answer
        full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "full_trace_done",
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "trace_id": trace_id,
                "full_trace_text": full_text,
                "full_trace_answer": full_answer,
                "full_trace_correct": full_eval.is_correct,
                "full_trace_api_trace": _trace_payload(full_trace_api),
                "parse_incomplete": parse_incomplete,
                "segments": segments,
                "prefix_oracle_steps": [],
            },
        )

    small_pricing = _pricing_from_cfg(cfg, str(cfg.small_worker.model))
    full_eval = evaluator(full_answer, task) if full_answer else evaluator("", task)
    start_step_index = len(oracle_steps) + 1
    for step_index in range(start_step_index, len(segments) + 1):
        prefix_segments = segments[:step_index]
        prefix_text = "\n".join(
            str(seg.get("text", "")).strip()
            for seg in prefix_segments
            if str(seg.get("text", "")).strip()
        )
        prefix_id = f"{trace_id}_p{step_index}"
        parent_prefix_id = f"{trace_id}_p{step_index - 1}" if step_index > 1 else ""
        logger.info("prefix.step.small_only task=%s step=%d/%d", task.id, step_index, len(segments))

        try:
            small_text, small_answer, small_api = await _run_continue(
                task=task,
                client=small_client,
                extract_client=extract_client,
                system_prompt=_SMALL_CONTINUE_SYSTEM_PROMPT,
                prefix_text=prefix_text,
                timeout_sec=int(cfg.timeouts.small_worker_sec),
                tag="small_continue_prefix",
                retry_attempts=retry_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
        except Exception:
            logger.error(
                "small_continue_prefix.error task=%s step=%d\n%s",
                task.id,
                step_index,
                traceback.format_exc(),
            )
            small_text, small_answer, small_api = "", "", None

        small_eval = evaluator(small_answer, task) if small_answer else evaluator("", task)
        oracle_steps.append(
            PrefixOracleGainStepResult(
                prefix_id=prefix_id,
                parent_prefix_id=parent_prefix_id,
                step_index=step_index,
                prefix_segments=[PrefixSegment.model_validate(s) for s in prefix_segments],
                prefix_text=prefix_text,
                small_continue_answer=small_answer,
                small_continue_correct=small_eval.is_correct,
                large_takeover_answer="",
                large_takeover_correct=False,
                delta_correctness=0.0,
                small_continue_cost=trace_to_cost(small_api, small_pricing),
                large_takeover_cost=trace_to_cost(None, None),
                small_continue_text=small_text,
                large_takeover_text="",
                small_continue_api_trace=small_api,
                large_takeover_api_trace=None,
            )
        )

        partial_store.save(
            task.id,
            {
                "task_id": task.id,
                "status": "prefix_step_done",
                "completed_step_index": step_index,
                "small_model": str(cfg.small_worker.model),
                "large_model": str(cfg.large_worker.model),
                "trace_id": trace_id,
                "full_trace_text": full_text,
                "full_trace_answer": full_answer,
                "full_trace_correct": full_eval.is_correct,
                "full_trace_api_trace": _trace_payload(full_trace_api),
                "parse_incomplete": parse_incomplete,
                "segments": segments,
                "prefix_oracle_steps": [s.model_dump() for s in oracle_steps],
            },
        )

    return PrefixOracleGainTaskResult(
        task_id=task.id,
        benchmark=task.source,
        small_model=str(cfg.small_worker.model),
        large_model=str(cfg.large_worker.model),
        trace_id=trace_id,
        full_trace_text=full_text,
        full_trace_answer=full_answer,
        full_trace_correct=full_eval.is_correct,
        full_trace_api_trace=full_trace_api,
        segments=[PrefixSegment.model_validate(s) for s in segments],
        prefix_oracle_steps=oracle_steps,
        metadata={
            "reference_answer": task.reference_answer,
            "evaluator_name": task.evaluator_name,
            "category": task.category,
            "external_difficulty": task.external_difficulty,
            "takeover_mode": "small_only_precompute",
        },
    )


async def _run_single(args: argparse.Namespace) -> None:
    cfg = _build_cfg(args)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"{args.small_label.replace('/', '_').replace(':', '_')}_to_"
        f"{args.placeholder_large_label.replace('/', '_').replace(':', '_')}"
    )
    out_path = output_dir / f"{file_stem}.json"
    store = ResultStore(out_path)

    requested = int(args.limit)
    sample_cap = requested if requested > 0 else 1_000_000
    if args.benchmark == "olympiadbench":
        questions = load_olympiadbench(num_samples=sample_cap)
    else:
        questions = load_livebench_reasoning(num_samples=sample_cap)
    if requested > 0:
        questions = questions[:requested]

    sem = asyncio.Semaphore(max(1, int(args.task_concurrency)))
    save_lock = asyncio.Lock()

    async def _run_one(i: int, task: Any) -> None:
        if store.has(task.id):
            return
        async with sem:
            logger.info("[%d/%d] %s", i, len(questions), task.id)
            try:
                result = await asyncio.wait_for(
                    _map_task_small_only(task, cfg), timeout=float(args.task_sec)
                )
            except Exception:
                logger.error("task.error task=%s\n%s", task.id, traceback.format_exc())
                return
            async with save_lock:
                store.save(result)

    pending = [asyncio.create_task(_run_one(i, t)) for i, t in enumerate(questions, start=1)]
    if pending:
        await asyncio.gather(*pending)


def _add_common_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark", choices=["olympiadbench", "livebench_reasoning"], required=True
    )
    parser.add_argument("--limit", type=int, required=True, help="0 means full benchmark split.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--small-label", required=True)
    parser.add_argument(
        "--small-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument("--small-local-model-name", default=None)
    parser.add_argument("--small-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--enable-thinking", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--placeholder-large-model", default="pending/large")
    parser.add_argument("--placeholder-large-label", default="PENDING_LARGE")
    parser.add_argument("--extractor-enabled", action="store_true")
    parser.add_argument("--extractor-model", default="openai/gpt-5.4")
    parser.add_argument(
        "--extractor-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    parser.add_argument("--extractor-max-tokens", type=int, default=512)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--full-trace-sec", type=int, default=900)
    parser.add_argument("--small-worker-sec", type=int, default=420)
    parser.add_argument("--task-sec", type=int, default=3600)


def _run_matrix(args: argparse.Namespace) -> None:
    preset_table = MODEL_PRESETS if args.small_backend != "local" else LOCAL_MODEL_PRESETS
    if args.models == "all":
        model_keys = list(preset_table.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in model_keys if m not in preset_table]
        if unknown:
            raise SystemExit(f"Unknown model preset(s): {unknown}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for benchmark in benchmarks:
        if benchmark not in {"olympiadbench", "livebench_reasoning"}:
            raise SystemExit(f"Unsupported benchmark: {benchmark}")

    total = len(model_keys) * len(benchmarks)
    idx = 0
    for model_key in model_keys:
        preset = preset_table[model_key]
        local_tag = str(preset["small_local_model_name"])
        if args.small_backend == "ollama" and args.pull_before_run:
            _run(["ollama", "pull", local_tag])
        for benchmark in benchmarks:
            idx += 1
            limit = args.olympiad_limit if benchmark == "olympiadbench" else args.livebench_limit
            out_dir = (
                ROOT
                / "outputs"
                / "results"
                / f"{args.output_prefix}_{model_key}_{benchmark}_{'full' if limit == 0 else limit}"
            )
            print(
                f"\n[{idx}/{total}] model={model_key} benchmark={benchmark} "
                f"limit={limit} out={out_dir}",
                flush=True,
            )
            cmd = [
                "uv",
                "run",
                "python",
                str(Path(__file__).resolve()),
                "single",
                "--benchmark",
                benchmark,
                "--limit",
                str(limit),
                "--output-dir",
                str(out_dir),
                "--small-model",
                str(preset["small_model"]),
                "--small-label",
                str(preset["small_label"]),
                "--small-backend",
                args.small_backend,
                "--small-local-model-name",
                str(preset["small_local_model_name"]),
                "--small-max-tokens",
                str(preset["small_max_tokens"]),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--top-k",
                str(args.top_k),
                "--seed",
                str(args.seed),
                "--num-ctx",
                str(preset["num_ctx"]),
                "--num-predict",
                str(preset["num_predict"]),
                "--enable-thinking",
                args.enable_thinking,
                "--task-concurrency",
                str(args.task_concurrency),
                "--retry-attempts",
                str(args.retry_attempts),
                "--retry-backoff-sec",
                str(args.retry_backoff_sec),
                "--full-trace-sec",
                str(args.full_trace_sec),
                "--small-worker-sec",
                str(args.small_worker_sec),
                "--task-sec",
                str(args.task_sec),
            ]
            if args.extractor_enabled:
                cmd.extend(
                    [
                        "--extractor-enabled",
                        "--extractor-model",
                        args.extractor_model,
                        "--extractor-backend",
                        args.extractor_backend,
                    ]
                )
            _run(cmd)
        if args.small_backend == "ollama" and args.delete_after_run:
            _run_optional(["ollama", "rm", local_tag])


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified small-only runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run one small-only benchmark job.")
    _add_common_single_args(single)

    matrix = subparsers.add_parser("matrix", help="Run a preset matrix of small-only jobs.")
    matrix.add_argument("--models", default="all")
    matrix.add_argument("--benchmarks", default="olympiadbench,livebench_reasoning")
    matrix.add_argument("--olympiad-limit", type=int, default=0)
    matrix.add_argument("--livebench-limit", type=int, default=0)
    matrix.add_argument("--output-prefix", default="colab_full_small")
    matrix.add_argument(
        "--small-backend", choices=["ollama", "openrouter", "local"], default="ollama"
    )
    matrix.add_argument("--temperature", type=float, default=0.0)
    matrix.add_argument("--top-p", type=float, default=0.9)
    matrix.add_argument("--top-k", type=int, default=20)
    matrix.add_argument("--seed", type=int, default=42)
    matrix.add_argument("--enable-thinking", choices=["true", "false"], default="false")
    matrix.add_argument("--task-concurrency", type=int, default=1)
    matrix.add_argument("--retry-attempts", type=int, default=3)
    matrix.add_argument("--retry-backoff-sec", type=float, default=2.0)
    matrix.add_argument("--full-trace-sec", type=int, default=1200)
    matrix.add_argument("--small-worker-sec", type=int, default=900)
    matrix.add_argument("--task-sec", type=int, default=7200)
    matrix.add_argument("--extractor-enabled", action="store_true")
    matrix.add_argument("--extractor-model", default="openai/gpt-5.4")
    matrix.add_argument(
        "--extractor-backend", choices=["openrouter", "ollama", "local"], default="openrouter"
    )
    matrix.add_argument("--pull-before-run", action="store_true")
    matrix.add_argument("--delete-after-run", action="store_true")

    args = parser.parse_args()
    if args.command == "single":
        asyncio.run(_run_single(args))
    else:
        _run_matrix(args)


if __name__ == "__main__":
    main()
