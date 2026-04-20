from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback

from confidence_tom.client import LLMClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal local-model smoke test.")
    parser.add_argument("--model", required=True, help="Logical model id used by LLMClient.")
    parser.add_argument("--backend", default="local", help="Client backend, default=local.")
    parser.add_argument("--local-model-name", default=None, help="Optional explicit HF checkpoint.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--enable-thinking", type=str, choices=["true", "false"], default=None)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: LOCAL_SMOKE_OK",
        help="User prompt to send.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    started = time.time()
    print(
        json.dumps(
            {
                "event": "smoke_start",
                "model": args.model,
                "backend": args.backend,
                "local_model_name": args.local_model_name,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "enable_thinking": args.enable_thinking,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        client = LLMClient(
            model=args.model,
            backend=args.backend,
            local_model_name=args.local_model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            enable_thinking=(
                None if args.enable_thinking is None else args.enable_thinking == "true"
            ),
        )
        text, trace = await client.agenerate_text_with_trace(
            [
                {"role": "system", "content": "You are helpful and concise."},
                {"role": "user", "content": args.prompt},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(
            json.dumps(
                {
                    "event": "smoke_result",
                    "elapsed_sec": round(time.time() - started, 3),
                    "text": text,
                    "trace_model_id": trace.model_id,
                    "prompt_tokens": trace.prompt_tokens,
                    "completion_tokens": trace.completion_tokens,
                    "total_tokens": trace.total_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "smoke_error",
                    "elapsed_sec": round(time.time() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
