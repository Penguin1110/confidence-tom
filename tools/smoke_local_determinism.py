#!/usr/bin/env python3
"""Smoke test: verify local model returns identical outputs for the same prompt (temperature=0)."""
from __future__ import annotations

import argparse
import sys

from confidence_tom.infra.client_local import local_generate_text

PROMPT = [{"role": "user", "content": "What is 7 multiplied by 8? Reply with just the number."}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local model determinism smoke test.")
    parser.add_argument("--model", required=True, help="HuggingFace model name, e.g. google/gemma-3-4b-it")
    parser.add_argument("--n", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"model     : {args.model}")
    print(f"prompt    : {PROMPT[0]['content']}")
    print(f"runs      : {args.n}")
    print(f"max_tokens: {args.max_tokens}")
    print()

    responses: list[str] = []
    for i in range(args.n):
        text, _ = local_generate_text(
            model_name=args.model,
            trust_remote_code=True,
            messages=PROMPT,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        text = text.strip()
        responses.append(text)
        print(f"  run {i + 1}: {text!r}")

    print()
    if len(set(responses)) == 1:
        print("[PASS] All responses are identical.")
    else:
        print("[FAIL] Responses differ across runs!")
        for i, r in enumerate(responses, 1):
            print(f"  run {i}: {r!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
