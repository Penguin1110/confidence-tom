"""Run Plancraft using its native gym-style wrapper.

This script is intentionally benchmark-specific: it interacts with the
PlancraftGymWrapper directly instead of the generic dynamic generator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples

from confidence_tom.client import LLMClient


SYSTEM_PROMPT = """You are solving a Plancraft task in the native environment.
You must output exactly one next action each turn.

Valid actions:
- move: from [Source] to [Target] with quantity N
- smelt: from [Source] to [Target] with quantity N
- impossible: <reason>

Rules:
- Only output one action line.
- Do not explain.
- Do not invent inventory state; rely on the observation."""


async def run_one(client: LLMClient, example, max_steps: int) -> dict:
    env = PlancraftGymWrapper(example=example, max_steps=max_steps)
    obs, reward, terminated, truncated, info = env.step()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs["text"]},
    ]
    trajectory = []

    while not terminated and not truncated:
        action = await asyncio.to_thread(client.generate_text, messages)
        action = action.strip().splitlines()[0] if action.strip() else "impossible: no action produced"
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(
            {
                "action": action,
                "observation": obs.get("text", ""),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        messages.extend(
            [
                {"role": "assistant", "content": action},
                {"role": "user", "content": obs.get("text", "")},
            ]
        )

    return {
        "example_id": example.id,
        "target": example.target,
        "success": bool(reward >= 1.0),
        "trajectory": trajectory,
    }


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Run Plancraft natively")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output", default="results/plancraft_native.json")
    args = parser.parse_args()

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=128)
    examples = [ex for ex in get_plancraft_examples(split=args.split) if not ex.impossible][: args.num_samples]
    results = []
    for example in examples:
        results.append(await run_one(client, example, args.max_steps))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
