"""Run tau-bench via its native environment/agent entrypoint.

This is the benchmark-faithful path for tau-bench. It delegates to the
official tau-bench runner instead of using the generic dynamic generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT / "external" / "tau-bench"
sys.path.insert(0, str(TAU_ROOT))

from tau_bench.run import run  # noqa: E402
from tau_bench.types import RunConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tau-bench natively")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-provider", default="openrouter")
    parser.add_argument("--user-model", default="")
    parser.add_argument("--user-model-provider", default="")
    parser.add_argument("--env", choices=["retail", "airline"], default="retail")
    parser.add_argument("--task-split", choices=["train", "test", "dev"], default="test")
    parser.add_argument(
        "--agent-strategy",
        choices=["tool-calling", "act", "react", "few-shot"],
        default="tool-calling",
    )
    parser.add_argument("--user-strategy", default="llm")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", default="outputs/results/tau_bench_native")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--few-shot-displays-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_model = args.user_model or args.model
    user_provider = args.user_model_provider or args.model_provider
    config = RunConfig(
        model_provider=args.model_provider,
        user_model_provider=user_provider,
        model=args.model,
        user_model=user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
    )
    run(config)


if __name__ == "__main__":
    main()
