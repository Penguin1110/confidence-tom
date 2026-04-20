from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_running(pattern: str) -> bool:
    result = subprocess.run(
        ["bash", "-lc", f"pgrep -f {shlex.quote(pattern)} >/dev/null 2>&1"],
        cwd=ROOT,
    )
    return result.returncode == 0


def _wait_for_pattern_to_stop(pattern: str, poll_sec: int) -> None:
    while _is_running(pattern):
        print(f"[wait] still running: {pattern}")
        time.sleep(poll_sec)


def _run_queue_item(config_name: str) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(ROOT / "experiments" / "run_prefix_family_sweep.py"),
        "--config-name",
        config_name,
    ]
    print("\n[run]")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local-small family sweep configs sequentially after a wait target finishes."
    )
    parser.add_argument(
        "--wait-pattern",
        default="results/local_mistral_to_openai_50",
        help="Substring pattern passed to pgrep -f; queue starts after this process pattern disappears.",
    )
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent configs even if one config exits non-zero.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "prefix_family_sweep_qwen_local",
            "prefix_family_sweep_livebench_mistral_local",
            "prefix_family_sweep_livebench_qwen_local",
        ],
    )
    args = parser.parse_args()

    _wait_for_pattern_to_stop(str(args.wait_pattern), poll_sec=int(args.poll_sec))
    print(f"[start] wait target finished: {args.wait_pattern}")

    for config_name in args.configs:
        try:
            _run_queue_item(str(config_name))
        except subprocess.CalledProcessError as exc:
            print(f"[error] config failed: {config_name} returncode={exc.returncode}")
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
