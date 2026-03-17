"""Run smoke checks for dynamic benchmarks (excluding AgentBench by default)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SmokeResult:
    benchmark: str
    ok: bool
    detail: str


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def check_tau_bench() -> SmokeResult:
    run_py = ROOT / "external" / "tau-bench" / "run.py"
    if not run_py.exists():
        return SmokeResult("tau-bench", False, f"missing:{run_py}")
    proc = _run([sys.executable, str(run_py), "--help"], cwd=run_py.parent)
    if proc.returncode == 0:
        return SmokeResult("tau-bench", True, "cli_help_ok")
    tail = (proc.stderr or proc.stdout)[-300:]
    return SmokeResult("tau-bench", False, f"cli_help_failed:{tail}")


def check_plancraft() -> SmokeResult:
    code = (
        "from plancraft.simple import get_plancraft_examples, PlancraftGymWrapper\n"
        "examples = get_plancraft_examples(split='test')\n"
        "ex = examples[0]\n"
        "env = PlancraftGymWrapper(example=ex, max_steps=1, use_text_inventory=True)\n"
        "_ = env.step()\n"
        "_ = env.step('')\n"
        "print('plancraft_ok')\n"
    )
    proc = _run([sys.executable, "-c", code], cwd=ROOT)
    if proc.returncode == 0:
        return SmokeResult("plancraft", True, "gym_wrapper_ok")
    tail = (proc.stderr or proc.stdout)[-300:]
    return SmokeResult("plancraft", False, f"plancraft_runtime_failed:{tail}")


def check_intercode() -> SmokeResult:
    code = (
        "import docker\n"
        "from intercode.envs import BashEnv\n"
        "from intercode.assets import bash_image_name\n"
        "client = docker.from_env()\n"
        "client.ping()\n"
        "print('intercode_ok', bash_image_name)\n"
    )
    proc = _run([sys.executable, "-c", code], cwd=ROOT)
    if proc.returncode == 0:
        return SmokeResult("intercode", True, "docker_and_import_ok")
    tail = (proc.stderr or proc.stdout)[-300:]
    if "Error while fetching server API version" in (proc.stderr or ""):
        return SmokeResult(
            "intercode",
            False,
            "docker_not_running_or_not_reachable",
        )
    return SmokeResult("intercode", False, f"intercode_check_failed:{tail}")


def check_birdsql() -> SmokeResult:
    eval_py = ROOT / "external" / "birdsql" / "bird" / "llm" / "src" / "evaluation.py"
    if not eval_py.exists():
        return SmokeResult("birdsql", False, f"missing:{eval_py}")

    proc = _run([sys.executable, "-m", "py_compile", str(eval_py)], cwd=ROOT)
    if proc.returncode == 0:
        return SmokeResult("birdsql", True, "evaluation_script_compiles")
    tail = (proc.stderr or proc.stdout)[-300:]
    return SmokeResult("birdsql", False, f"evaluation_compile_failed:{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test dynamic benchmarks")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["tau-bench", "plancraft", "intercode", "birdsql"],
        help="Benchmarks to check",
    )
    parser.add_argument(
        "--output",
        default="external/dynamic_smoke_status.json",
        help="Where to write smoke status JSON",
    )
    args = parser.parse_args()

    checks = {
        "tau-bench": check_tau_bench,
        "plancraft": check_plancraft,
        "intercode": check_intercode,
        "birdsql": check_birdsql,
    }

    results: list[SmokeResult] = []
    for key in args.benchmarks:
        if key not in checks:
            results.append(SmokeResult(key, False, "unknown_benchmark"))
            continue
        results.append(checks[key]())

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.version,
        "platform": os.name,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.benchmark}: {result.detail}")
    print(f"[done] wrote smoke status to {output_path}")


if __name__ == "__main__":
    main()