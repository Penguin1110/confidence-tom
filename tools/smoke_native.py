"""Smoke test for the native dynamic benchmark pipeline.

Runs 1 task x k=1 from each enabled benchmark and verifies:
  - c_beh is a float in [0, 1]
  - c_rep is not None  (confidence elicitation worked)
  - gap is not None
  - primary_trace / trace_text is non-empty
  - runs[0].reported_confidence is not None

Usage:
    uv run python tools/smoke_native.py
    uv run python tools/smoke_native.py --model qwen/qwen3-32b
    uv run python tools/smoke_native.py --skip tau_bench
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── colour helpers ────────────────────────────────────────────────────────────
_G, _R, _Y, _B, _X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
def ok(m):   return f"{_G}PASS{_X}  {m}"
def fail(m): return f"{_R}FAIL{_X}  {m}"
def hdr(m):  return f"\n{_B}{m}{_X}"


# ── check helpers ─────────────────────────────────────────────────────────────

def _check_result(label: str, result) -> list[tuple[bool, str]]:
    checks = []

    def c(cond, msg):
        checks.append((bool(cond), f"[{label}] {msg}"))

    c(isinstance(result.c_beh, float) and 0.0 <= result.c_beh <= 1.0,
      f"c_beh={result.c_beh!r} in [0,1]")

    c(result.c_rep is not None,
      f"c_rep={result.c_rep!r} (confidence elicitation returned a value)")

    c(result.gap is not None,
      f"gap={result.gap!r}")

    trace = result.primary_trace or ""
    c(len(trace) > 20,
      f"primary_trace len={len(trace)}")

    run0 = result.runs[0] if result.runs else None
    c(run0 is not None, "has at least one run")
    if run0:
        c(run0.reported_confidence is not None,
          f"runs[0].reported_confidence={run0.reported_confidence!r}")
        c(len(run0.trace_text) > 0,
          f"runs[0].trace_text len={len(run0.trace_text)}")

    return checks


# ── per-benchmark runners ─────────────────────────────────────────────────────

async def run_tau_bench(model_id: str, user_model: str) -> object:
    from run_scale_generator import (
        _load_tau_tasks, _run_tau_native_task,
        _TauBenchUserSimulator,
    )
    from confidence_tom.client import LLMClient

    tau_root = ROOT / "external" / "tau-bench"
    sys.path.insert(0, str(tau_root))
    from tau_bench.envs import get_env  # type: ignore
    from tau_bench.types import Action  # type: ignore

    tau_cfg = SimpleNamespace(
        env="retail", split="test", max_steps=20,
        get=lambda k, d=None: d,
    )
    tau_cfg.get = lambda k, d=None: {"max_steps": 20}.get(k, d)

    agent_client = LLMClient(model=model_id, temperature=0.7, max_tokens=4096)
    user_client  = LLMClient(model=user_model, temperature=0.0, max_tokens=256)

    task_spec = {"task_id": "tau_retail_test_0000", "task_index": 0}
    return await _run_tau_native_task(
        task_spec,
        lambda *a, **kw: get_env(*a, **kw),
        Action,
        tau_cfg,
        agent_client,
        user_client,
        k_samples=1,
        log_prefix="[smoke/tau_bench]",
        use_tool_calling=True,
    )


async def run_bird_sql(model_id: str) -> object:
    from run_scale_generator import _run_bird_native_task
    from confidence_tom.benchmarks.bird_sql import load_bird_sql
    from confidence_tom.client import LLMClient

    tasks = load_bird_sql(split="dev", num_samples=1)
    client = LLMClient(model=model_id, temperature=0.7, max_tokens=2048)
    return await _run_bird_native_task(tasks[0], client, k_samples=1)


async def run_plancraft(model_id: str) -> object:
    from run_scale_generator import _load_plancraft_examples, _run_plancraft_native_task
    from confidence_tom.client import LLMClient

    specs = _load_plancraft_examples(split="test", limit=1)
    client = LLMClient(model=model_id, temperature=0.7, max_tokens=512)
    return await _run_plancraft_native_task(specs[0], client, k_samples=1)


# ── main ──────────────────────────────────────────────────────────────────────

RUNNERS = {
    "tau_bench": run_tau_bench,
    "bird_sql":  run_bird_sql,
    "plancraft":  run_plancraft,
}


async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="qwen/qwen3-8b")
    parser.add_argument("--user-model", default="openai/gpt-4o-mini",
                        help="User simulator model for tau_bench")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Benchmarks to skip, e.g. --skip tau_bench")
    args = parser.parse_args()

    print(hdr("=" * 55))
    print(hdr("  NATIVE PIPELINE SMOKE TEST  (k=1, n=1)"))
    print(hdr("=" * 55))
    print(f"  model      : {args.model}")
    print(f"  user_model : {args.user_model}")
    print(f"  skipping   : {args.skip or 'none'}")

    all_checks: list[tuple[bool, str]] = []

    for bm, runner_fn in RUNNERS.items():
        print(hdr(f"── {bm} ──"))
        if bm in args.skip:
            print(f"  {_Y}SKIPPED{_X}")
            continue
        try:
            if bm == "tau_bench":
                result = await runner_fn(args.model, args.user_model)
            else:
                result = await runner_fn(args.model)

            checks = _check_result(bm, result)
            for passed, msg in checks:
                print(ok(msg) if passed else fail(msg))
            all_checks.extend(checks)

            # Show key values
            print(f"\n  c_beh={result.c_beh}  c_rep={result.c_rep}  gap={result.gap}")
            if result.runs:
                r0 = result.runs[0]
                rs = r0.run_summary
                has_trajectory = bool(rs and rs.trajectory)
                confidence_source = "structured" if has_trajectory else "text-fallback"
                print(f"  is_correct={r0.is_correct}  reported_confidence={r0.reported_confidence}  [{confidence_source}]")
                if rs:
                    print(f"  final_confidence (raw)={rs.final_confidence}  final_answer={rs.final_answer!r:.60}")
                print(f"  trace_text preview: {r0.trace_text[:120]!r}")

        except Exception:
            msg = f"[{bm}] raised exception"
            print(fail(msg))
            print(traceback.format_exc())
            all_checks.append((False, msg))

    # ── summary ───────────────────────────────────────────────────────────────
    print(hdr("=" * 55))
    print(hdr("  SUMMARY"))
    print(hdr("=" * 55))
    n_pass = sum(1 for p, _ in all_checks if p)
    n_fail = sum(1 for p, _ in all_checks if not p)
    print(f"  {_G}{n_pass} passed{_X}  {_R}{n_fail} failed{_X}  (total {len(all_checks)})")
    if n_fail:
        print(f"\n  {_R}Failed checks:{_X}")
        for p, msg in all_checks:
            if not p:
                print(f"    {_R}✗{_X} {msg}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    asyncio.run(amain())
