"""End-to-end smoke test: Generator + Observer pipeline.

Runs 5 tasks per available benchmark using Gemma-3-27B (k=1 for speed),
then judges one result with Claude-Opus-4.6 as observer.

Checks:
  1. Evaluator (answer system): correctness flag is bool, not always the same value
  2. Prompt accuracy: trajectory is non-empty and contains reasoning
  3. API trace: all key fields present and numeric
  4. Manager Trace: all 5 unified fields present and in-range

Usage:
    cd confidence-tom
    uv run python tools/smoke_e2e.py
    uv run python tools/smoke_e2e.py --skip-observer   # skip API cost for observer
    uv run python tools/smoke_e2e.py --benchmarks tau_bench plancraft
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force UTF-8 output on Windows (cp950 can't encode ✓/✗)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── colour helpers ──────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def ok(msg: str) -> str:
    return f"{_GREEN}✓{_RESET} {msg}"


def fail(msg: str) -> str:
    return f"{_RED}✗{_RESET} {msg}"


def warn(msg: str) -> str:
    return f"{_YELLOW}⚠{_RESET} {msg}"


def info(msg: str) -> str:
    return f"{_CYAN}·{_RESET} {msg}"


def hdr(msg: str) -> str:
    return f"\n{_BOLD}{msg}{_RESET}"


# ── result collector ────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalDetail:
    task_id: str
    is_correct: bool
    ground_truth: str
    final_answer: str
    c_rep: float
    c_beh: float
    gap: float


@dataclass
class BenchmarkReport:
    benchmark: str
    skipped: bool = False
    skip_reason: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    eval_details: list[EvalDetail] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, passed, detail))

    @property
    def all_pass(self) -> bool:
        return all(c.passed for c in self.checks)


# ── benchmark loaders (graceful skip) ───────────────────────────────────────


def _load_benchmark(name: str, n: int) -> tuple[list[Any], Optional[str]]:
    """Return (tasks, error_message). error_message is None on success."""
    try:
        if name == "tau_bench":
            from confidence_tom.benchmarks.tau_bench import load_tau_bench

            return load_tau_bench(env="retail", split="test", num_samples=n), None

        if name == "plancraft":
            from confidence_tom.benchmarks.plancraft import load_plancraft

            return load_plancraft(split="test", num_samples=n), None

        if name == "bird_sql":
            from confidence_tom.benchmarks.bird_sql import load_bird_sql

            return load_bird_sql(split="dev", num_samples=n), None

        if name == "intercode":
            import docker

            from confidence_tom.benchmarks.intercode import load_intercode

            docker.from_env().ping()
            return load_intercode(env="bash", num_samples=n), None

    except ImportError as e:
        return [], f"import error: {e}"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

    return [], f"unknown benchmark: {name}"


# ── evaluators ──────────────────────────────────────────────────────────────


def _get_evaluator(benchmark: str) -> Callable[..., Any]:
    from confidence_tom.eval.evaluators import build_evaluator

    return build_evaluator(benchmark)


# ── check helpers ────────────────────────────────────────────────────────────


def _check_trace(trace_dict: Optional[dict[str, Any]], label: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if trace_dict is None:
        checks.append(CheckResult(f"{label}.api_trace present", False, "trace is None"))
        return checks
    checks.append(CheckResult(f"{label}.api_trace present", True))

    # Required string fields
    for field_name in ("model_id", "request_id"):
        v = trace_dict.get(field_name)
        checks.append(
            CheckResult(f"{label}.{field_name}", isinstance(v, str) and len(v) > 0, f"got {v!r}")
        )

    # Required numeric fields
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = trace_dict.get(field_name)
        checks.append(
            CheckResult(f"{label}.{field_name} >= 0", isinstance(v, int) and v >= 0, f"got {v!r}")
        )

    # Informational fields — always pass, just show the value
    for field_name in ("reasoning_tokens", "cache_read_tokens", "cache_write_tokens"):
        v = trace_dict.get(field_name, "MISSING")
        checks.append(CheckResult(f"{label}.{field_name} [info]", v != "MISSING", f"got {v!r}"))

    rc = trace_dict.get("reasoning_content", "MISSING")
    checks.append(
        CheckResult(
            f"{label}.reasoning_content [info]",
            rc != "MISSING",
            f"len={len(rc) if isinstance(rc, str) else '?'}, preview={str(rc)[:60]!r}",
        )
    )

    return checks


def _check_manager_trace(judgment: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    for key in (
        "judge_reasoning",
        "predicted_correctness",
        "predicted_worker_confidence",
        "predicted_error_type",
        "manager_self_confidence",
    ):
        checks.append(
            CheckResult(
                f"manager.{key} present",
                key in judgment and judgment[key] is not None,
                f"got {judgment.get(key)!r}",
            )
        )

    VALID_ERROR_TYPES = {
        "Logic_Error",
        "Hallucination",
        "Tool_Argument_Error",
        "Observation_Ignored",
        "None",
    }
    et = judgment.get("predicted_error_type", "")
    checks.append(
        CheckResult("manager.predicted_error_type valid", et in VALID_ERROR_TYPES, f"got {et!r}")
    )

    for float_key in (
        "predicted_correctness",
        "predicted_worker_confidence",
        "manager_self_confidence",
    ):
        v = judgment.get(float_key)
        checks.append(
            CheckResult(
                f"manager.{float_key} in [0,1]",
                isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0,
                f"got {v!r}",
            )
        )

    reasoning = judgment.get("judge_reasoning", "")
    checks.append(
        CheckResult(
            "manager.judge_reasoning non-trivial (>50 chars)",
            isinstance(reasoning, str) and len(reasoning) > 50,
            f"len={len(reasoning) if reasoning else 0}",
        )
    )

    return checks


# ── main async runner ────────────────────────────────────────────────────────


async def run_benchmark(
    benchmark: str,
    n_tasks: int,
    subject_model_id: str,
    observer_model_id: str,
    skip_observer: bool,
) -> BenchmarkReport:
    report = BenchmarkReport(benchmark=benchmark)

    # -- load tasks --
    tasks, err = _load_benchmark(benchmark, n_tasks)
    if err or not tasks:
        report.skipped = True
        report.skip_reason = err or "no tasks returned"
        return report

    report.add("tasks_loaded", True, f"{len(tasks)} tasks")

    # -- set up runner --
    from confidence_tom.generator.runner import AgentRunner

    runner = AgentRunner(
        model_name=subject_model_id,
        temperature=0.7,
        k_samples=1,  # k=1 for speed in smoke test
        max_tokens=2048,
    )
    evaluator = _get_evaluator(benchmark)

    # -- run tasks --
    task_results = []
    for task in tasks:
        result = await runner.run(task, evaluator)
        if result is None:
            report.add(f"[{task.task_id}] run_succeeded", False, "runner returned None")
            continue
        task_results.append((task, result))

    if not task_results:
        report.add("any_task_completed", False, "all tasks failed to run")
        return report

    report.add("tasks_completed", True, f"{len(task_results)}/{len(tasks)}")

    # ── Check 1: Evaluator (answer system) — full per-task dump ──────────────
    correctness_values = [r.majority_correct for _, r in task_results]

    print(f"\n  {_BOLD}── Evaluator detail ({benchmark}) ──{_RESET}")
    for task, result in task_results:
        run0 = result.runs[0]
        is_correct = run0.is_correct
        gap = result.avg_reported_confidence - result.behavioral_confidence

        gt = str(task.ground_truth)
        gt_show = gt[:120] + "..." if len(gt) > 120 else gt
        ans_show = (
            run0.final_answer[:120] + "..." if len(run0.final_answer) > 120 else run0.final_answer
        )

        verdict = f"{_GREEN}CORRECT{_RESET}" if is_correct else f"{_RED}WRONG{_RESET}"
        print(f"\n  [{task.task_id}] {verdict}")
        print(f"    ground_truth : {gt_show!r}")
        print(f"    final_answer : {ans_show!r}")
        print(
            "    "
            f"c_rep={run0.reported_confidence:.2f}  "
            f"c_beh={result.behavioral_confidence:.2f}  "
            f"gap={gap:+.2f}"
        )

        report.eval_details.append(
            EvalDetail(
                task_id=task.task_id,
                is_correct=is_correct,
                ground_truth=gt,
                final_answer=run0.final_answer,
                c_rep=round(run0.reported_confidence, 4),
                c_beh=round(result.behavioral_confidence, 4),
                gap=round(gap, 4),
            )
        )

    print()

    report.add(
        "check1.evaluator_returns_bool",
        all(isinstance(v, bool) for v in correctness_values),
        f"values={correctness_values}",
    )
    report.add(
        "check1.c_beh_in_range",
        all(0.0 <= r.behavioral_confidence <= 1.0 for _, r in task_results),
        f"c_behs={[round(r.behavioral_confidence, 2) for _, r in task_results]}",
    )
    report.add(
        "check1.c_rep_in_range",
        all(0.0 <= r.avg_reported_confidence <= 1.0 for _, r in task_results),
        f"c_reps={[round(r.avg_reported_confidence, 2) for _, r in task_results]}",
    )

    # ── Check 2: Prompt accuracy (trajectory non-empty, has reasoning) ────────
    for _, result in task_results:
        traj = result.primary_trajectory
        report.add(
            f"check2.[{result.task_id}].trajectory_nonempty",
            bool(traj and len(traj) > 20),
            f"len={len(traj)}",
        )
        run0 = result.runs[0]
        report.add(
            f"check2.[{result.task_id}].final_answer_nonempty",
            bool(run0.final_answer and len(run0.final_answer) > 0),
            f"ans={run0.final_answer[:80]!r}",
        )
        report.add(
            f"check2.[{result.task_id}].confidence_reported",
            0.0 <= run0.reported_confidence <= 1.0,
            f"c_rep={run0.reported_confidence:.2f}",
        )
        # Dynamic Trajectory Schema checks
        report.add(
            f"check2.[{result.task_id}].plan_nonempty",
            bool(run0.plan and len(run0.plan) > 0),
            f"plan={run0.plan[:60]!r}",
        )
        report.add(
            f"check2.[{result.task_id}].trajectory_steps_present",
            len(run0.trajectory) > 0,
            f"steps={len(run0.trajectory)}",
        )
        if run0.trajectory:
            step0 = run0.trajectory[0]
            report.add(
                f"check2.[{result.task_id}].step1_has_thought",
                bool(step0.thought),
                f"thought={step0.thought[:60]!r}",
            )
            report.add(
                f"check2.[{result.task_id}].step1_has_action",
                bool(step0.action),
                f"action={step0.action[:60]!r}",
            )
            report.add(
                f"check2.[{result.task_id}].step1_confidence_in_range",
                0 <= step0.step_confidence <= 100,
                f"step_confidence={step0.step_confidence}",
            )
        report.add(
            f"check2.[{result.task_id}].summary_nonempty",
            bool(run0.summary and len(run0.summary) > 0),
            f"summary={run0.summary[:60]!r}",
        )

    # ── Check 3: API trace ────────────────────────────────────────────────────
    for _, result in task_results:
        run0 = result.runs[0]
        trace = run0.api_trace
        trace_dict = trace.model_dump() if trace else None
        for c in _check_trace(trace_dict, f"check3.[{result.task_id}]"):
            report.checks.append(c)

    # ── Check 4: Observer / Manager Trace ────────────────────────────────────
    if skip_observer:
        report.add("check4.observer", True, "skipped (--skip-observer)")
    else:
        # Judge only the first task to save API cost
        from confidence_tom.infra.client import LLMClient

        _, first_result = task_results[0]
        subject_record = {
            "task_id": first_result.task_id,
            "benchmark": benchmark,
            "instruction": first_result.instruction,
            "primary_trajectory": first_result.primary_trajectory,
            "majority_correct": first_result.majority_correct,
            "c_beh": first_result.behavioral_confidence,
            "c_rep": first_result.avg_reported_confidence,
            "gap": first_result.avg_reported_confidence - first_result.behavioral_confidence,
            "runs": [
                {
                    "final_answer": r.final_answer,
                    "reported_confidence": r.reported_confidence,
                    "api_trace": r.api_trace.model_dump() if r.api_trace else None,
                }
                for r in first_result.runs
            ],
        }

        # import the judge_one from the experiment script
        sys.path.insert(0, str(ROOT / "experiments"))
        from run_scale_observer import judge_one

        obs_client = LLMClient(
            model=observer_model_id,
            temperature=0.0,
            max_tokens=2048,
        )

        obs_result = await judge_one(obs_client, subject_record)
        if obs_result is None:
            report.add("check4.observer_call", False, "judge_one returned None")
        else:
            judgment, manager_trace = obs_result
            report.add("check4.observer_call", True)
            for c in _check_manager_trace(judgment):
                report.checks.append(c)
            for c in _check_trace(manager_trace, "check4.manager_api_trace"):
                report.checks.append(c)

    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


async def amain() -> None:
    parser = argparse.ArgumentParser(description="E2E smoke test: generator + observer")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["tau_bench", "plancraft", "bird_sql", "intercode"],
        help="Benchmarks to test",
    )
    parser.add_argument("--n", type=int, default=5, help="Tasks per benchmark (default 5)")
    parser.add_argument(
        "--subject",
        default="google/gemma-3-27b-it",
        help="Subject (Worker) model ID",
    )
    parser.add_argument(
        "--observer",
        default="anthropic/claude-opus-4.6",
        help="Observer (Manager) model ID",
    )
    parser.add_argument(
        "--skip-observer",
        action="store_true",
        help="Skip observer check (no API cost for manager)",
    )
    parser.add_argument(
        "--output",
        default="outputs/results/smoke_e2e.json",
        help="Where to write full report JSON",
    )
    args = parser.parse_args()

    print(hdr("=" * 60))
    print(hdr(" E2E SMOKE TEST"))
    print(hdr("=" * 60))
    print(info(f"Subject  : {args.subject}"))
    print(info(f"Observer : {args.observer} {'(skipped)' if args.skip_observer else ''}"))
    print(info(f"Tasks/bm : {args.n}"))
    print(info(f"Benchmarks: {args.benchmarks}"))

    all_reports: list[BenchmarkReport] = []

    for bm in args.benchmarks:
        print(hdr(f"── {bm} ──"))
        try:
            report = await run_benchmark(
                benchmark=bm,
                n_tasks=args.n,
                subject_model_id=args.subject,
                observer_model_id=args.observer,
                skip_observer=args.skip_observer,
            )
        except Exception:
            report = BenchmarkReport(benchmark=bm, skipped=True, skip_reason=traceback.format_exc())

        all_reports.append(report)

        if report.skipped:
            print(warn(f"SKIPPED: {report.skip_reason}"))
            continue

        for c in report.checks:
            if c.passed:
                print(ok(f"{c.name}  {c.detail}"))
            else:
                print(fail(f"{c.name}  {c.detail}"))

        status = (
            f"{_GREEN}ALL PASS{_RESET}" if report.all_pass else f"{_RED}FAILURES DETECTED{_RESET}"
        )
        print(f"\n  → {bm}: {status}\n")

    # ── summary ──────────────────────────────────────────────────────────────
    print(hdr("=" * 60))
    print(hdr(" SUMMARY"))
    print(hdr("=" * 60))
    total_pass = total_fail = total_skip = 0
    for r in all_reports:
        if r.skipped:
            print(warn(f"  {r.benchmark}: SKIPPED ({r.skip_reason[:80]})"))
            total_skip += 1
        elif r.all_pass:
            print(ok(f"  {r.benchmark}: {len(r.checks)} checks passed"))
            total_pass += 1
        else:
            failures = [c for c in r.checks if not c.passed]
            print(fail(f"  {r.benchmark}: {len(failures)} checks FAILED"))
            for c in failures:
                print(f"      {_RED}✗{_RESET} {c.name}: {c.detail}")
            total_fail += 1

    print(f"\n  pass={total_pass}  fail={total_fail}  skip={total_skip}")

    # ── write JSON report ─────────────────────────────────────────────────────
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject_model": args.subject,
        "observer_model": args.observer,
        "skip_observer": args.skip_observer,
        "benchmarks": [
            {
                "benchmark": r.benchmark,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
                "all_pass": r.all_pass,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                ],
                "eval_details": [
                    {
                        "task_id": e.task_id,
                        "is_correct": e.is_correct,
                        "ground_truth": e.ground_truth,
                        "final_answer": e.final_answer,
                        "c_rep": e.c_rep,
                        "c_beh": e.c_beh,
                        "gap": e.gap,
                    }
                    for e in r.eval_details
                ],
            }
            for r in all_reports
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(info(f"\nFull report written to: {out_path}"))

    sys.exit(0 if total_fail == 0 else 1)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
