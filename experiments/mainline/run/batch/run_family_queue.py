from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = ROOT / "outputs" / "logs"
JOB_DIR = ROOT / "outputs" / "jobs"
SWEEP_RUNNER = ROOT / "experiments" / "mainline" / "run" / "batch" / "run_prefix_family_sweep.py"
REENTRY_RUNNER = ROOT / "experiments" / "mainline" / "run" / "batch" / "run_reentry_mainline.py"


def _job_paths(job_name: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{job_name}.log", JOB_DIR / f"{job_name}.pid"


def _is_running_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_running_pattern(pattern: str) -> bool:
    result = subprocess.run(
        ["bash", "-lc", f"pgrep -f {shlex.quote(pattern)} >/dev/null 2>&1"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except Exception:
        return None


def _sweep_command(config_name: str, run_name_prefix: str) -> str:
    return (
        f"uv run python {shlex.quote(str(SWEEP_RUNNER))} "
        f"--config-name {shlex.quote(config_name)} "
        f"launcher.run_name_prefix={shlex.quote(run_name_prefix)} "
        "launcher.overwrite_output_dir=false"
    )


def _queue_command(config_name: str) -> list[str]:
    return ["uv", "run", "python", str(SWEEP_RUNNER), "--config-name", config_name]


def _reentry_command(preset: str, phase: str, dry_run: bool) -> list[str]:
    cmd = ["uv", "run", "python", str(REENTRY_RUNNER), "--preset", preset, "--phase", phase]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _start_job(job_name: str, command: str) -> None:
    log_path, pid_path = _job_paths(job_name)
    existing_pid = _read_pid(pid_path)
    if existing_pid and _is_running_pid(existing_pid):
        print(f"JOB_NAME={job_name}")
        print("STATUS=running")
        print(f"PID={existing_pid}")
        print(f"LOG={log_path}")
        return

    with log_path.open("ab") as lf:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid))
    print(f"JOB_NAME={job_name}")
    print("STATUS=started")
    print(f"PID={proc.pid}")
    print(f"LOG={log_path}")


def _status_job(job_name: str) -> None:
    log_path, pid_path = _job_paths(job_name)
    pid = _read_pid(pid_path)
    if pid is None:
        print("STATUS=missing")
        return
    status = "running" if _is_running_pid(pid) else "stopped"
    print(f"STATUS={status}")
    print(f"PID={pid}")
    print(f"LOG={log_path}")
    print(f"LOG_EXISTS={int(log_path.exists())}")


def _tail_job(job_name: str, lines: int) -> None:
    log_path, _ = _job_paths(job_name)
    if not log_path.exists():
        print(f"LOG_MISSING={log_path}")
        return
    subprocess.run(["tail", "-n", str(lines), str(log_path)], check=False)


def _stop_job(job_name: str, force: bool) -> None:
    _, pid_path = _job_paths(job_name)
    pid = _read_pid(pid_path)
    if pid is None:
        print("STATUS=missing")
        return
    if not _is_running_pid(pid):
        print("STATUS=already_stopped")
        pid_path.unlink(missing_ok=True)
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    print(f"STATUS=sent_{'sigkill' if force else 'sigterm'}")
    print(f"PID={pid}")


def _build_target_jobs(target: str, run_prefix_base: str) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    if target in {"olympiad", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_olympiad_colab",
                "prefix_family_sweep",
                run_prefix_base,
            )
        )
    if target in {"livebench", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_livebench_colab",
                "prefix_family_sweep_livebench",
                f"{run_prefix_base}livebench_",
            )
        )
    return jobs


def _run_job_mode(args: argparse.Namespace) -> None:
    jobs = _build_target_jobs(args.target, args.run_prefix_base)
    if args.mode == "run":
        for _, config_name, run_name_prefix in jobs:
            cmd = _sweep_command(config_name, run_name_prefix)
            print("$", cmd)
            result = subprocess.run(["/bin/bash", "-lc", cmd], cwd=ROOT, check=False)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        return

    for job_name, config_name, run_name_prefix in jobs:
        if args.mode == "start":
            _start_job(job_name, _sweep_command(config_name, run_name_prefix))
        elif args.mode == "status":
            _status_job(job_name)
        elif args.mode == "tail":
            _tail_job(job_name, args.tail_lines)
        elif args.mode == "stop":
            _stop_job(job_name, force=args.force)


def _run_wait_queue(args: argparse.Namespace) -> None:
    while _is_running_pattern(str(args.wait_pattern)):
        print(f"[wait] still running: {args.wait_pattern}")
        time.sleep(int(args.poll_sec))

    print(f"[start] wait target finished: {args.wait_pattern}")
    for config_name in args.configs:
        cmd = _queue_command(str(config_name))
        print("\n[run]")
        print(" ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"[error] config failed: {config_name} returncode={result.returncode}")
            if not args.continue_on_error:
                raise SystemExit(result.returncode)


def _run_reentry_mode(args: argparse.Namespace) -> None:
    cmd = _reentry_command(args.preset, args.phase, args.dry_run)
    if args.mode == "run":
        print("$", " ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return

    job_name = args.job_name or f"reentry_{args.preset}"
    command = " ".join(shlex.quote(part) for part in cmd)
    if args.mode == "start":
        _start_job(job_name, command)
    elif args.mode == "status":
        _status_job(job_name)
    elif args.mode == "tail":
        _tail_job(job_name, args.tail_lines)
    elif args.mode == "stop":
        _stop_job(job_name, force=args.force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified queue launcher for batch family sweep workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage background family sweep jobs for olympiad/livebench targets.",
    )
    jobs_parser.add_argument("--target", choices=["olympiad", "livebench", "both"], default="both")
    jobs_parser.add_argument(
        "--mode", choices=["run", "start", "status", "tail", "stop"], default="start"
    )
    jobs_parser.add_argument("--tail-lines", type=int, default=80)
    jobs_parser.add_argument("--run-prefix-base", default="colab_rerun_")
    jobs_parser.add_argument("--force", action="store_true")

    wait_parser = subparsers.add_parser(
        "wait",
        help="Wait for an existing pattern to stop, then run a queue of family sweep configs.",
    )
    wait_parser.add_argument(
        "--wait-pattern",
        default="outputs/results/local_mistral_to_openai_50",
        help="Substring pattern passed to pgrep -f; queue starts after this process disappears.",
    )
    wait_parser.add_argument("--poll-sec", type=int, default=60)
    wait_parser.add_argument("--continue-on-error", action="store_true")
    wait_parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "prefix_family_sweep_qwen_local",
            "prefix_family_sweep_livebench_mistral_local",
            "prefix_family_sweep_livebench_qwen_local",
        ],
    )

    reentry_parser = subparsers.add_parser(
        "reentry",
        help="Manage large-scale re-entry mainline runs via run_reentry_mainline.py presets.",
    )
    reentry_parser.add_argument("--preset", default="reentry_livebench_local")
    reentry_parser.add_argument(
        "--phase", choices=["prepare", "reentry", "analyze", "probe", "both", "all"], default="both"
    )
    reentry_parser.add_argument(
        "--mode", choices=["run", "start", "status", "tail", "stop"], default="run"
    )
    reentry_parser.add_argument("--job-name", default=None)
    reentry_parser.add_argument("--tail-lines", type=int, default=80)
    reentry_parser.add_argument("--force", action="store_true")
    reentry_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "jobs":
        _run_job_mode(args)
    elif args.command == "reentry":
        _run_reentry_mode(args)
    else:
        _run_wait_queue(args)


if __name__ == "__main__":
    main()
