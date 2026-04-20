from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
JOB_DIR = ROOT / ".local_jobs"


def _job_paths(job_name: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{job_name}.log", JOB_DIR / f"{job_name}.pid"


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except Exception:
        return None


def _start_job(job_name: str, command: str) -> None:
    log_path, pid_path = _job_paths(job_name)
    existing_pid = _read_pid(pid_path)
    if existing_pid and _is_running(existing_pid):
        print(f"JOB_NAME={job_name}")
        print("STATUS=running")
        print(f"PID={existing_pid}")
        print(f"LOG={log_path}")
        return

    with log_path.open("ab") as lf:
        proc = subprocess.Popen(  # noqa: S603
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
    status = "running" if _is_running(pid) else "stopped"
    print(f"STATUS={status}")
    print(f"PID={pid}")
    print(f"LOG={log_path}")
    print(f"LOG_EXISTS={int(log_path.exists())}")


def _tail_job(job_name: str, lines: int) -> None:
    log_path, _ = _job_paths(job_name)
    if not log_path.exists():
        print(f"LOG_MISSING={log_path}")
        return
    cmd = ["tail", "-n", str(lines), str(log_path)]
    subprocess.run(cmd, check=False)


def _stop_job(job_name: str, force: bool) -> None:
    _, pid_path = _job_paths(job_name)
    pid = _read_pid(pid_path)
    if pid is None:
        print("STATUS=missing")
        return
    if not _is_running(pid):
        print("STATUS=already_stopped")
        pid_path.unlink(missing_ok=True)
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    print(f"STATUS=sent_{'sigkill' if force else 'sigterm'}")
    print(f"PID={pid}")


def _sweep_command(config_name: str, run_name_prefix: str) -> str:
    return (
        "uv run python experiments/run_prefix_family_sweep.py "
        f"--config-name {shlex.quote(config_name)} "
        f"launcher.run_name_prefix={shlex.quote(run_name_prefix)} "
        "launcher.overwrite_output_dir=false"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab Enterprise local launcher for prefix family sweeps (no SSH)."
    )
    parser.add_argument("--target", choices=["olympiad", "livebench", "both"], default="both")
    parser.add_argument(
        "--mode", choices=["run", "start", "status", "tail", "stop"], default="start"
    )
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--run-prefix-base", default="colab_rerun_")
    parser.add_argument(
        "--force", action="store_true", help="With --mode stop, send SIGKILL instead of SIGTERM."
    )
    args = parser.parse_args()

    jobs: list[tuple[str, str, str]] = []
    if args.target in {"olympiad", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_olympiad_colab",
                "prefix_family_sweep",
                args.run_prefix_base,
            )
        )
    if args.target in {"livebench", "both"}:
        jobs.append(
            (
                "prefix_family_sweep_livebench_colab",
                "prefix_family_sweep_livebench",
                f"{args.run_prefix_base}livebench_",
            )
        )

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


if __name__ == "__main__":
    main()
