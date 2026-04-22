from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from confidence_tom.infra.paths import output_root, project_root

ROOT = project_root()
JOBS_DIR = output_root() / "jobs"
REMOTE_HOST = os.environ.get("REMOTE_SSH_HOST", "").strip()
REMOTE_USER = os.environ.get("REMOTE_SSH_USER", "").strip()
REMOTE_PASSWORD = os.environ.get("REMOTE_SSH_PASSWORD", "").strip()
REMOTE_PROJECT_DIR = os.environ.get("REMOTE_PROJECT_DIR", "~/confidence-tom").strip()


def _remote_target() -> str:
    if not REMOTE_HOST:
        raise SystemExit("REMOTE_SSH_HOST is not set")
    return f"{REMOTE_USER}@{REMOTE_HOST}" if REMOTE_USER else REMOTE_HOST


def _remote_cmd_prefix() -> list[str]:
    target = _remote_target()
    if REMOTE_PASSWORD:
        return ["sshpass", "-p", REMOTE_PASSWORD, "ssh", "-o", "StrictHostKeyChecking=no", target]
    return ["ssh", "-o", "StrictHostKeyChecking=no", target]


def _run(cmd: list[str]) -> int:
    print("$", shlex.join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def _remote_bash(script: str) -> int:
    return _run([*_remote_cmd_prefix(), "bash", "-lc", script])


def _job_dir(name: str) -> Path:
    path = JOBS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_sync(args: argparse.Namespace) -> int:
    remote_dir = REMOTE_PROJECT_DIR.rstrip("/")
    mkdir_rc = _remote_bash(f"mkdir -p {shlex.quote(remote_dir)}")
    if mkdir_rc != 0:
        return mkdir_rc
    cmd = (
        "tar "
        "--exclude=.git "
        "--exclude=outputs "
        "--exclude=.mypy_cache "
        "--exclude=.pytest_cache "
        "--exclude=.ruff_cache "
        "-cf - . "
        f"| {' '.join(shlex.quote(part) for part in _remote_cmd_prefix())} "
        f"bash -lc 'mkdir -p {shlex.quote(remote_dir)} && tar -xf - -C {shlex.quote(remote_dir)}'"
    )
    return _run(["bash", "-lc", cmd])


def cmd_run(args: argparse.Namespace, *, background: bool) -> int:
    if not args.command:
        raise SystemExit("run/run-bg requires a command after '--'")
    command = shlex.join(args.command)
    if background:
        if not args.job_name:
            raise SystemExit("run-bg requires --job-name")
        job_dir = _job_dir(args.job_name)
        remote_log_dir = f"{REMOTE_PROJECT_DIR.rstrip('/')}/outputs/logs"
        remote_log_path = f"{remote_log_dir}/{args.job_name}.log"
        remote_script = (
            f"mkdir -p {shlex.quote(remote_log_dir)} && "
            f"cd {shlex.quote(REMOTE_PROJECT_DIR)} && "
            f"nohup {command} > {shlex.quote(remote_log_path)} 2>&1 & echo $!"
        )
        result = subprocess.run(
            [*_remote_cmd_prefix(), "bash", "-lc", remote_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return result.returncode
        pid = result.stdout.strip().splitlines()[-1].strip()
        (job_dir / "pid").write_text(pid + "\n")
        (job_dir / "log_path").write_text(remote_log_path + "\n")
        print(f"Started {args.job_name} pid={pid} log={remote_log_path}")
        return 0
    return _remote_bash(f"cd {shlex.quote(REMOTE_PROJECT_DIR)} && {command}")


def cmd_status(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args.job_name)
    pid_path = job_dir / "pid"
    if not pid_path.exists():
        print("STATUS=missing")
        return 0
    pid = pid_path.read_text().strip()
    rc = _remote_bash(f"kill -0 {shlex.quote(pid)} >/dev/null 2>&1")
    print(f"STATUS={'running' if rc == 0 else 'stopped'}")
    print(f"PID={pid}")
    log_path = job_dir / "log_path"
    if log_path.exists():
        print(f"LOG={log_path.read_text().strip()}")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args.job_name)
    log_path_file = job_dir / "log_path"
    if not log_path_file.exists():
        raise SystemExit(f"No log path for job {args.job_name}")
    log_path = log_path_file.read_text().strip()
    return _remote_bash(f"tail -n {int(args.lines)} {shlex.quote(log_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote job helper for mainline experiments")
    parser.add_argument("--timeout", type=int, default=7200)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sync")

    run_p = sub.add_parser("run")
    run_p.add_argument("--skip-sync", action="store_true")
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    bg_p = sub.add_parser("run-bg")
    bg_p.add_argument("--skip-sync", action="store_true")
    bg_p.add_argument("--job-name", required=True)
    bg_p.add_argument("command", nargs=argparse.REMAINDER)

    status_p = sub.add_parser("status")
    status_p.add_argument("--job-name", required=True)

    tail_p = sub.add_parser("tail")
    tail_p.add_argument("--job-name", required=True)
    tail_p.add_argument("--lines", type=int, default=60)

    args = parser.parse_args()
    if args.cmd == "sync":
        raise SystemExit(cmd_sync(args))
    if args.cmd == "run":
        cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
        args.command = cmd
        raise SystemExit(cmd_run(args, background=False))
    if args.cmd == "run-bg":
        cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
        args.command = cmd
        raise SystemExit(cmd_run(args, background=True))
    if args.cmd == "status":
        raise SystemExit(cmd_status(args))
    if args.cmd == "tail":
        raise SystemExit(cmd_tail(args))


if __name__ == "__main__":
    main()
