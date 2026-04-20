from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import pexpect
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDES = [
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    "outputs/",
    "wandb/",
    ".DS_Store",
]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _build_ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}"


def _remote_path(remote_dir: str) -> str:
    if remote_dir == "~":
        return "$HOME"
    if remote_dir.startswith("~/"):
        return f"$HOME/{shlex.quote(remote_dir[2:])}"
    return shlex.quote(remote_dir)


def _stream_command(command: str, password: str, cwd: Path | None = None, timeout: int = 30) -> int:
    child = pexpect.spawn(
        "/bin/bash", ["-lc", command], cwd=str(cwd) if cwd else None, encoding="utf-8"
    )
    child.logfile_read = sys.stdout

    while True:
        index = child.expect(
            [
                r"Are you sure you want to continue connecting \(yes/no(/\[fingerprint\])?\)\?",
                r"(?i)(?:password|passphrase).*:",
                pexpect.EOF,
                pexpect.TIMEOUT,
            ],
            timeout=timeout,
        )
        if index == 0:
            child.sendline("yes")
            continue
        if index == 1:
            child.sendline(password)
            continue
        if index == 2:
            break
        raise SystemExit(f"Command timed out after {timeout}s: {command}")

    child.close()
    if child.exitstatus is not None:
        return child.exitstatus
    return 1 if child.signalstatus is not None else 0


def _remote_ssh_command(target: str, remote_cmd: str) -> str:
    return f"ssh -o StrictHostKeyChecking=no {shlex.quote(target)} {shlex.quote(remote_cmd)}"


def _ensure_remote_dir(target: str, password: str, remote_dir: str, timeout: int) -> None:
    mkdir_cmd = _remote_ssh_command(target, f"mkdir -p {_remote_path(remote_dir)}")
    exit_code = _stream_command(mkdir_cmd, password=password, cwd=REPO_ROOT, timeout=timeout)
    if exit_code != 0:
        raise SystemExit(f"Failed to prepare remote directory: {remote_dir}")


def _sync_repo(target: str, password: str, remote_dir: str, timeout: int, delete: bool) -> None:
    _ensure_remote_dir(target, password=password, remote_dir=remote_dir, timeout=timeout)

    exclude_flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in DEFAULT_EXCLUDES)
    delete_flag = "--delete " if delete else ""
    rsync_cmd = (
        f"rsync -az {delete_flag}{exclude_flags} "
        f"{shlex.quote(str(REPO_ROOT))}/ {shlex.quote(target)}:{_remote_path(remote_dir)}/"
    )
    exit_code = _stream_command(rsync_cmd, password=password, cwd=REPO_ROOT, timeout=timeout)
    if exit_code != 0:
        raise SystemExit("Repo sync failed.")


def cmd_ping(args: argparse.Namespace) -> int:
    target = _build_ssh_target(args.remote_user, args.remote_host)
    command = _remote_ssh_command(target, "hostname && whoami && pwd")
    return _stream_command(
        command, password=args.remote_password, cwd=REPO_ROOT, timeout=args.timeout
    )


def cmd_sync(args: argparse.Namespace) -> int:
    target = _build_ssh_target(args.remote_user, args.remote_host)
    _sync_repo(
        target,
        password=args.remote_password,
        remote_dir=args.remote_dir,
        timeout=args.timeout,
        delete=args.delete,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    command_parts = list(args.command)
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]

    if not command_parts:
        raise SystemExit("run requires a command after '--'")

    target = _build_ssh_target(args.remote_user, args.remote_host)
    if not args.skip_sync:
        _sync_repo(
            target,
            password=args.remote_password,
            remote_dir=args.remote_dir,
            timeout=args.timeout,
            delete=args.delete,
        )
    remote_cmd = f"cd {_remote_path(args.remote_dir)} && {shlex.join(command_parts)}"
    command = _remote_ssh_command(target, remote_cmd)
    return _stream_command(
        command, password=args.remote_password, cwd=REPO_ROOT, timeout=args.timeout
    )


def cmd_run_bg(args: argparse.Namespace) -> int:
    command_parts = list(args.command)
    if command_parts and command_parts[0] == "--":
        command_parts = command_parts[1:]
    if not command_parts:
        raise SystemExit("run-bg requires a command after '--'")

    target = _build_ssh_target(args.remote_user, args.remote_host)
    if not args.skip_sync:
        _sync_repo(
            target,
            password=args.remote_password,
            remote_dir=args.remote_dir,
            timeout=args.timeout,
            delete=args.delete,
        )

    job_name = args.job_name or "remote_job"
    remote_logs_dir = f"{args.remote_dir}/logs"
    remote_job_dir = f"{args.remote_dir}/.remote_jobs"
    remote_log = f"{remote_logs_dir}/{job_name}.log"
    remote_pid = f"{remote_job_dir}/{job_name}.pid"
    remote_cmd = (
        f"mkdir -p {_remote_path(remote_logs_dir)} {_remote_path(remote_job_dir)} && "
        f"cd {_remote_path(args.remote_dir)} && "
        f"nohup {shlex.join(command_parts)} > {_remote_path(remote_log)} 2>&1 < /dev/null & "
        f"echo $! > {_remote_path(remote_pid)} && "
        f"echo JOB_NAME={shlex.quote(job_name)} && "
        f"echo PID=$(cat {_remote_path(remote_pid)}) && "
        f"echo LOG={_remote_path(remote_log)}"
    )
    command = _remote_ssh_command(target, remote_cmd)
    return _stream_command(
        command, password=args.remote_password, cwd=REPO_ROOT, timeout=args.timeout
    )


def cmd_status(args: argparse.Namespace) -> int:
    job_name = args.job_name or "remote_job"
    target = _build_ssh_target(args.remote_user, args.remote_host)
    remote_pid = f"{args.remote_dir}/.remote_jobs/{job_name}.pid"
    remote_log = f"{args.remote_dir}/logs/{job_name}.log"
    remote_cmd = (
        f"if [ ! -f {_remote_path(remote_pid)} ]; then "
        f"echo STATUS=missing; exit 0; fi; "
        f"pid=$(cat {_remote_path(remote_pid)}); "
        f'if ps -p "$pid" > /dev/null 2>&1; then '
        f"echo STATUS=running; "
        f"else echo STATUS=stopped; fi; "
        f"echo PID=$pid; "
        f"echo LOG={_remote_path(remote_log)}; "
        f"if [ -f {_remote_path(remote_log)} ]; then echo LOG_EXISTS=1; else echo LOG_EXISTS=0; fi"
    )
    command = _remote_ssh_command(target, remote_cmd)
    return _stream_command(
        command, password=args.remote_password, cwd=REPO_ROOT, timeout=args.timeout
    )


def cmd_tail(args: argparse.Namespace) -> int:
    job_name = args.job_name or "remote_job"
    lines = int(args.lines)
    target = _build_ssh_target(args.remote_user, args.remote_host)
    remote_log = f"{args.remote_dir}/logs/{job_name}.log"
    remote_cmd = (
        f"if [ -f {_remote_path(remote_log)} ]; then "
        f"tail -n {lines} {_remote_path(remote_log)}; "
        f"else echo LOG_MISSING={_remote_path(remote_log)}; fi"
    )
    command = _remote_ssh_command(target, remote_cmd)
    return _stream_command(
        command, password=args.remote_password, cwd=REPO_ROOT, timeout=args.timeout
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync this repo to a remote machine and run experiments there."
    )
    parser.add_argument("--remote-host", default=os.environ.get("REMOTE_SSH_HOST", ""))
    parser.add_argument("--remote-user", default=os.environ.get("REMOTE_SSH_USER", ""))
    parser.add_argument("--remote-password", default=os.environ.get("REMOTE_SSH_PASSWORD", ""))
    parser.add_argument(
        "--remote-dir",
        default=os.environ.get("REMOTE_PROJECT_DIR", "~/confidence-tom"),
    )
    parser.add_argument("--timeout", type=int, default=300)

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    ping = subparsers.add_parser("ping", help="Verify remote login works.")
    ping.set_defaults(handler=cmd_ping)

    sync = subparsers.add_parser("sync", help="Sync the repo to the remote machine.")
    sync.add_argument(
        "--delete", action="store_true", help="Delete remote files not present locally."
    )
    sync.set_defaults(handler=cmd_sync)

    run = subparsers.add_parser("run", help="Sync and run a command on the remote machine.")
    run.add_argument("--skip-sync", action="store_true", help="Run without syncing first.")
    run.add_argument(
        "--delete", action="store_true", help="Delete remote files not present locally."
    )
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=cmd_run)

    run_bg = subparsers.add_parser(
        "run-bg", help="Sync and start a background job on the remote machine."
    )
    run_bg.add_argument("--skip-sync", action="store_true", help="Run without syncing first.")
    run_bg.add_argument(
        "--delete", action="store_true", help="Delete remote files not present locally."
    )
    run_bg.add_argument("--job-name", default="remote_job")
    run_bg.add_argument("command", nargs=argparse.REMAINDER)
    run_bg.set_defaults(handler=cmd_run_bg)

    status = subparsers.add_parser("status", help="Check status for a remote background job.")
    status.add_argument("--job-name", default="remote_job")
    status.set_defaults(handler=cmd_status)

    tail = subparsers.add_parser("tail", help="Show tail of a remote background job log.")
    tail.add_argument("--job-name", default="remote_job")
    tail.add_argument("--lines", type=int, default=40)
    tail.set_defaults(handler=cmd_tail)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.remote_host = args.remote_host or _require_env("REMOTE_SSH_HOST")
    args.remote_user = args.remote_user or _require_env("REMOTE_SSH_USER")
    args.remote_password = args.remote_password or _require_env("REMOTE_SSH_PASSWORD")

    exit_code = args.handler(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
