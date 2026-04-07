from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def _print_cmd(cmd: list[str]) -> None:
    print("$", shlex.join(cmd), flush=True)


def _has_changes(repo: Path, paths: list[str]) -> bool:
    cmd = ["git", "status", "--porcelain", "--", *paths]
    out = _run(cmd, cwd=repo, check=False).stdout.strip()
    return bool(out)


def _ensure_branch(repo: Path, branch: str | None) -> str:
    if branch:
        _print_cmd(["git", "checkout", branch])
        _run(["git", "checkout", branch], cwd=repo)
        return branch
    current = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
    return current


def _sync_once(
    *,
    repo: Path,
    paths: list[str],
    remote: str,
    branch: str,
    message_prefix: str,
    do_rebase: bool,
) -> bool:
    if not _has_changes(repo, paths):
        print("No tracked changes in watched paths.", flush=True)
        return False

    add_cmd = ["git", "add", "--", *paths]
    _print_cmd(add_cmd)
    _run(add_cmd, cwd=repo)

    if not _has_changes(repo, paths):
        print("Nothing to commit after add.", flush=True)
        return False

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = f"{message_prefix} {ts}"
    commit_cmd = ["git", "commit", "-m", msg]
    _print_cmd(commit_cmd)
    commit = _run(commit_cmd, cwd=repo, check=False)
    if commit.returncode != 0:
        if "nothing to commit" in (commit.stdout + commit.stderr).lower():
            print("Commit skipped: nothing to commit.", flush=True)
            return False
        print(commit.stdout, end="")
        print(commit.stderr, end="", file=sys.stderr)
        raise RuntimeError("git commit failed")

    if do_rebase:
        pull_cmd = ["git", "pull", "--rebase", "--autostash", remote, branch]
        _print_cmd(pull_cmd)
        pull = _run(pull_cmd, cwd=repo, check=False)
        if pull.returncode != 0:
            print(pull.stdout, end="")
            print(pull.stderr, end="", file=sys.stderr)
            raise RuntimeError("git pull --rebase failed")

    push_cmd = ["git", "push", remote, branch]
    _print_cmd(push_cmd)
    push = _run(push_cmd, cwd=repo, check=False)
    if push.returncode != 0:
        print(push.stdout, end="")
        print(push.stderr, end="", file=sys.stderr)
        raise RuntimeError("git push failed")

    print("Pushed successfully.", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto commit+push selected paths periodically (useful on Colab)."
    )
    parser.add_argument("--repo", default=".", help="Path to git repo root.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Watched paths to add/commit/push (e.g. results logs).",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=None, help="Default: current branch.")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--message-prefix", default="chore(colab): checkpoint")
    parser.add_argument("--no-rebase", action="store_true", help="Skip pull --rebase before push.")
    parser.add_argument("--once", action="store_true", help="Run one sync cycle and exit.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    branch = _ensure_branch(repo, args.branch)
    print(f"repo={repo}", flush=True)
    print(f"branch={branch}", flush=True)
    print(f"paths={args.paths}", flush=True)

    if args.once:
        _sync_once(
            repo=repo,
            paths=args.paths,
            remote=args.remote,
            branch=branch,
            message_prefix=args.message_prefix,
            do_rebase=not args.no_rebase,
        )
        return

    print(f"Start autopush loop every {args.interval_sec}s ...", flush=True)
    while True:
        try:
            _sync_once(
                repo=repo,
                paths=args.paths,
                remote=args.remote,
                branch=branch,
                message_prefix=args.message_prefix,
                do_rebase=not args.no_rebase,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[autopush] error: {exc}", file=sys.stderr, flush=True)
        time.sleep(max(10, args.interval_sec))


if __name__ == "__main__":
    main()
