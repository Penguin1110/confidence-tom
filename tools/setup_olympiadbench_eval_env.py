#!/usr/bin/env python3
"""Create an isolated venv for OlympiadBench official scoring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env_dir = root / ".venvs" / "olympiadbench-eval"
    env_dir.parent.mkdir(parents=True, exist_ok=True)

    if not env_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)

    py = env_dir / "bin" / "python"
    pip = [str(py), "-m", "pip"]
    subprocess.run(pip + ["install", "--upgrade", "pip"], check=True)
    subprocess.run(
        pip + ["install", "sympy>=1.14.0", "antlr4-python3-runtime==4.11.*"],
        check=True,
    )
    print(py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
