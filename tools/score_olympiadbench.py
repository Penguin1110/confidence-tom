#!/usr/bin/env python3
"""Score one OlympiadBench prediction using the official MathJudger in an isolated env."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _candidate_repo_paths() -> list[Path]:
    cwd = Path.cwd()
    return [
        cwd / "external" / "OlympiadBench" / "inference" / "code",
        cwd / "OlympiadBench" / "inference" / "code",
        Path("/tmp/bench_inspect/OlympiadBench/inference/code"),
    ]


def _load_math_judger() -> type[object]:
    for path in _candidate_repo_paths():
        if not path.exists():
            continue
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        try:
            module = importlib.import_module("math_judger")
            return getattr(module, "MathJudger")
        except Exception:
            continue
    raise RuntimeError("Could not import OlympiadBench MathJudger from any known repo path")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--precision", default="1e-8")
    args = parser.parse_args()

    MathJudger = _load_math_judger()
    judger = MathJudger()
    precision: object = float(args.precision)
    result = bool(judger.judge(args.reference, args.prediction, precision))
    print(json.dumps({"is_correct": result, "score": 1.0 if result else 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
