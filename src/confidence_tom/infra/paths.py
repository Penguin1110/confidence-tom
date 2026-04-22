"""Project-wide path helpers.

The repository uses a single unified output root for generated artifacts.
The root can be configured via CONFIDENCE_TOM_OUTPUT_ROOT or pyproject
[tool.confidence_tom].output_root.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import tomllib

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT_PATH = _PROJECT_ROOT / "pyproject.toml"


@lru_cache(maxsize=1)
def project_root() -> Path:
    return _PROJECT_ROOT


@lru_cache(maxsize=1)
def output_root() -> Path:
    env_root = os.getenv("CONFIDENCE_TOM_OUTPUT_ROOT", "").strip()
    if env_root:
        return _resolve_root(env_root)

    if _PYPROJECT_PATH.exists():
        try:
            with open(_PYPROJECT_PATH, "rb") as f:
                data = tomllib.load(f)
            configured = str(
                data.get("tool", {}).get("confidence_tom", {}).get("output_root", "")
            ).strip()
            if configured:
                return _resolve_root(configured)
        except Exception:
            pass

    return _PROJECT_ROOT / "outputs"


@lru_cache(maxsize=1)
def results_root() -> Path:
    return output_root() / "results"


@lru_cache(maxsize=1)
def logs_root() -> Path:
    return output_root() / "logs"


def _resolve_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path
