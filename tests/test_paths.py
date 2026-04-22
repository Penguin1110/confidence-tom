from __future__ import annotations

from pathlib import Path

import tomllib
from _pytest.monkeypatch import MonkeyPatch

import confidence_tom.infra.paths as paths

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_unified_output_root() -> None:
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert data["tool"]["confidence_tom"]["output_root"] == "outputs"


def test_output_root_defaults_to_outputs_and_supports_env_override(
    monkeypatch: MonkeyPatch,
) -> None:
    paths.output_root.cache_clear()
    paths.results_root.cache_clear()
    paths.logs_root.cache_clear()

    monkeypatch.delenv("CONFIDENCE_TOM_OUTPUT_ROOT", raising=False)
    assert paths.output_root() == ROOT / "outputs"
    assert paths.results_root() == ROOT / "outputs" / "results"
    assert paths.logs_root() == ROOT / "outputs" / "logs"

    paths.output_root.cache_clear()
    paths.results_root.cache_clear()
    paths.logs_root.cache_clear()

    monkeypatch.setenv("CONFIDENCE_TOM_OUTPUT_ROOT", "/tmp/confidence-tom-out")
    assert paths.output_root() == Path("/tmp/confidence-tom-out")
    assert paths.results_root() == Path("/tmp/confidence-tom-out") / "results"
    assert paths.logs_root() == Path("/tmp/confidence-tom-out") / "logs"
