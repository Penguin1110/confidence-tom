from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_controls.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_prefix_reentry_controls", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_family_model_map_accepts_valid_entries() -> None:
    module = _load_module()
    mapping = module._parse_family_model_map(["qwen=qwen3:14b", "gemma=gemma3:4b"])
    assert mapping == {"qwen": "qwen3:14b", "gemma": "gemma3:4b"}


def test_parse_family_model_map_rejects_invalid_entries() -> None:
    module = _load_module()
    try:
        module._parse_family_model_map(["not-valid"])
    except ValueError as exc:
        assert "Expected FAMILY=MODEL" in str(exc)
    else:
        raise AssertionError("expected invalid re-entry family map to raise ValueError")


def test_resolve_local_model_name_uses_backend_defaults_and_overrides() -> None:
    module = _load_module()
    assert (
        module._resolve_local_model_name("qwen", None, "ollama", {}) == "qwen3:14b"
    )
    assert (
        module._resolve_local_model_name("qwen", None, "local", {}) == "Qwen/Qwen3-14B"
    )
    assert (
        module._resolve_local_model_name(
            "qwen", None, "ollama", {"qwen": "qwen3.5:27b"}
        )
        == "qwen3.5:27b"
    )
