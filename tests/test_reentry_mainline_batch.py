from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "mainline" / "run" / "batch" / "run_reentry_mainline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_reentry_mainline", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reentry_presets_load() -> None:
    module = _load_module()
    presets = module.load_presets(module.PRESET_PATH)
    assert "reentry_livebench_local" in presets
    assert "reentry_olympiad_local" in presets


def test_build_reentry_command_includes_manifest_defaults() -> None:
    module = _load_module()
    preset = module.load_presets(module.PRESET_PATH)["reentry_livebench_local"]
    args = Namespace(
        output_dir=None,
        prepare_start_index=None,
        prepare_limit=None,
        small_backend=None,
        small_local_model_name=None,
        small_local_model_map=[],
        probe_output_dir=None,
        probe_backend=None,
        probe_local_model_name=None,
        probe_local_model_map=[],
        selected_layer=-1,
        run_name_prefix=[],
        benchmark=[],
        small_family=[],
        category=[],
        max_rows=5,
        concurrency=2,
        max_tokens=1024,
        full_rerun_temperature=0.0,
        reentry_temperature=0.0,
    )
    cmd = module.build_reentry_cmd("reentry_livebench_local", preset, args)
    rendered = " ".join(cmd)
    assert "--run-name-prefix reentry_livebench_" in rendered
    assert "--benchmark livebench_reasoning" in rendered
    assert "--small-local-model-map qwen=Qwen/Qwen3-14B" in rendered
    assert "--max-rows 5" in rendered


def test_build_prepare_command_supports_prepare_shards() -> None:
    module = _load_module()
    preset = module.load_presets(module.PRESET_PATH)["reentry_livebench_local"]
    args = Namespace(
        benchmark=[],
        small_family=[],
        prepare_start_index=10,
        prepare_limit=5,
    )
    cmd = module.build_prepare_cmd(str(preset["family_sweep_config"]), preset, args)
    rendered = " ".join(cmd)
    assert "+dataset.start_index=10" in rendered
    assert "dataset.limit=5" in rendered
    assert "dataset.livebench=5" in rendered
    assert "dataset.livebench_reasoning=5" in rendered


def test_build_prepare_command_supports_small_family_filter() -> None:
    module = _load_module()
    preset = module.load_presets(module.PRESET_PATH)["reentry_livebench_local"]
    args = Namespace(
        benchmark=[],
        small_family=["qwen3"],
        prepare_start_index=None,
        prepare_limit=None,
    )
    cmd = module.build_prepare_cmd(str(preset["family_sweep_config"]), preset, args)
    rendered = " ".join(cmd)
    assert "launcher.only_small_families=[qwen3]" in rendered


def test_build_probe_command_includes_manifest_defaults() -> None:
    module = _load_module()
    preset = module.load_presets(module.PRESET_PATH)["reentry_livebench_local"]
    args = Namespace(
        output_dir=None,
        probe_output_dir=None,
        probe_backend=None,
        probe_local_model_name=None,
        probe_local_model_map=[],
        selected_layer=-1,
        max_rows=3,
    )
    cmd = module.build_probe_cmd(args, preset)
    rendered = " ".join(cmd)
    assert "run_prefix_reentry_probe.py" in rendered
    assert "--backend transformers" in rendered
    assert "--local-model-map qwen=Qwen/Qwen3-14B" in rendered
    assert "--max-rows 3" in rendered
