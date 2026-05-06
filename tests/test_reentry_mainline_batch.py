from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "mainline" / "run" / "batch" / "run_reentry_mainline.py"
FAMILY_SWEEP_MODULE_PATH = (
    ROOT / "experiments" / "mainline" / "run" / "batch" / "run_prefix_family_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_reentry_mainline", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_family_sweep_module():
    spec = importlib.util.spec_from_file_location("run_prefix_family_sweep", FAMILY_SWEEP_MODULE_PATH)
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
    livebench_maps = presets["reentry_livebench_local"]["small_local_model_map"]
    assert "qwen3=Qwen/Qwen3-14B" in livebench_maps
    assert "qwen25=Qwen/Qwen2.5-14B-Instruct" in livebench_maps
    assert "gemma4=google/gemma-4-E4B-it" in livebench_maps
    assert "gemma3=google/gemma-3-4b-it" in livebench_maps
    assert "ministral=mistralai/Ministral-8B-Instruct-2410" in livebench_maps
    assert "mistral7=mistralai/Mistral-7B-Instruct-v0.3" in livebench_maps


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
        task_start_index=3,
        task_limit=2,
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
    assert "--small-local-model-map qwen3=Qwen/Qwen3-14B" in rendered
    assert "--small-local-model-map qwen25=Qwen/Qwen2.5-14B-Instruct" in rendered
    assert "--task-start-index 3" in rendered
    assert "--task-limit 2" in rendered
    assert "--max-rows 5" in rendered


def test_single_family_reentry_defaults_to_family_specific_output_dirs() -> None:
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
        small_family=["qwen3"],
        category=[],
        task_start_index=None,
        task_limit=None,
        max_rows=None,
        concurrency=2,
        max_tokens=1024,
        full_rerun_temperature=0.0,
        reentry_temperature=0.0,
    )
    reentry_cmd = module.build_reentry_cmd("reentry_livebench_local", preset, args)
    analyze_cmd = module.build_analyze_cmd(args, preset)
    probe_cmd = module.build_probe_cmd(args, preset)

    reentry_rendered = " ".join(reentry_cmd).replace("\\", "/")
    analyze_rendered = " ".join(analyze_cmd).replace("\\", "/")
    probe_rendered = " ".join(probe_cmd).replace("\\", "/")

    assert "outputs/results/_reentry_livebench_local_v1_qwen3" in reentry_rendered
    assert "prefix_reentry_controls_qwen3.md" in analyze_rendered
    assert "outputs/results/_reentry_livebench_local_v1_qwen3/probe" in probe_rendered


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
    assert "+launcher.only_small_families=[qwen3]" in rendered


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
        small_family=[],
    )
    cmd = module.build_probe_cmd(args, preset)
    rendered = " ".join(cmd)
    assert "run_prefix_reentry_probe.py" in rendered
    assert "--backend transformers" in rendered
    assert "--local-model-map qwen3=Qwen/Qwen3-14B" in rendered
    assert "--local-model-map qwen25=Qwen/Qwen2.5-14B-Instruct" in rendered
    assert "--max-rows 3" in rendered


def test_family_sweep_run_name_includes_shard_suffix_for_reentry_prepare() -> None:
    module = _load_family_sweep_module()
    run_name = module._run_name_for_config(
        run_name_prefix="reentry_livebench_",
        small_family="qwen3",
        large_family="reentry_prepare",
        limit=1,
        prepare_mode="small_only_reentry",
        start_index=20,
    )
    assert run_name == "reentry_livebench_qwen3_s020_1"


def test_reentry_controls_expand_segments_when_prefix_steps_missing() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_prefix_reentry_controls",
        ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_controls.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    task = {
        "task_id": "livebench_reasoning_demo_0000",
        "small_model": "Qwen/Qwen3-14B",
        "full_trace_answer": "demo",
        "full_trace_correct": False,
        "segments": [
            {"index": 1, "text": "step one"},
            {"index": 2, "text": "step two"},
        ],
        "prefix_oracle_steps": [],
        "metadata": {"prepare_mode": "segments_only"},
    }
    result_json = ROOT / "tmp_reentry_controls_test.json"
    result_json.write_text("[\n  " + __import__("json").dumps(task) + "\n]", encoding="utf-8")
    try:
        original_find = module._find_result_json
        module._find_result_json = lambda run_name: result_json
        rows = module._load_prefix_rows(["reentry_livebench_qwen3_s000_1"], None)
    finally:
        module._find_result_json = original_find
        result_json.unlink(missing_ok=True)

    assert len(rows) == 2
    assert rows[0]["prefix_text"] == "step one"
    assert rows[1]["prefix_text"] == "step one\n\nstep two"
    assert rows[0]["prepare_mode"] == "segments_only"


def test_reentry_controls_find_result_json_skips_run_metadata(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_prefix_reentry_controls",
        ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_controls.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = tmp_path / "reentry_livebench_qwen3_1"
    run_dir.mkdir()
    status_json = run_dir / "_run_status.json"
    result_json = run_dir / "HF_Qwen3_14B_small_only.json"
    status_json.write_text('{"status": "completed"}', encoding="utf-8")
    result_json.write_text(
        '[{"task_id": "livebench_reasoning_demo_0000", "prefix_oracle_steps": []}]',
        encoding="utf-8",
    )

    original_candidates = module.RESULT_DIR_CANDIDATES
    try:
        module.RESULT_DIR_CANDIDATES = [tmp_path]
        assert module._find_result_json("reentry_livebench_qwen3_1") == result_json
    finally:
        module.RESULT_DIR_CANDIDATES = original_candidates
