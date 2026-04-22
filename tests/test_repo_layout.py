from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_and_compat_layout() -> None:
    expected_exists = [
        ROOT / "src" / "confidence_tom" / "infra",
        ROOT / "src" / "confidence_tom" / "data",
        ROOT / "src" / "confidence_tom" / "eval",
        ROOT / "src" / "confidence_tom" / "compat",
        ROOT / "src" / "confidence_tom" / "compat" / "generator",
        ROOT / "src" / "confidence_tom" / "compat" / "observer",
        ROOT / "src" / "confidence_tom" / "generator",
        ROOT / "src" / "confidence_tom" / "observer",
        ROOT / "experiments" / "mainline",
        ROOT / "docs" / "mainline",
        ROOT / "experiments" / "mainline" / "run",
        ROOT / "experiments" / "mainline" / "run" / "core",
        ROOT / "experiments" / "mainline" / "run" / "batch",
        ROOT / "experiments" / "mainline" / "run" / "remote",
        ROOT / "experiments" / "mainline" / "analysis",
        ROOT / "experiments" / "mainline" / "analysis" / "prefix",
        ROOT / "experiments" / "mainline" / "analysis" / "trace",
        ROOT / "experiments" / "mainline" / "analysis" / "routing",
        ROOT / "experiments" / "mainline" / "analysis" / "embedding",
        ROOT / "experiments" / "mainline" / "analysis" / "maintenance",
        ROOT / "experiments" / "mainline" / "data",
        ROOT / "docs" / "mainline" / "notes",
        ROOT / "docs" / "mainline" / "notes" / "reports",
        ROOT / "docs" / "mainline" / "notes" / "proposals",
        ROOT / "docs" / "mainline" / "generated",
        ROOT / "docs" / "mainline" / "generated" / "analysis",
        ROOT / "docs" / "mainline" / "generated" / "analysis" / "prefix",
        ROOT / "docs" / "mainline" / "generated" / "analysis" / "trace",
        ROOT / "docs" / "mainline" / "generated" / "analysis" / "routing",
        ROOT / "docs" / "mainline" / "generated" / "analysis" / "embedding",
        ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_controls.py",
        ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_oracle_gain_mapping.py",
        ROOT / "experiments" / "mainline" / "run" / "core" / "run_api_determinism_audit.py",
        ROOT / "experiments" / "mainline" / "run" / "batch" / "run_family_queue.py",
        ROOT / "experiments" / "mainline" / "run" / "batch" / "run_small_only.py",
        ROOT / "experiments" / "mainline" / "analysis" / "prefix" / "analyze_prefix_diagnostics.py",
        ROOT / "experiments" / "mainline" / "analysis" / "maintenance" / "manage_prefix_results.py",
        ROOT / "docs" / "mainline" / "notes" / "reports" / "prefix_based_intervention_framework.md",
        ROOT / "docs" / "mainline" / "notes" / "reports" / "prefix_case_study_candidates.md",
        ROOT / "docs" / "mainline" / "notes" / "proposals" / "oracle_gain_fragility_proposal.md",
        ROOT
        / "docs"
        / "mainline"
        / "generated"
        / "analysis"
        / "prefix"
        / "prefix_reentry_controls.md",
        ROOT
        / "docs"
        / "mainline"
        / "generated"
        / "analysis"
        / "trace"
        / "trace_taxonomy_analysis.md",
        ROOT / "src" / "confidence_tom" / "infra" / "client.py",
        ROOT / "src" / "confidence_tom" / "data" / "task_models.py",
        ROOT / "src" / "confidence_tom" / "eval" / "static_evaluators.py",
        ROOT / "src" / "confidence_tom" / "eval" / "parsing.py",
        ROOT / "src" / "confidence_tom" / "data" / "dynamic_benchmarks.py",
    ]
    for path in expected_exists:
        assert path.exists(), f"missing expected path: {path.relative_to(ROOT)}"


def test_old_top_level_entrypoints_are_gone() -> None:
    expected_absent = [
        ROOT / "src" / "confidence_tom" / "legacy",
        ROOT / "src" / "confidence_tom" / "archive",
        ROOT / "src" / "confidence_tom" / "client.py",
        ROOT / "src" / "confidence_tom" / "dataset_models.py",
        ROOT / "src" / "confidence_tom" / "evaluators.py",
        ROOT / "src" / "confidence_tom" / "metrics.py",
        ROOT / "src" / "confidence_tom" / "model_config.py",
        ROOT / "src" / "confidence_tom" / "paths.py",
        ROOT / "src" / "confidence_tom" / "scale_dataset.py",
        ROOT / "src" / "confidence_tom" / "static_evaluators.py",
        ROOT / "src" / "confidence_tom" / "task_models.py",
        ROOT / "experiments" / "archive",
        ROOT / "docs" / "archive",
        ROOT / "docs" / "mainline" / "analysis",
        ROOT / "docs" / "mainline" / "reports",
        ROOT / "docs" / "mainline" / "proposals",
        ROOT / "experiments" / "mainline" / "run_prefix_reentry_controls.py",
        ROOT / "experiments" / "mainline" / "run_prefix_oracle_gain_mapping.py",
        ROOT / "experiments" / "mainline" / "run_generator.py",
        ROOT / "experiments" / "mainline" / "run_observer.py",
        ROOT / "experiments" / "mainline" / "analyze_trace_taxonomy.py",
        ROOT / "docs" / "mainline" / "prefix_reentry_controls.md",
        ROOT / "docs" / "mainline" / "trace_taxonomy_analysis.md",
        ROOT / "docs" / "mainline" / "early_decision_baseline.md",
        ROOT / "docs" / "mainline" / "dynamic-benchmarks.md",
        ROOT / "experiments" / "mainline" / "run" / "run_prefix_reentry_controls.py",
        ROOT / "experiments" / "mainline" / "run" / "run_prefix_oracle_gain_mapping.py",
        ROOT / "experiments" / "mainline" / "run" / "run_api_determinism_audit.py",
        ROOT / "experiments" / "mainline" / "analysis" / "plot_prefix_task_structure.py",
        ROOT / "experiments" / "mainline" / "run" / "batch" / "run_colab_prefix_family_rerun.py",
        ROOT / "experiments" / "mainline" / "run" / "batch" / "run_local_family_queue.py",
        ROOT / "experiments" / "mainline" / "run" / "batch" / "run_prefix_small_only_mapping.py",
        ROOT
        / "experiments"
        / "mainline"
        / "run"
        / "batch"
        / "run_prefix_small_only_full_matrix.py",
        ROOT / "experiments" / "mainline" / "analysis" / "analyze_prefix_signal_patterns.py",
        ROOT / "experiments" / "mainline" / "analysis" / "analyze_prefix_predictor_failure.py",
        ROOT / "experiments" / "mainline" / "analysis" / "backfill_large_takeover_from_small.py",
        ROOT / "experiments" / "mainline" / "analysis" / "show_prefix_progress.py",
        ROOT / "experiments" / "mainline" / "analysis" / "analyze_prefix_diagnostics.py",
        ROOT / "experiments" / "mainline" / "analysis" / "manage_prefix_results.py",
    ]
    for path in expected_absent:
        assert not path.exists(), f"old top-level path should be removed: {path.relative_to(ROOT)}"


def test_mainline_root_is_subdivided() -> None:
    mainline_root = ROOT / "experiments" / "mainline"
    assert sorted(p.name for p in mainline_root.glob("*.py")) == ["__init__.py"]
    assert sorted(p.name for p in mainline_root.glob("*.md")) == ["README.md"]

    docs_mainline_root = ROOT / "docs" / "mainline"
    assert not any(docs_mainline_root.glob("*_analysis.md"))
    assert sorted(p.name for p in docs_mainline_root.glob("*.md")) == ["README.md"]
