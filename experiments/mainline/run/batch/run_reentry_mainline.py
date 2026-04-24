from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from confidence_tom.infra.paths import project_root

ROOT = project_root()
PRESET_PATH = ROOT / "experiments" / "mainline" / "run" / "batch" / "reentry_presets.json"
FAMILY_SWEEP = ROOT / "experiments" / "mainline" / "run" / "batch" / "run_prefix_family_sweep.py"
REENTRY_RUNNER = ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_controls.py"
REENTRY_ANALYZER = (
    ROOT / "experiments" / "mainline" / "analysis" / "prefix" / "analyze_prefix_reentry_controls.py"
)
REENTRY_PROBE = ROOT / "experiments" / "mainline" / "run" / "core" / "run_prefix_reentry_probe.py"
DEFAULT_SUMMARY_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "prefix" / "prefix_reentry_controls.md"
)


def load_presets(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid preset file: {path}")
    return {str(key): dict(value) for key, value in presets.items()}


def build_prepare_cmd(config_name: str) -> list[str]:
    return ["uv", "run", "python", str(FAMILY_SWEEP), "--config-name", config_name]


def build_reentry_cmd(preset_name: str, preset: dict[str, Any], args: argparse.Namespace) -> list[str]:
    output_dir = str(args.output_dir or preset["reentry_output_dir"])
    cmd = [
        "uv",
        "run",
        "python",
        str(REENTRY_RUNNER),
        "--output-dir",
        output_dir,
        "--small-backend",
        str(args.small_backend or preset["small_backend"]),
        "--concurrency",
        str(args.concurrency),
        "--max-tokens",
        str(args.max_tokens),
        "--full-rerun-temperature",
        str(args.full_rerun_temperature),
        "--reentry-temperature",
        str(args.reentry_temperature),
    ]
    for prefix in args.run_name_prefix or preset.get("run_name_prefixes", []):
        cmd += ["--run-name-prefix", str(prefix)]
    for benchmark in args.benchmark or preset.get("benchmarks", []):
        cmd += ["--benchmark", str(benchmark)]
    for category in args.category or preset.get("categories", []):
        cmd += ["--category", str(category)]
    for family in args.small_family:
        cmd += ["--small-family", str(family)]
    if args.max_rows is not None:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.small_local_model_name:
        cmd += ["--small-local-model-name", str(args.small_local_model_name)]
    for item in args.small_local_model_map or preset.get("small_local_model_map", []):
        cmd += ["--small-local-model-map", str(item)]
    return cmd


def build_analyze_cmd(args: argparse.Namespace, preset: dict[str, Any]) -> list[str]:
    output_dir = Path(str(args.output_dir or preset["reentry_output_dir"]))
    return [
        "uv",
        "run",
        "python",
        str(REENTRY_ANALYZER),
        "--rows",
        str(output_dir / "reentry_rows.jsonl"),
        "--summary-json",
        str(output_dir / "reentry_summary.json"),
        "--summary-md",
        str(DEFAULT_SUMMARY_MD),
    ]


def build_probe_cmd(args: argparse.Namespace, preset: dict[str, Any]) -> list[str]:
    reentry_output_dir = Path(str(args.output_dir or preset["reentry_output_dir"]))
    probe_output_dir = str(args.probe_output_dir or preset.get("probe_output_dir") or reentry_output_dir)
    cmd = [
        "uv",
        "run",
        "python",
        str(REENTRY_PROBE),
        "--rows",
        str(reentry_output_dir / "reentry_rows.jsonl"),
        "--output-dir",
        probe_output_dir,
        "--backend",
        str(args.probe_backend or preset.get("probe_backend", "transformers")),
        "--selected-layer",
        str(args.selected_layer),
    ]
    if args.max_rows is not None:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.probe_local_model_name:
        cmd += ["--local-model-name", str(args.probe_local_model_name)]
    for item in args.probe_local_model_map or preset.get("probe_local_model_map", []):
        cmd += ["--local-model-map", str(item)]
    return cmd


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("$", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mainline batch entrypoint for large-scale re-entry experiments."
    )
    parser.add_argument("--preset", required=True, help="Preset name from reentry_presets.json")
    parser.add_argument("--preset-file", default=str(PRESET_PATH))
    parser.add_argument(
        "--phase",
        choices=["prepare", "reentry", "analyze", "probe", "both", "all"],
        default="both",
        help="prepare runs family sweep, reentry runs controls, analyze recomputes summaries, probe extracts transformer representations.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--small-backend", default=None, choices=["ollama", "local"])
    parser.add_argument("--small-local-model-name", default=None)
    parser.add_argument("--small-local-model-map", action="append", default=[])
    parser.add_argument("--probe-output-dir", default=None)
    parser.add_argument("--probe-backend", default=None, choices=["transformers"])
    parser.add_argument("--probe-local-model-name", default=None)
    parser.add_argument("--probe-local-model-map", action="append", default=[])
    parser.add_argument("--selected-layer", type=int, default=-1)
    parser.add_argument("--run-name-prefix", action="append", default=[])
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--small-family", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--full-rerun-temperature", type=float, default=0.0)
    parser.add_argument("--reentry-temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    presets = load_presets(Path(args.preset_file))
    if args.preset not in presets:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"Unknown preset {args.preset!r}. Known presets: {known}")
    preset = presets[args.preset]

    if args.phase in {"prepare", "both", "all"}:
        run_cmd(build_prepare_cmd(str(preset["family_sweep_config"])), dry_run=args.dry_run)

    if args.phase in {"reentry", "both", "all"}:
        run_cmd(build_reentry_cmd(args.preset, preset, args), dry_run=args.dry_run)

    if args.phase in {"analyze", "both", "all"}:
        run_cmd(build_analyze_cmd(args, preset), dry_run=args.dry_run)

    if args.phase in {"probe", "all"}:
        run_cmd(build_probe_cmd(args, preset), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
