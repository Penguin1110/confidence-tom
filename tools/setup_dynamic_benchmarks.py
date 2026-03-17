"""Bootstrap external dynamic agent benchmarks.

This script intentionally separates light-weight installs that can live in the
current project environment from heavy benchmarks that should stay in external
checkouts or their own Python environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_registry_functions() -> tuple[Any, Any]:
    module = importlib.import_module("confidence_tom.dynamic_benchmarks")
    return module.get_dynamic_benchmark, module.list_dynamic_benchmarks


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _pip_install(package_spec: str) -> None:
    _run([sys.executable, "-m", "pip", "install", package_spec])


def _patch_plancraft_windows_bug() -> tuple[bool, str]:
    """Patch Plancraft tag-file parsing on Windows.

    The current upstream package derives tag names with split("/") which breaks
    on Windows paths. That leads to KeyError such as "acacia_logs" at import
    time. We patch the installed file in place to use os.path.basename.
    """
    if os.name != "nt":
        return True, "not_windows_no_patch_needed"

    recipes_path = Path(sysconfig.get_paths()["purelib"]) / "plancraft" / "environment" / "recipes.py"
    if not recipes_path.exists():
        return False, f"plancraft_recipes_not_found:{recipes_path}"

    text = recipes_path.read_text(encoding="utf-8")
    old = 'tag_name = tag_file.split("/")[-1].split(".")[0]'
    new = "tag_name = os.path.splitext(os.path.basename(tag_file))[0]"

    if new in text:
        return True, "already_patched"
    if old not in text:
        return False, "expected_pattern_not_found"

    recipes_path.write_text(text.replace(old, new), encoding="utf-8")
    return True, f"patched:{recipes_path}"


def _editable_install(repo_dir: Path) -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])


def _clone_repo(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        print(f"[skip] clone exists: {target_dir}")
        return
    _run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def _sparse_clone_repo(repo_url: str, target_dir: Path, sparse_subdir: str) -> None:
    if target_dir.exists():
        print(f"[skip] sparse clone exists: {target_dir}")
        return
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(target_dir)])
    _run(["git", "sparse-checkout", "set", sparse_subdir], cwd=target_dir)


def _python_version() -> str:
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def _status_record(benchmark_key: str, success: bool, message: str, **extra: Any) -> dict[str, Any]:
    record = {
        "benchmark": benchmark_key,
        "success": success,
        "message": message,
    }
    record.update(extra)
    return record


def main() -> None:
    get_dynamic_benchmark, list_dynamic_benchmarks = _load_registry_functions()

    parser = argparse.ArgumentParser(description="Setup dynamic agent benchmarks")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            spec.key
            for spec in list_dynamic_benchmarks()
            if spec.key != "agentbench-os"
        ],
        help="Benchmarks to setup",
    )
    parser.add_argument(
        "--include-agentbench",
        action="store_true",
        help="Include agentbench-os in default benchmark list",
    )
    parser.add_argument(
        "--external-dir",
        default="external",
        help="Directory to store cloned external benchmarks",
    )
    parser.add_argument(
        "--clone-only",
        action="store_true",
        help="Clone repositories but skip any pip installation steps",
    )
    parser.add_argument(
        "--write-plan-only",
        action="store_true",
        help="Only write the setup plan JSON without making changes",
    )
    args = parser.parse_args()

    if args.include_agentbench and "agentbench-os" not in args.benchmarks:
        args.benchmarks.append("agentbench-os")

    external_dir = ROOT / args.external_dir
    external_dir.mkdir(parents=True, exist_ok=True)

    plan: list[dict[str, Any]] = []
    setup_status: list[dict[str, Any]] = []

    for benchmark_key in args.benchmarks:
        spec = get_dynamic_benchmark(benchmark_key)
        plan.append(asdict(spec))

        if args.write_plan_only:
            setup_status.append(
                _status_record(
                    benchmark_key,
                    True,
                    "plan_only",
                    install_mode=spec.install_mode,
                    recommended_python=spec.recommended_python,
                )
            )
            continue

        try:
            if spec.install_mode == "pip":
                if args.clone_only:
                    setup_status.append(
                        _status_record(
                            benchmark_key,
                            True,
                            "skipped pip install because --clone-only was set",
                            install_mode=spec.install_mode,
                        )
                    )
                else:
                    assert spec.pip_package is not None
                    _pip_install(spec.pip_package)

                    if benchmark_key == "plancraft":
                        patched, patch_msg = _patch_plancraft_windows_bug()
                    else:
                        patched, patch_msg = True, "not_applicable"

                    setup_status.append(
                        _status_record(
                            benchmark_key,
                            bool(patched),
                            f"installed pip package {spec.pip_package}",
                            install_mode=spec.install_mode,
                            post_install_patch=patch_msg,
                        )
                    )
            elif spec.install_mode == "editable_clone":
                assert spec.git_url is not None
                assert spec.clone_dir is not None
                repo_dir = external_dir / spec.clone_dir
                _clone_repo(spec.git_url, repo_dir)
                if args.clone_only:
                    setup_status.append(
                        _status_record(
                            benchmark_key,
                            True,
                            f"cloned into {repo_dir}",
                            install_mode=spec.install_mode,
                        )
                    )
                else:
                    _editable_install(repo_dir)
                    setup_status.append(
                        _status_record(
                            benchmark_key,
                            True,
                            f"cloned and installed editable package from {repo_dir}",
                            install_mode=spec.install_mode,
                        )
                    )
            elif spec.install_mode == "sparse_clone":
                assert spec.git_url is not None
                assert spec.clone_dir is not None
                assert spec.sparse_subdir is not None
                repo_dir = external_dir / spec.clone_dir
                _sparse_clone_repo(spec.git_url, repo_dir, spec.sparse_subdir)
                setup_status.append(
                    _status_record(
                        benchmark_key,
                        True,
                        f"sparse-cloned {spec.sparse_subdir} into {repo_dir}",
                        install_mode=spec.install_mode,
                    )
                )
            elif spec.install_mode == "clone_only":
                assert spec.git_url is not None
                assert spec.clone_dir is not None
                repo_dir = external_dir / spec.clone_dir
                _clone_repo(spec.git_url, repo_dir)
                setup_status.append(
                    _status_record(
                        benchmark_key,
                        True,
                        f"cloned into {repo_dir}",
                        install_mode=spec.install_mode,
                    )
                )
            else:
                raise ValueError(f"Unsupported install mode: {spec.install_mode}")
        except Exception as exc:  # noqa: BLE001
            setup_status.append(
                _status_record(
                    benchmark_key,
                    False,
                    str(exc),
                    install_mode=spec.install_mode,
                )
            )

    output = {
        "python": _python_version(),
        "external_dir": str(external_dir),
        "benchmarks": plan,
        "status": setup_status,
    }
    status_file = external_dir / "benchmark_setup_status.json"
    status_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[done] wrote setup summary to {status_file}")


if __name__ == "__main__":
    main()