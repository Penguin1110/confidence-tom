"""Registry for dynamic agent benchmarks used in the next experiment stage.

This module keeps installation and environment requirements in one place so the
repo can bootstrap external benchmarks without hard-coding benchmark-specific
logic across multiple scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InstallMode = Literal["pip", "editable_clone", "sparse_clone", "clone_only"]


@dataclass(frozen=True)
class DynamicBenchmarkSpec:
    key: str
    display_name: str
    install_mode: InstallMode
    pip_package: str | None = None
    git_url: str | None = None
    clone_dir: str | None = None
    sparse_subdir: str | None = None
    docker_required: bool = False
    recommended_python: str = "3.11+"
    notes: tuple[str, ...] = ()


DYNAMIC_BENCHMARKS: dict[str, DynamicBenchmarkSpec] = {
    "tau-bench": DynamicBenchmarkSpec(
        key="tau-bench",
        display_name="tau-bench",
        install_mode="editable_clone",
        git_url="https://github.com/sierra-research/tau-bench.git",
        clone_dir="tau-bench",
        recommended_python="3.10+",
        notes=(
            "Recommended install path is a local editable source checkout.",
            "Requires API keys for the model provider and user simulator you select.",
        ),
    ),
    "plancraft": DynamicBenchmarkSpec(
        key="plancraft",
        display_name="Plancraft",
        install_mode="pip",
        pip_package="plancraft",
        recommended_python="3.10+",
        notes=(
            "Available on PyPI.",
            "Use the PlancraftGymWrapper for the simplest agent integration path.",
        ),
    ),
    "birdsql": DynamicBenchmarkSpec(
        key="birdsql",
        display_name="BIRD-SQL",
        install_mode="sparse_clone",
        git_url="https://github.com/AlibabaResearch/DAMO-ConvAI.git",
        clone_dir="birdsql",
        sparse_subdir="bird",
        recommended_python="3.10+",
        notes=(
            (
                "BIRD-SQL is distributed inside the DAMO-ConvAI repository "
                "rather than as a PyPI package."
            ),
            (
                "The full benchmark data can be large; keep it in an external "
                "directory instead of vendoring it into this repo."
            ),
        ),
    ),
    "intercode": DynamicBenchmarkSpec(
        key="intercode",
        display_name="InterCode",
        install_mode="pip",
        pip_package="intercode-bench",
        docker_required=True,
        recommended_python="3.8+",
        notes=(
            "PyPI package exists, but the environments still require a running Docker daemon.",
            "The SQL and Bash environments are the lowest-friction starting point.",
        ),
    ),
}


def get_dynamic_benchmark(key: str) -> DynamicBenchmarkSpec:
    try:
        return DYNAMIC_BENCHMARKS[key]
    except KeyError as exc:
        known = ", ".join(sorted(DYNAMIC_BENCHMARKS))
        raise KeyError(f"Unknown benchmark '{key}'. Known benchmarks: {known}") from exc


def list_dynamic_benchmarks() -> list[DynamicBenchmarkSpec]:
    return [DYNAMIC_BENCHMARKS[key] for key in sorted(DYNAMIC_BENCHMARKS)]
