from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from confidence_tom.infra.paths import results_root

RESULTS_DIR = results_root()
OUTPUT_DIR = RESULTS_DIR / "_prefix_predictor_v1"
OUTPUT_CSV = OUTPUT_DIR / "prefix_predictor_rows.csv"
OUTPUT_META = OUTPUT_DIR / "dataset_meta.json"


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    benchmark: str
    small_family: str
    large_family: str


RUN_SPECS = [
    RunSpec("qwen_to_openai_50", "olympiadbench", "qwen", "openai"),
    RunSpec("qwen_to_anthropic_50", "olympiadbench", "qwen", "anthropic"),
    RunSpec("llama_to_openai_50", "olympiadbench", "llama", "openai"),
    RunSpec("llama_to_anthropic_50", "olympiadbench", "llama", "anthropic"),
    RunSpec("mistral_to_openai_50", "olympiadbench", "mistral", "openai"),
    RunSpec("mistral_to_anthropic_50", "olympiadbench", "mistral", "anthropic"),
    RunSpec("livebench_qwen_to_openai_30", "livebench_reasoning", "qwen", "openai"),
    RunSpec("livebench_qwen_to_anthropic_30", "livebench_reasoning", "qwen", "anthropic"),
    RunSpec("livebench_llama_to_openai_30", "livebench_reasoning", "llama", "openai"),
    RunSpec("livebench_llama_to_anthropic_30", "livebench_reasoning", "llama", "anthropic"),
    RunSpec("livebench_mistral_to_openai_30", "livebench_reasoning", "mistral", "openai"),
    RunSpec("livebench_mistral_to_anthropic_30", "livebench_reasoning", "mistral", "anthropic"),
]


# Only include prefix-observable signals. Do not leak rollout outcomes or
# continuation token counts into the predictor dataset.
NUMERIC_FEATURE_COLUMNS = [
    "step_index",
    "prefix_segments_count",
    "prefix_tokens",
    "current_segment_tokens",
    "semantic_drift_score",
    "hedge_density",
    "confidence_proxy",
]

TEXT_FEATURE_COLUMNS = [
    "prefix_text_tokens",
    "backtracking_flag",
    "backtracking_mentions",
    "self_correction_cue_density",
    "certainty_density",
    "commitment_score",
]

_BACKTRACK_PATTERNS = [
    "go back",
    "back to",
    "revisit",
    "earlier",
    "previous step",
    "instead",
]

_SELF_CORRECTION_PATTERNS = [
    "actually",
    "however",
    "but",
    "instead",
    "reconsider",
    "mistake",
    "correction",
    "on second thought",
]

_CERTAINTY_PATTERNS = [
    "therefore",
    "thus",
    "hence",
    "so ",
    "combining these",
    "we conclude",
    "this implies",
    "it follows that",
    "final answer",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _pattern_density(text: str, patterns: list[str]) -> tuple[int, float]:
    lowered = text.lower()
    hits = sum(lowered.count(pattern) for pattern in patterns)
    token_count = max(1, len(_tokenize(text)))
    return hits, hits / token_count


def _extract_text_features(prefix_text: str) -> dict[str, float]:
    token_count = float(len(_tokenize(prefix_text)))
    backtrack_hits, _ = _pattern_density(prefix_text, _BACKTRACK_PATTERNS)
    correction_hits, correction_density = _pattern_density(prefix_text, _SELF_CORRECTION_PATTERNS)
    certainty_hits, certainty_density = _pattern_density(prefix_text, _CERTAINTY_PATTERNS)
    return {
        "prefix_text_tokens": token_count,
        "backtracking_flag": float(int(backtrack_hits > 0)),
        "backtracking_mentions": float(backtrack_hits),
        "self_correction_cue_density": correction_density,
        "certainty_density": certainty_density,
        "commitment_score": certainty_density - correction_density,
    }


def _find_result_json(run_name: str) -> Path:
    run_dir = RESULTS_DIR / run_name
    candidates = [
        path
        for path in run_dir.glob("*.json")
        if path.name not in {"summary.json", "dataset_meta.json", "baseline_results.json"}
    ]
    json_candidates = [path for path in candidates if "per_prefix_rows" not in path.name]
    if not json_candidates:
        raise FileNotFoundError(f"Could not find main result JSON in {run_dir}")
    return json_candidates[0]


def _load_prefix_text_index(run_name: str) -> dict[str, str]:
    result_json = _find_result_json(run_name)
    data = json.loads(result_json.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for task in data:
        for step in task.get("prefix_oracle_steps", []):
            index[str(step["prefix_id"])] = str(step.get("prefix_text", ""))
    return index


def _to_float(value: str) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


def _to_int(value: str) -> int:
    if value == "" or value is None:
        return 0
    return int(float(value))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    per_run_counts: dict[str, int] = {}
    per_run_positive: dict[str, int] = {}

    for spec in RUN_SPECS:
        csv_path = RESULTS_DIR / spec.run_name / "per_prefix_rows.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing per-prefix CSV: {csv_path}")
        prefix_text_index = _load_prefix_text_index(spec.run_name)

        count = 0
        positive = 0
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                delta = _to_float(raw["delta_correctness"])
                positive_gain = _to_int(raw["positive_gain"])
                row: dict[str, object] = {
                    "run_name": spec.run_name,
                    "benchmark": spec.benchmark,
                    "small_family": spec.small_family,
                    "large_family": spec.large_family,
                    "task_id": raw["task_id"],
                    "prefix_id": raw["prefix_id"],
                    "delta_t": delta,
                    "delta_positive": int(delta > 0.0),
                    "delta_nonzero": int(abs(delta) > 0.0),
                    "delta_sign": "positive"
                    if delta > 0.0
                    else "negative"
                    if delta < 0.0
                    else "zero",
                    "positive_gain": positive_gain,
                }
                for key in NUMERIC_FEATURE_COLUMNS:
                    row[key] = _to_float(raw[key])
                prefix_text = prefix_text_index.get(str(raw["prefix_id"]), "")
                for key, value in _extract_text_features(prefix_text).items():
                    row[key] = value
                rows.append(row)
                count += 1
                positive += positive_gain

        per_run_counts[spec.run_name] = count
        per_run_positive[spec.run_name] = positive

    fieldnames = [
        "run_name",
        "benchmark",
        "small_family",
        "large_family",
        "task_id",
        "prefix_id",
        "delta_t",
        "delta_positive",
        "delta_nonzero",
        "delta_sign",
        "positive_gain",
        *NUMERIC_FEATURE_COLUMNS,
        *TEXT_FEATURE_COLUMNS,
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "rows": len(rows),
        "runs": len(RUN_SPECS),
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "text_feature_columns": TEXT_FEATURE_COLUMNS,
        "target": "delta_positive",
        "notes": [
            "Rows come from per_prefix_rows.csv across OlympiadBench and LiveBench family runs.",
            "Only prefix-observable features are kept in the training dataset.",
            (
                "Rollout outcomes and continuation token totals are intentionally "
                "excluded to avoid leakage."
            ),
            (
                "Confidence-related text features are extracted from prefix_text "
                "in the main result JSONs."
            ),
        ],
        "per_run_counts": per_run_counts,
        "per_run_positive": per_run_positive,
    }
    OUTPUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote dataset CSV to {OUTPUT_CSV}")
    print(f"Wrote dataset metadata to {OUTPUT_META}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
