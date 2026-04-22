from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "outputs" / "results" / "_prefix_reentry_controls_v1"
DEFAULT_ROWS = DEFAULT_DIR / "reentry_rows.jsonl"
DEFAULT_SUMMARY = DEFAULT_DIR / "reentry_summary.json"
DEFAULT_MD = (
    ROOT / "docs" / "mainline" / "generated" / "analysis" / "prefix" / "prefix_reentry_controls.md"
)

Row = dict[str, Any]
BucketSummary = dict[str, float | int | None]


def _load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            rows.append(row)
    return rows


def _rate(rows: list[Row], field: str) -> float | None:
    if not rows:
        return None
    return sum(int(row[field]) for row in rows) / len(rows)


def _conditional_rate(
    rows: list[Row], *, cond_field: str, cond_value: int, target_field: str
) -> float | None:
    block = [row for row in rows if int(row[cond_field]) == cond_value]
    if not block:
        return None
    return sum(int(row[target_field]) for row in block) / len(block)


def _group(rows: list[Row], field: str) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        key = str(row[field])
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize(rows: list[Row]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(rows),
        "full_rerun_match_rate": _rate(rows, "full_rerun_matches_original_full"),
        "reentry_match_rate": _rate(rows, "reentry_exact_matches_original_small"),
        "reentry_repeat_match_rate": _rate(rows, "reentry_repeat_matches_first"),
        "marker_boundary_match_rate": _rate(rows, "reentry_marker_matches_exact"),
        "fenced_boundary_match_rate": _rate(rows, "reentry_fenced_matches_exact"),
        "full_trace_success_given_reentry_match": _conditional_rate(
            rows,
            cond_field="reentry_exact_matches_original_small",
            cond_value=1,
            target_field="full_trace_correct",
        ),
        "full_trace_success_given_reentry_mismatch": _conditional_rate(
            rows,
            cond_field="reentry_exact_matches_original_small",
            cond_value=0,
            target_field="full_trace_correct",
        ),
        "positive_takeover_given_reentry_match": _conditional_rate(
            rows,
            cond_field="reentry_exact_matches_original_small",
            cond_value=1,
            target_field="positive_gain",
        ),
        "positive_takeover_given_reentry_mismatch": _conditional_rate(
            rows,
            cond_field="reentry_exact_matches_original_small",
            cond_value=0,
            target_field="positive_gain",
        ),
        "by_benchmark": {},
        "by_small_family": {},
    }
    for field, bucket_name in [("benchmark", "by_benchmark"), ("small_family", "by_small_family")]:
        bucket: dict[str, BucketSummary] = {}
        for key, block in _group(rows, field).items():
            bucket[key] = {
                "rows": len(block),
                "reentry_match_rate": _rate(block, "reentry_exact_matches_original_small"),
                "full_rerun_match_rate": _rate(block, "full_rerun_matches_original_full"),
                "positive_takeover_given_reentry_match": _conditional_rate(
                    block,
                    cond_field="reentry_exact_matches_original_small",
                    cond_value=1,
                    target_field="positive_gain",
                ),
                "positive_takeover_given_reentry_mismatch": _conditional_rate(
                    block,
                    cond_field="reentry_exact_matches_original_small",
                    cond_value=0,
                    target_field="positive_gain",
                ),
            }
        summary[bucket_name] = bucket
    return summary


def to_markdown(summary: dict[str, Any]) -> str:
    if summary["rows"] == 0:
        return "# Prefix Re-entry Controls\n\nNo completed rows.\n"

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"

    lines = [
        "# Prefix Re-entry Controls",
        "",
        f"- rows: `{summary['rows']}`",
        f"- full rerun match rate: `{fmt(summary['full_rerun_match_rate'])}`",
        f"- re-entry match rate: `{fmt(summary['reentry_match_rate'])}`",
        f"- re-entry repeat match rate: `{fmt(summary['reentry_repeat_match_rate'])}`",
        f"- marker boundary match rate: `{fmt(summary['marker_boundary_match_rate'])}`",
        f"- fenced boundary match rate: `{fmt(summary['fenced_boundary_match_rate'])}`",
        (
            "- P(full-trace success | re-entry match): "
            f"`{fmt(summary['full_trace_success_given_reentry_match'])}`"
        ),
        (
            "- P(full-trace success | re-entry mismatch): "
            f"`{fmt(summary['full_trace_success_given_reentry_mismatch'])}`"
        ),
        (
            "- P(positive takeover | re-entry match): "
            f"`{fmt(summary['positive_takeover_given_reentry_match'])}`"
        ),
        (
            "- P(positive takeover | re-entry mismatch): "
            f"`{fmt(summary['positive_takeover_given_reentry_mismatch'])}`"
        ),
        "",
        "## By Benchmark",
        "",
    ]
    for benchmark, block in cast(dict[str, BucketSummary], summary["by_benchmark"]).items():
        lines.append(
            f"- `{benchmark}`: rows={block['rows']}, "
            f"reentry_match={fmt(block['reentry_match_rate'])}, "
            f"full_rerun_match={fmt(block['full_rerun_match_rate'])}, "
            f"p_pos|match={fmt(block['positive_takeover_given_reentry_match'])}, "
            f"p_pos|mismatch={fmt(block['positive_takeover_given_reentry_mismatch'])}"
        )
    lines += ["", "## By Small Family", ""]
    for family, block in cast(dict[str, BucketSummary], summary["by_small_family"]).items():
        lines.append(
            f"- `{family}`: rows={block['rows']}, "
            f"reentry_match={fmt(block['reentry_match_rate'])}, "
            f"full_rerun_match={fmt(block['full_rerun_match_rate'])}, "
            f"p_pos|match={fmt(block['positive_takeover_given_reentry_match'])}, "
            f"p_pos|mismatch={fmt(block['positive_takeover_given_reentry_mismatch'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prefix re-entry control rows.")
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--summary-md", default=str(DEFAULT_MD))
    args = parser.parse_args()

    rows = _load_rows(Path(args.rows))
    summary = summarize(rows)
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.summary_md).write_text(to_markdown(summary), encoding="utf-8")
    print(f"Wrote summary JSON to {args.summary_json}")
    print(f"Wrote summary MD to {args.summary_md}")


if __name__ == "__main__":
    main()
