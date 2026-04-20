from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "_prefix_reentry_controls_v1"
DEFAULT_ROWS = DEFAULT_DIR / "reentry_rows.jsonl"
DEFAULT_SUMMARY = DEFAULT_DIR / "reentry_summary.json"
DEFAULT_MD = ROOT / "docs" / "prefix_reentry_controls.md"


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            rows.append(row)
    return rows


def _rate(rows: list[dict[str, object]], field: str) -> float | None:
    if not rows:
        return None
    return sum(int(row[field]) for row in rows) / len(rows)


def _conditional_rate(
    rows: list[dict[str, object]], *, cond_field: str, cond_value: int, target_field: str
) -> float | None:
    block = [row for row in rows if int(row[cond_field]) == cond_value]
    if not block:
        return None
    return sum(int(row[target_field]) for row in block) / len(block)


def _group(rows: list[dict[str, object]], field: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = str(row[field])
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
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
        bucket: dict[str, object] = {}
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


def to_markdown(summary: dict[str, object]) -> str:
    if summary["rows"] == 0:
        return "# Prefix Re-entry Controls\n\nNo completed rows.\n"

    def fmt(value: object) -> str:
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
        f"- P(full-trace success | re-entry match): `{fmt(summary['full_trace_success_given_reentry_match'])}`",
        f"- P(full-trace success | re-entry mismatch): `{fmt(summary['full_trace_success_given_reentry_mismatch'])}`",
        f"- P(positive takeover | re-entry match): `{fmt(summary['positive_takeover_given_reentry_match'])}`",
        f"- P(positive takeover | re-entry mismatch): `{fmt(summary['positive_takeover_given_reentry_mismatch'])}`",
        "",
        "## By Benchmark",
        "",
    ]
    for benchmark, block in summary["by_benchmark"].items():
        lines.append(
            f"- `{benchmark}`: rows={block['rows']}, "
            f"reentry_match={fmt(block['reentry_match_rate'])}, "
            f"full_rerun_match={fmt(block['full_rerun_match_rate'])}, "
            f"p_pos|match={fmt(block['positive_takeover_given_reentry_match'])}, "
            f"p_pos|mismatch={fmt(block['positive_takeover_given_reentry_mismatch'])}"
        )
    lines += ["", "## By Small Family", ""]
    for family, block in summary["by_small_family"].items():
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
