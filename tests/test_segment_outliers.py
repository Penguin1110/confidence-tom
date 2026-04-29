import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mainline.run.core.run_prefix_oracle_gain_mapping import (
    annotate_segment_count_outliers,
)


def _row(task_id: str, segment_count: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "segments": [
            {"segment_id": f"seg_{idx}", "index": idx, "text": f"step {idx}"}
            for idx in range(1, segment_count + 1)
        ],
        "metadata": {},
    }


def test_annotate_segment_count_outliers_marks_long_tail(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            [
                _row("task_1", 3),
                _row("task_2", 4),
                _row("task_3", 5),
                _row("task_4", 6),
                _row("task_5", 200),
            ]
        ),
        encoding="utf-8",
    )

    annotate_segment_count_outliers(path)

    rows = json.loads(path.read_text(encoding="utf-8"))
    flags = [row["metadata"]["segment_count_outlier"] for row in rows]
    counts = [row["metadata"]["segment_count"] for row in rows]

    assert counts == [3, 4, 5, 6, 200]
    assert flags == [False, False, False, False, True]
    assert rows[-1]["metadata"]["segment_count_outlier_stats"]["method"] == "iqr_1.5"
