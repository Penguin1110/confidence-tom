from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
PROBE_DIR = DATA_DIR / "probe"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_hidden_state_onset_persistence_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "hidden_state_onset_persistence.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


def _stable_test_split(task_id: str, test_ratio: float = 0.25) -> bool:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    steps: int = 1200,
    l2: float = 1e-3,
) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        probs = _sigmoid(x @ weights)
        grad = (x.T @ (probs - y)) / len(y)
        grad += l2 * weights
        weights -= lr * grad
    return weights


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_positive = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    candidates = np.unique(np.concatenate([np.array([0.05, 0.1, 0.2, 0.3, 0.5]), probs]))
    for threshold in candidates:
        preds = (probs >= threshold).astype(np.int64)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _zscore(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0)
    stds[stds == 0.0] = 1.0
    return (train_x - means) / stds, (test_x - means) / stds


def _project_pca(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    max_components: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if train_x.shape[1] <= max_components:
        return train_x, test_x
    train_center = np.mean(train_x, axis=0, keepdims=True)
    centered_train = train_x - train_center
    centered_test = test_x - train_center
    _u, _s, vh = np.linalg.svd(centered_train, full_matrices=False)
    max_rank = max(1, min(centered_train.shape[0] - 1, centered_train.shape[1], max_components))
    basis = vh[:max_rank].T
    return centered_train @ basis, centered_test @ basis


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_category(full_correct: bool, seq: list[int]) -> str:
    step_count = len(seq)
    any_small_correct = any(seq)
    first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
    first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
    local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
    if full_correct:
        if (
            any_small_correct
            and first_correct_frac <= EARLY_FRACTION
            and local_correct_rate >= MIN_STABLE_LOCAL_RATE
        ):
            return "stable-success"
        return "late-success"
    return "late-failure" if any_small_correct else "persistent-failure"


def _merge_family(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reentry_rows:
        by_task[str(row["task_id"])].append(row)
    out: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda row: int(row["step_index"]))
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        if not any(seq):
            continue
        full_correct = bool(task_rows[0].get("full_trace_correct", 0))
        category = _task_category(full_correct, seq)
        first_idx = seq.index(1)
        onset_row = task_rows[first_idx]
        probe = probe_by_prefix.get(str(onset_row["prefix_id"]))
        if probe is None:
            continue
        tail = seq[first_idx:]
        collapse_after_onset = int(
            any(tail[i] == 1 and tail[i + 1] == 0 for i in range(len(tail) - 1))
        )
        out.append(
            {
                "task_id": task_id,
                "category": category,
                "full_trace_correct": int(full_correct),
                "last_small_correct": int(seq[-1]),
                "tail_correct_rate": float(np.mean(tail)),
                "collapse_after_onset": collapse_after_onset,
                "onset_step_index": int(onset_row["step_index"]),
                "onset_frac": float((first_idx + 1) / len(seq)),
                "steps": len(seq),
                "mean_hidden": probe["mean_pool_hidden"],
                "last_token_hidden": probe["last_token_hidden"],
                "hidden_dim": int(probe["hidden_dim"]),
                "selected_layer": int(probe["selected_layer"]),
            }
        )
    return out


def _probe_binary(rows: list[dict[str, Any]], vector_key: str, label_key: str) -> dict[str, Any]:
    if not rows:
        return {"error": "no_rows"}
    train_rows = [row for row in rows if not _stable_test_split(str(row["task_id"]))]
    test_rows = [row for row in rows if _stable_test_split(str(row["task_id"]))]
    if not train_rows or not test_rows:
        return {"error": "insufficient_split_rows"}
    x_train = np.asarray([row[vector_key] for row in train_rows], dtype=np.float64)
    x_test = np.asarray([row[vector_key] for row in test_rows], dtype=np.float64)
    y_train = np.asarray([float(row[label_key]) for row in train_rows], dtype=np.float64)
    y_test = np.asarray([float(row[label_key]) for row in test_rows], dtype=np.float64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "error": "single_class_split",
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_base_rate": float(np.mean(y_train)),
            "test_base_rate": float(np.mean(y_test)),
        }
    x_train, x_test = _project_pca(x_train, x_test)
    x_train, x_test = _zscore(x_train, x_test)
    x_train = np.concatenate([np.ones((len(x_train), 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((len(x_test), 1)), x_test], axis=1)
    weights = _fit_logistic_regression(x_train, y_train)
    train_probs = _sigmoid(x_train @ weights)
    test_probs = _sigmoid(x_test @ weights)
    threshold, train_f1 = _best_f1_threshold(y_train, train_probs)
    preds = (test_probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_tasks": len({str(row["task_id"]) for row in train_rows}),
        "test_tasks": len({str(row["task_id"]) for row in test_rows}),
        "train_base_rate": float(np.mean(y_train)),
        "test_base_rate": float(np.mean(y_test)),
        "threshold": threshold,
        "train_f1": train_f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auroc": _roc_auc_score(y_test, test_probs),
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "tasks": len({str(row["task_id"]) for row in rows}),
        "selected_layer": int(rows[0]["selected_layer"]),
        "hidden_dim": int(rows[0]["hidden_dim"]),
        "category_counts": dict(
            defaultdict(
                int,
                {
                    str(k): sum(1 for row in rows if row["category"] == k)
                    for k in {row["category"] for row in rows}
                },
            )
        ),
        "mean_onset_frac": float(np.mean([row["onset_frac"] for row in rows])),
        "mean_tail_correct_rate": float(np.mean([row["tail_correct_rate"] for row in rows])),
        "probes": {
            vector_key: {
                label_key: _probe_binary(rows, vector_key, label_key)
                for label_key in [
                    "full_trace_correct",
                    "last_small_correct",
                    "collapse_after_onset",
                ]
            }
            for vector_key in ["mean_hidden", "last_token_hidden"]
        },
    }


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        out[vector_key] = {}
        for label_key in ["full_trace_correct", "last_small_correct", "collapse_after_onset"]:
            scores = [
                family["probes"][vector_key][label_key]["test_auroc"]
                for family in families.values()
                if "test_auroc" in family["probes"][vector_key][label_key]
            ]
            out[vector_key][label_key] = {
                "mean_auroc": float(np.mean(scores)) if scores else 0.5,
                "family_count": len(scores),
            }
    return out


def _to_markdown(summary: dict[str, Any]) -> str:
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    families = cast(dict[str, dict[str, Any]], summary["families"])
    lines = [
        "# Hidden State At First-Correct Onset",
        "",
        "## Question",
        "",
        "- Look only at tasks that ever become locally correct.",
        "- Take the hidden state at the first correct prefix.",
        "- Ask whether that onset state predicts preservation to the end.",
        "",
        "## Aggregate AUROC",
        "",
        "| Pooling | Predict full-trace correct | Predict last-small-correct | Predict collapse-after-onset |",
        "| --- | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        lines.append(
            f"| {vector_key} | {aggregate[vector_key]['full_trace_correct']['mean_auroc']:.3f} | "
            f"{aggregate[vector_key]['last_small_correct']['mean_auroc']:.3f} | "
            f"{aggregate[vector_key]['collapse_after_onset']['mean_auroc']:.3f} |"
        )
    lines += [
        "",
        "## Family Breakdown",
        "",
    ]
    for family, block in sorted(families.items()):
        lines.append(f"### `{family}`")
        lines.append("")
        lines.append(f"- tasks with any local correctness: `{block['tasks']}`")
        lines.append(f"- mean onset frac: `{block['mean_onset_frac']:.3f}`")
        lines.append(f"- mean tail correct rate: `{block['mean_tail_correct_rate']:.3f}`")
        lines.append(
            f"- category counts: `{json.dumps(block['category_counts'], ensure_ascii=False)}`"
        )
        lines.append("")
        lines.append(
            "| Pooling | Full-trace correct AUROC | Last-small-correct AUROC | Collapse-after-onset AUROC |"
        )
        lines.append("| --- | ---: | ---: | ---: |")
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            a = block["probes"][vector_key]["full_trace_correct"]
            b = block["probes"][vector_key]["last_small_correct"]
            c = block["probes"][vector_key]["collapse_after_onset"]

            def fmt(x: dict[str, Any]) -> str:
                return f"{x['test_auroc']:.3f}" if "test_auroc" in x else "n/a"

            lines.append(f"| {vector_key} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
        lines.append("")
    lines += [
        "## Main Read",
        "",
        "- This analysis isolates the onset state: not whether correctness ever appears, but whether the first correct state is already a preservable state.",
        "- If onset-state prediction works, it supports a `state persistence bottleneck` interpretation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, dict[str, Any]] = {}
    for probe_path in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family = probe_path.parent.name.replace("_reentry_livebench_local_v1_", "")
        reentry_path = REENTRY_DIR / probe_path.parent.name / "reentry_rows.jsonl"
        if not reentry_path.exists():
            continue
        rows = _merge_family(probe_path, reentry_path)
        if not rows:
            continue
        families[family] = _family_analysis(rows)
    summary = {
        "source_dir": str(DATA_DIR),
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
