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
OUT_DIR = ROOT / "results" / "_hidden_state_late_success_vs_failure_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "hidden_state_late_success_vs_failure.md"
)

EARLY_FRACTION = 1.0 / 3.0
MIN_STABLE_LOCAL_RATE = 0.5


def _stable_test_split(task_id: str, test_ratio: float = 0.2) -> bool:
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


def _family_probe_paths() -> dict[str, tuple[Path, Path]]:
    mapping: dict[str, tuple[Path, Path]] = {}
    for probe_rows in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family_dir = probe_rows.parent.name
        reentry_rows = REENTRY_DIR / family_dir / "reentry_rows.jsonl"
        if reentry_rows.exists():
            mapping[family_dir] = (probe_rows, reentry_rows)
    return mapping


def _task_taxonomy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda r: int(r["step_index"]))
        seq = [int(row.get("reentry_exact_correct", 0) or 0) for row in task_rows]
        step_count = len(seq)
        any_small_correct = any(seq)
        first_correct_step = next((i + 1 for i, flag in enumerate(seq) if flag), None)
        first_correct_frac = (first_correct_step / step_count) if first_correct_step else 1.0
        local_correct_rate = (sum(seq) / step_count) if step_count else 0.0
        last_small_correct = bool(seq[-1]) if seq else False
        full_correct = bool(task_rows[0].get("full_trace_correct", 0))
        if full_correct:
            if (
                any_small_correct
                and first_correct_frac <= EARLY_FRACTION
                and local_correct_rate >= MIN_STABLE_LOCAL_RATE
            ):
                category = "stable-success"
            else:
                category = "late-success"
        else:
            category = "late-failure" if any_small_correct else "persistent-failure"
        out[task_id] = {
            "category": category,
            "full_trace_correct": int(full_correct),
            "any_small_correct": int(any_small_correct),
            "first_correct_frac": float(first_correct_frac),
            "local_correct_rate": float(local_correct_rate),
            "last_small_correct": int(last_small_correct),
        }
    return out


def _merged_rows(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}
    taxonomy = _task_taxonomy(reentry_rows)
    merged: list[dict[str, Any]] = []
    for row in reentry_rows:
        task_meta = taxonomy[str(row["task_id"])]
        if task_meta["category"] not in {"late-success", "late-failure"}:
            continue
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        merged.append(
            {
                **row,
                **task_meta,
                "mean_hidden": probe["mean_pool_hidden"],
                "last_token_hidden": probe["last_token_hidden"],
                "hidden_dim": int(probe["hidden_dim"]),
                "selected_layer": int(probe["selected_layer"]),
            }
        )
    return merged


def _probe_binary(rows: list[dict[str, Any]], vector_key: str) -> dict[str, Any]:
    if not rows:
        return {"error": "no_rows"}
    train_rows = [row for row in rows if not _stable_test_split(str(row["task_id"]))]
    test_rows = [row for row in rows if _stable_test_split(str(row["task_id"]))]
    if not train_rows or not test_rows:
        return {
            "error": "insufficient_split_rows",
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_tasks": len({str(row["task_id"]) for row in train_rows}),
            "test_tasks": len({str(row["task_id"]) for row in test_rows}),
        }
    x_train = np.asarray([row[vector_key] for row in train_rows], dtype=np.float64)
    x_test = np.asarray([row[vector_key] for row in test_rows], dtype=np.float64)
    y_train = np.asarray(
        [1.0 if row["category"] == "late-success" else 0.0 for row in train_rows], dtype=np.float64
    )
    y_test = np.asarray(
        [1.0 if row["category"] == "late-success" else 0.0 for row in test_rows], dtype=np.float64
    )
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
        "train_base_rate_late_success": float(np.mean(y_train)),
        "test_base_rate_late_success": float(np.mean(y_test)),
        "threshold": threshold,
        "train_f1": train_f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auroc": _roc_auc_score(y_test, test_probs),
    }


def _step_bin(step_index: int) -> str:
    if step_index <= 1:
        return "step1"
    if step_index == 2:
        return "step2"
    if step_index == 3:
        return "step3"
    return "step4plus"


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": len(rows),
        "tasks": len({str(row["task_id"]) for row in rows}),
        "selected_layer": int(rows[0]["selected_layer"]),
        "hidden_dim": int(rows[0]["hidden_dim"]),
        "category_counts": {
            "late-success": sum(1 for row in rows if row["category"] == "late-success"),
            "late-failure": sum(1 for row in rows if row["category"] == "late-failure"),
        },
        "summary_stats": {},
        "results": {},
    }
    for cat in ["late-success", "late-failure"]:
        block = [row for row in rows if row["category"] == cat]
        out["summary_stats"][cat] = {
            "rows": len(block),
            "tasks": len({str(row["task_id"]) for row in block}),
            "mean_first_correct_frac": float(np.mean([row["first_correct_frac"] for row in block])),
            "mean_local_correct_rate": float(np.mean([row["local_correct_rate"] for row in block])),
            "mean_last_small_correct": float(np.mean([row["last_small_correct"] for row in block])),
        }
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        report = {"overall": _probe_binary(rows, vector_key), "by_step_bin": {}}
        for bin_name in ["step1", "step2", "step3", "step4plus"]:
            block = [row for row in rows if _step_bin(int(row["step_index"])) == bin_name]
            if block:
                report["by_step_bin"][bin_name] = _probe_binary(block, vector_key)
        out["results"][vector_key] = report
    return out


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        blocks = [
            cast(dict[str, Any], fam["results"][vector_key]["overall"])
            for fam in families.values()
            if "test_auroc" in cast(dict[str, Any], fam["results"][vector_key]["overall"])
        ]
        if not blocks:
            out[vector_key] = {
                "macro_auroc": 0.0,
                "weighted_auroc": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
            }
            continue
        weight = sum(int(block["test_rows"]) for block in blocks)
        out[vector_key] = {
            "macro_auroc": float(np.mean([float(block["test_auroc"]) for block in blocks])),
            "weighted_auroc": (
                float(
                    sum(float(block["test_auroc"]) * int(block["test_rows"]) for block in blocks)
                    / weight
                )
                if weight
                else 0.0
            ),
            "macro_f1": float(np.mean([float(block["test_f1"]) for block in blocks])),
            "weighted_f1": (
                float(
                    sum(float(block["test_f1"]) * int(block["test_rows"]) for block in blocks)
                    / weight
                )
                if weight
                else 0.0
            ),
        }
    return out


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, dict[str, Any]], summary["families"])
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    lines = [
        "# Hidden State: Late-Success Vs Late-Failure",
        "",
        "## Question",
        "",
        "- Restrict to tasks that show some reusable local correctness signal.",
        "- Ask whether hidden state can separate `late-success` from `late-failure`.",
        "",
        "## Aggregate",
        "",
        "| Pooling | Macro AUROC | Weighted AUROC | Macro F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for vector_key in ["mean_hidden", "last_token_hidden"]:
        report = aggregate[vector_key]
        lines.append(
            f"| {vector_key} | {report['macro_auroc']:.3f} | {report['weighted_auroc']:.3f} | "
            f"{report['macro_f1']:.3f} | {report['weighted_f1']:.3f} |"
        )
    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        lines += [
            f"### `{family}`",
            "",
            f"- rows: `{report['rows']}`",
            f"- tasks: `{report['tasks']}`",
            f"- selected layer: `{report['selected_layer']}`",
            f"- hidden dim: `{report['hidden_dim']}`",
            "",
            "| Category | Rows | Tasks | Mean first-correct frac | Mean local correct rate | Mean last-small-correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cat in ["late-success", "late-failure"]:
            block = report["summary_stats"][cat]
            lines.append(
                f"| {cat} | {block['rows']} | {block['tasks']} | {block['mean_first_correct_frac']:.3f} | "
                f"{block['mean_local_correct_rate']:.3f} | {block['mean_last_small_correct']:.3f} |"
            )
        lines += [
            "",
            "| Pooling | AUROC | F1 | Precision | Recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            metrics = report["results"][vector_key]["overall"]
            if "test_auroc" in metrics:
                lines.append(
                    f"| {vector_key} | {metrics['test_auroc']:.3f} | {metrics['test_f1']:.3f} | "
                    f"{metrics['test_precision']:.3f} | {metrics['test_recall']:.3f} |"
                )
            else:
                lines.append(f"| {vector_key} | n/a | n/a | n/a | n/a |")
        lines += ["", "| Step Bin | Pooling | AUROC | F1 |", "| --- | --- | ---: | ---: |"]
        for vector_key in ["mean_hidden", "last_token_hidden"]:
            for step_bin, metrics in report["results"][vector_key]["by_step_bin"].items():
                if "test_auroc" in metrics:
                    lines.append(
                        f"| {step_bin} | {vector_key} | {metrics['test_auroc']:.3f} | {metrics['test_f1']:.3f} |"
                    )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    families: dict[str, dict[str, Any]] = {}
    for family_dir, (probe_path, reentry_path) in _family_probe_paths().items():
        rows = _merged_rows(probe_path, reentry_path)
        if rows:
            families[family_dir.replace("_reentry_livebench_local_v1_", "")] = _family_analysis(
                rows
            )
    summary = {
        "source_dir": str(DATA_DIR),
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
