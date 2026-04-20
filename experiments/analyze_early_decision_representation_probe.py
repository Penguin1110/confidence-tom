from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "_early_decision_representation_v1"
ROWS_PATH = DATA_DIR / "rows.jsonl"
REP_PATH = DATA_DIR / "representations.npz"
META_PATH = DATA_DIR / "meta.json"
OUT_JSON = DATA_DIR / "probe_results.json"
OUT_MD = ROOT / "docs" / "early_decision_representation_probe.md"


def _stable_test_split(task_id: str, test_ratio: float = 0.2) -> bool:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, *, lr: float = 0.05, steps: int = 3000, l2: float = 1e-3
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
    for threshold in np.unique(np.concatenate([np.array([0.05, 0.1, 0.2, 0.3, 0.5]), probs])):
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


def _load() -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    rows = [
        json.loads(line)
        for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reps = np.load(REP_PATH)["representations"]
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return rows, reps, meta


def _split(rows: list[dict[str, object]], reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = [i for i, row in enumerate(rows) if not _stable_test_split(str(row["task_id"]))]
    test_idx = [i for i, row in enumerate(rows) if _stable_test_split(str(row["task_id"]))]
    return np.asarray(train_idx), np.asarray(test_idx)


def _zscore(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0)
    stds[stds == 0.0] = 1.0
    return (train_x - means) / stds, (test_x - means) / stds


def _probe_binary(rows: list[dict[str, object]], reps: np.ndarray, target_fn) -> dict[str, object]:
    train_idx, test_idx = _split(rows, reps)
    x_train, x_test = reps[train_idx], reps[test_idx]
    y_train = np.asarray([float(target_fn(rows[i])) for i in train_idx], dtype=np.float64)
    y_test = np.asarray([float(target_fn(rows[i])) for i in test_idx], dtype=np.float64)
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
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_base_rate": float(np.mean(y_train)),
        "test_base_rate": float(np.mean(y_test)),
        "threshold": threshold,
        "train_f1": train_f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auroc": _roc_auc_score(y_test, test_probs),
    }


def _probe_outcome_by_step(
    rows: list[dict[str, object]], reps: np.ndarray
) -> dict[str, dict[str, float]]:
    buckets = ["1", "2", "3", "4+"]
    out: dict[str, dict[str, float]] = {}
    for bucket in buckets:
        idx = np.asarray([i for i, row in enumerate(rows) if row.get("step_bucket") == bucket])
        if len(idx) < 8:
            continue
        bucket_rows = [rows[i] for i in idx]
        bucket_reps = reps[idx]
        report = _probe_binary(
            bucket_rows, bucket_reps, lambda row: int(row["small_full_trace_success"])
        )
        out[bucket] = {
            "test_auroc": report["test_auroc"],
            "test_f1": report["test_f1"],
            "test_base_rate": report["test_base_rate"],
        }
    return out


def _pca_2d(reps: np.ndarray) -> dict[str, object]:
    x = reps.astype(np.float64)
    x = x - np.mean(x, axis=0)
    _u, s, _vh = np.linalg.svd(x, full_matrices=False)
    explained = (s[:2] ** 2) / max(1e-12, np.sum(s**2))
    return {"explained_variance_ratio": explained.tolist()}


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Early Decision Representation Probe",
        "",
        "這份 probe 先用 representation-level signal 測試 early-decision 問題。這一版表示來源還是 embedding，不是真正 hidden states；但 probe 目標與評估方式已對齊 hidden-state probe，之後可直接替換表示來源。",
        "",
        "## 設定",
        "",
        f"- rows：`{report['rows']}`",
        f"- representation type：`{report['representation_type']}`",
        f"- representation model：`{report['representation_model']}`",
        f"- representation dim：`{report['representation_dim']}`",
        "",
        "## Probes",
        "",
    ]
    for name, probe in report["probes"].items():
        lines += [
            f"### `{name}`",
            "",
            f"- train/test：`{probe['train_size']}` / `{probe['test_size']}`",
            f"- train/test base rate：`{probe['train_base_rate']:.3f}` / `{probe['test_base_rate']:.3f}`",
            f"- test AUROC：`{probe['test_auroc']:.3f}`",
            f"- test F1：`{probe['test_f1']:.3f}`",
            f"- test precision / recall：`{probe['test_precision']:.3f}` / `{probe['test_recall']:.3f}`",
            "",
        ]
    lines += [
        "## Outcome By Step",
        "",
    ]
    for bucket, metrics in report["outcome_by_step"].items():
        lines.append(
            f"- step `{bucket}`: AUROC `{metrics['test_auroc']:.3f}`, F1 `{metrics['test_f1']:.3f}`, base rate `{metrics['test_base_rate']:.3f}`"
        )
    lines += [
        "",
        "## PCA",
        "",
        f"- explained variance ratio (PC1/PC2)：`{report['pca']['explained_variance_ratio'][0]:.3f}` / `{report['pca']['explained_variance_ratio'][1]:.3f}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows, reps, meta = _load()
    probes = {
        "small_full_trace_success": _probe_binary(
            rows, reps, lambda row: int(row["small_full_trace_success"])
        ),
        "msp_exists": _probe_binary(rows, reps, lambda row: int(row["msp_exists"])),
        "task_has_positive_takeover": _probe_binary(
            rows, reps, lambda row: int(row["task_has_positive_takeover"])
        ),
        "benchmark_is_livebench": _probe_binary(
            rows, reps, lambda row: int(row["benchmark"] == "livebench_reasoning")
        ),
        "small_family_is_llama": _probe_binary(
            rows, reps, lambda row: int(row["small_family"] == "llama")
        ),
    }
    report = {
        "rows": len(rows),
        "representation_type": meta["representation_type"],
        "representation_model": meta["representation_model"],
        "representation_dim": meta["representation_dim"],
        "benchmark_counts": meta["benchmark_counts"],
        "probes": probes,
        "outcome_by_step": _probe_outcome_by_step(rows, reps),
        "pca": _pca_2d(reps),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote probe report to {OUT_JSON}")
    print(f"Wrote markdown report to {OUT_MD}")
    for name, probe in probes.items():
        print(
            f"{name}: auroc={probe['test_auroc']:.3f} f1={probe['test_f1']:.3f} "
            f"precision={probe['test_precision']:.3f} recall={probe['test_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
