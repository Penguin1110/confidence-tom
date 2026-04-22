from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "_prefix_embedding_v1"
ROWS_PATH = DATA_DIR / "pilot_rows.jsonl"
EMBED_PATH = DATA_DIR / "pilot_embeddings.npz"
OUT_JSON = DATA_DIR / "pilot_probe_results.json"
OUT_MD = ROOT / "docs" / "prefix_embedding_pilot.md"


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
    steps: int = 3000,
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


def _load() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = [
        json.loads(line)
        for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = cast(np.ndarray, np.load(EMBED_PATH)["embeddings"])
    return cast(list[dict[str, Any]], rows), matrix


def _split(rows: list[dict[str, Any]], emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = [i for i, row in enumerate(rows) if not _stable_test_split(str(row["task_id"]))]
    test_idx = [i for i, row in enumerate(rows) if _stable_test_split(str(row["task_id"]))]
    return np.asarray(train_idx), np.asarray(test_idx)


def _zscore(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0)
    stds[stds == 0.0] = 1.0
    return (train_x - means) / stds, (test_x - means) / stds


def _probe_binary(
    rows: list[dict[str, Any]], emb: np.ndarray, target_key: str, positive_value: object
) -> dict[str, float | int]:
    train_idx, test_idx = _split(rows, emb)
    x_train, x_test = emb[train_idx], emb[test_idx]
    y_train = np.asarray(
        [1.0 if rows[i][target_key] == positive_value else 0.0 for i in train_idx], dtype=np.float64
    )
    y_test = np.asarray(
        [1.0 if rows[i][target_key] == positive_value else 0.0 for i in test_idx], dtype=np.float64
    )
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


def _pca_2d(emb: np.ndarray) -> dict[str, list[list[float]] | list[float]]:
    x = emb.astype(np.float64)
    x = x - np.mean(x, axis=0)
    u, s, _vh = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s[:2] ** 2) / max(1e-12, np.sum(s**2))
    return {
        "coords": coords.tolist(),
        "explained_variance_ratio": explained.tolist(),
    }


def _to_markdown(report: dict[str, Any]) -> str:
    probes = cast(dict[str, dict[str, float | int]], report["probes"])
    pca = cast(dict[str, list[float]], report["pca"])
    lines = [
        "# Prefix Embedding Pilot",
        "",
        "## 設定",
        "",
        f"- 樣本數：`{report['rows']}`",
        f"- embedding model：`{report['embedding_model']}`",
        f"- embedding dim：`{report['embedding_dim']}`",
        "",
        "## Probe 結果",
        "",
    ]
    for name, probe in probes.items():
        lines += [
            f"### `{name}`",
            "",
            f"- train/test：`{probe['train_size']}` / `{probe['test_size']}`",
            (
                f"- train/test base rate：`{probe['train_base_rate']:.3f}` / "
                f"`{probe['test_base_rate']:.3f}`"
            ),
            f"- test AUROC：`{probe['test_auroc']:.3f}`",
            f"- test F1：`{probe['test_f1']:.3f}`",
            (
                f"- test precision / recall：`{probe['test_precision']:.3f}` / "
                f"`{probe['test_recall']:.3f}`"
            ),
            "",
        ]
    lines += [
        "## PCA",
        "",
        (
            "- explained variance ratio (PC1/PC2)："
            f"`{pca['explained_variance_ratio'][0]:.3f}` / "
            f"`{pca['explained_variance_ratio'][1]:.3f}`"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows, emb = _load()
    probes = {
        "delta_positive": _probe_binary(rows, emb, "delta_positive", 1),
        "benchmark_is_livebench": _probe_binary(rows, emb, "benchmark", "livebench_reasoning"),
        "small_family_is_llama": _probe_binary(rows, emb, "small_family", "llama"),
    }
    meta = cast(
        dict[str, Any],
        json.loads(
            (DATA_DIR / "pilot_meta.json").read_text(encoding="utf-8"),
        ),
    )
    pca = _pca_2d(emb)
    report = {
        "rows": len(rows),
        "embedding_model": meta["embedding_model"],
        "embedding_dim": meta["embedding_dim"],
        "benchmark_counts": meta["benchmark_counts"],
        "probes": probes,
        "pca": {
            "explained_variance_ratio": pca["explained_variance_ratio"],
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote probe report to {OUT_JSON}")
    print(f"Wrote markdown report to {OUT_MD}")
    for name, probe in probes.items():
        print(
            f"{name}: auroc={probe['test_auroc']:.3f} f1={probe['test_f1']:.3f} "
            f"precision={probe['test_precision']:.3f} recall={probe['test_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
