from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "outputs" / "results" / "imported" / "results-livebench-reentry-a07127a"
PROBE_DIR = DATA_DIR / "probe"
REENTRY_DIR = DATA_DIR / "reentry"
OUT_DIR = ROOT / "results" / "_hidden_state_failure_mode_matched_pilot_v1"
OUT_JSON = OUT_DIR / "summary.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "trace"
    / "hidden_state_failure_mode_matched_pilot.md"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    return float((rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives))


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _family_paths() -> dict[str, tuple[Path, Path]]:
    out: dict[str, tuple[Path, Path]] = {}
    for probe_path in sorted(PROBE_DIR.glob("*_no_outliers/reentry_probe_rows.jsonl")):
        family_dir = probe_path.parent.name
        reentry_path = REENTRY_DIR / family_dir / "reentry_rows.jsonl"
        if reentry_path.exists():
            out[family_dir.replace("_reentry_livebench_local_v1_", "")] = (
                probe_path,
                reentry_path,
            )
    return out


def _merged_rows(probe_path: Path, reentry_path: Path) -> list[dict[str, Any]]:
    probe_rows = _load_jsonl(probe_path)
    reentry_rows = _load_jsonl(reentry_path)
    probe_by_prefix = {str(row["prefix_id"]): row for row in probe_rows}

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reentry_rows:
        by_task[str(row["task_id"])].append(row)
    step_counts = {task_id: len(rows) for task_id, rows in by_task.items()}

    out: list[dict[str, Any]] = []
    for row in reentry_rows:
        if int(row.get("reentry_exact_correct", 0) or 0) != 0:
            continue
        probe = probe_by_prefix.get(str(row["prefix_id"]))
        if probe is None:
            continue
        full_correct = int(bool(row.get("full_trace_correct", 0)))
        out.append(
            {
                **row,
                "label_reset_failure": full_correct,
                "group": "reset-failure" if full_correct else "ordinary-failure",
                "mean_hidden": probe["mean_pool_hidden"],
                "last_token_hidden": probe["last_token_hidden"],
                "prefix_tokens_estimated": float(probe.get("prefix_tokens_estimated", 0) or 0),
                "prompt_tokens": float(probe.get("prompt_tokens", 0) or 0),
                "step_frac": float(
                    int(row["step_index"]) / max(1, step_counts[str(row["task_id"])])
                ),
                "hidden_dim": int(probe["hidden_dim"]),
                "selected_layer": int(probe["selected_layer"]),
            }
        )
    return out


def _control_features(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["step_frac"]),
            float(row["prefix_tokens_estimated"]),
            float(row["prompt_tokens"]),
        ],
        dtype=np.float64,
    )


def _match_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reset = [row for row in rows if int(row["label_reset_failure"]) == 1]
    ordinary = [row for row in rows if int(row["label_reset_failure"]) == 0]
    if not reset or not ordinary:
        return [], {"error": "missing_group"}

    all_controls = np.stack([_control_features(row) for row in rows], axis=0)
    means = np.mean(all_controls, axis=0)
    stds = np.std(all_controls, axis=0)
    stds[stds == 0.0] = 1.0

    available = set(range(len(ordinary)))
    matched: list[dict[str, Any]] = []
    distances: list[float] = []
    for r_idx, reset_row in enumerate(reset):
        if not available:
            break
        reset_z = (_control_features(reset_row) - means) / stds
        reset_bin = int(min(3, np.floor(float(reset_row["step_frac"]) * 4.0)))
        candidates = [
            idx
            for idx in available
            if int(min(3, np.floor(float(ordinary[idx]["step_frac"]) * 4.0))) == reset_bin
        ]
        if not candidates:
            candidates = list(available)
        best_idx = min(
            candidates,
            key=lambda idx: float(
                np.linalg.norm(((_control_features(ordinary[idx]) - means) / stds) - reset_z)
            ),
        )
        available.remove(best_idx)
        ordinary_row = ordinary[best_idx]
        ordinary_z = (_control_features(ordinary_row) - means) / stds
        distances.append(float(np.linalg.norm(ordinary_z - reset_z)))
        matched.append({**reset_row, "matched_pair_id": f"p{r_idx:04d}"})
        matched.append({**ordinary_row, "matched_pair_id": f"p{r_idx:04d}"})

    def mean_control(block: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in block])) if block else 0.0

    matched_reset = [row for row in matched if int(row["label_reset_failure"]) == 1]
    matched_ordinary = [row for row in matched if int(row["label_reset_failure"]) == 0]
    meta = {
        "original_reset_rows": len(reset),
        "original_ordinary_rows": len(ordinary),
        "matched_pairs": len(matched) // 2,
        "mean_match_distance": float(np.mean(distances)) if distances else 0.0,
        "control_means": {
            "reset_step_frac": mean_control(matched_reset, "step_frac"),
            "ordinary_step_frac": mean_control(matched_ordinary, "step_frac"),
            "reset_prefix_tokens": mean_control(matched_reset, "prefix_tokens_estimated"),
            "ordinary_prefix_tokens": mean_control(matched_ordinary, "prefix_tokens_estimated"),
            "reset_prompt_tokens": mean_control(matched_reset, "prompt_tokens"),
            "ordinary_prompt_tokens": mean_control(matched_ordinary, "prompt_tokens"),
        },
    }
    return matched, meta


def _probe_binary_repeated(
    rows: list[dict[str, Any]],
    feature_fn: Callable[[dict[str, Any]], list[float]],
    *,
    seeds: int = 50,
    test_ratio: float = 0.25,
) -> dict[str, Any]:
    task_ids = sorted({str(row["task_id"]) for row in rows})
    if len(task_ids) < 4:
        return {"error": "too_few_tasks"}
    aucs: list[float] = []
    oriented_aucs: list[float] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        shuffled = task_ids.copy()
        rng.shuffle(shuffled)
        test_n = max(1, int(round(len(shuffled) * test_ratio)))
        test_tasks = set(shuffled[:test_n])
        train_rows = [row for row in rows if str(row["task_id"]) not in test_tasks]
        test_rows = [row for row in rows if str(row["task_id"]) in test_tasks]
        if not train_rows or not test_rows:
            continue
        y_train = np.asarray(
            [float(row["label_reset_failure"]) for row in train_rows], dtype=np.float64
        )
        y_test = np.asarray(
            [float(row["label_reset_failure"]) for row in test_rows], dtype=np.float64
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        x_train = np.asarray([feature_fn(row) for row in train_rows], dtype=np.float64)
        x_test = np.asarray([feature_fn(row) for row in test_rows], dtype=np.float64)
        x_train, x_test = _project_pca(x_train, x_test)
        x_train, x_test = _zscore(x_train, x_test)
        x_train = np.concatenate([np.ones((len(x_train), 1)), x_train], axis=1)
        x_test = np.concatenate([np.ones((len(x_test), 1)), x_test], axis=1)
        weights = _fit_logistic_regression(x_train, y_train)
        test_probs = _sigmoid(x_test @ weights)
        auc = _roc_auc_score(y_test, test_probs)
        aucs.append(auc)
        oriented_aucs.append(max(auc, 1.0 - auc))
    if not aucs:
        return {"error": "no_valid_splits"}
    return {
        "valid_splits": len(aucs),
        "mean_auroc": float(np.mean(aucs)),
        "median_auroc": float(np.median(aucs)),
        "mean_orientation_free_auroc": float(np.mean(oriented_aucs)),
        "median_orientation_free_auroc": float(np.median(oriented_aucs)),
        "p25_auroc": float(np.percentile(aucs, 25)),
        "p75_auroc": float(np.percentile(aucs, 75)),
    }


def _centroid_gap(rows: list[dict[str, Any]], vector_key: str) -> dict[str, Any]:
    reset = [
        np.asarray(row[vector_key], dtype=np.float64) for row in rows if row["label_reset_failure"]
    ]
    ordinary = [
        np.asarray(row[vector_key], dtype=np.float64)
        for row in rows
        if not row["label_reset_failure"]
    ]
    if not reset or not ordinary:
        return {"error": "missing_group"}
    reset_centroid = np.mean(np.stack(reset, axis=0), axis=0)
    ordinary_centroid = np.mean(np.stack(ordinary, axis=0), axis=0)
    return {
        "reset_rows": len(reset),
        "ordinary_rows": len(ordinary),
        "centroid_cosine": _cosine(reset_centroid, ordinary_centroid),
        "centroid_l2": float(np.linalg.norm(reset_centroid - ordinary_centroid)),
    }


def _family_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows, match_meta = _match_rows(rows)
    feature_fns: dict[str, Callable[[dict[str, Any]], list[float]]] = {
        "mean_hidden": lambda row: cast(list[float], row["mean_hidden"]),
        "last_token_hidden": lambda row: cast(list[float], row["last_token_hidden"]),
        "controls_step_tokens": lambda row: [
            float(row["step_frac"]),
            float(row["prefix_tokens_estimated"]),
            float(row["prompt_tokens"]),
        ],
    }
    repeated_probes = {
        name: _probe_binary_repeated(matched_rows, feature_fn)
        for name, feature_fn in feature_fns.items()
    }
    return {
        "rows": len(rows),
        "matched_rows": len(matched_rows),
        "tasks": len({str(row["task_id"]) for row in rows}),
        "matched_tasks": len({str(row["task_id"]) for row in matched_rows}),
        "match_meta": match_meta,
        "selected_layer": int(rows[0]["selected_layer"]) if rows else 0,
        "hidden_dim": int(rows[0]["hidden_dim"]) if rows else 0,
        "centroid_gaps": {
            vector_key: _centroid_gap(matched_rows, vector_key)
            for vector_key in ["mean_hidden", "last_token_hidden"]
        },
        "repeated_probes": repeated_probes,
    }


def _aggregate(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ["mean_hidden", "last_token_hidden", "controls_step_tokens"]:
        vals = [
            float(family["repeated_probes"][name]["mean_auroc"])
            for family in families.values()
            if "mean_auroc" in family["repeated_probes"][name]
        ]
        sep_vals = [
            float(family["repeated_probes"][name]["mean_orientation_free_auroc"])
            for family in families.values()
            if "mean_orientation_free_auroc" in family["repeated_probes"][name]
        ]
        out[name] = {
            "mean_auroc": float(np.mean(vals)) if vals else 0.5,
            "mean_orientation_free_auroc": float(np.mean(sep_vals)) if sep_vals else 0.5,
            "family_count": len(vals),
        }
    return out


def _fmt_repeated(probe: dict[str, Any]) -> str:
    if "mean_auroc" not in probe:
        return "n/a"
    return (
        f"{probe['mean_auroc']:.3f} "
        f"(sep={probe['mean_orientation_free_auroc']:.3f}, n={probe['valid_splits']})"
    )


def _to_markdown(summary: dict[str, Any]) -> str:
    families = cast(dict[str, dict[str, Any]], summary["families"])
    aggregate = cast(dict[str, dict[str, Any]], summary["aggregate"])
    lines = [
        "# Matched Hidden-State Failure Mode Pilot",
        "",
        "## Question",
        "",
        "- Repeat the reset-vs-ordinary failure pilot after matching on step fraction and token counts.",
        "- This checks whether separability survives after removing the most obvious prefix-position confound.",
        "",
        "## Aggregate",
        "",
        "| Feature | Matched repeated mean AUROC | Matched orientation-free separability | Families |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ["mean_hidden", "last_token_hidden", "controls_step_tokens"]:
        lines.append(
            f"| {name} | {aggregate[name]['mean_auroc']:.3f} | "
            f"{aggregate[name]['mean_orientation_free_auroc']:.3f} | "
            f"{aggregate[name]['family_count']} |"
        )

    lines += ["", "## Family Breakdown", ""]
    for family, report in sorted(families.items()):
        meta = report["match_meta"]
        lines.append(f"### `{family}`")
        lines.append("")
        lines.append(
            f"- matched pairs: `{meta.get('matched_pairs', 0)}` "
            f"(reset original `{meta.get('original_reset_rows', 0)}`, "
            f"ordinary original `{meta.get('original_ordinary_rows', 0)}`)"
        )
        controls = meta.get("control_means", {})
        if controls:
            lines.append(
                f"- matched step frac: reset `{controls['reset_step_frac']:.3f}`, "
                f"ordinary `{controls['ordinary_step_frac']:.3f}`"
            )
            lines.append(
                f"- matched prefix tokens: reset `{controls['reset_prefix_tokens']:.1f}`, "
                f"ordinary `{controls['ordinary_prefix_tokens']:.1f}`"
            )
        lines.append("")
        lines.append("| Feature | Matched repeated split AUROC |")
        lines.append("| --- | ---: |")
        for name in ["mean_hidden", "last_token_hidden", "controls_step_tokens"]:
            lines.append(f"| {name} | {_fmt_repeated(report['repeated_probes'][name])} |")
        lines.append("")
        lines.append("| Pooling | Matched centroid cosine | Matched centroid L2 |")
        lines.append("| --- | ---: | ---: |")
        for vector_key, gap in report["centroid_gaps"].items():
            if "centroid_cosine" not in gap:
                lines.append(f"| {vector_key} | n/a | n/a |")
            else:
                lines.append(
                    f"| {vector_key} | {gap['centroid_cosine']:.3f} | {gap['centroid_l2']:.3f} |"
                )
        lines.append("")
    lines += [
        "## Read",
        "",
        "- A meaningful hidden-state result should remain above the matched step/token control.",
        "- If the control remains competitive, the failure-mode distinction is not yet clean enough for a central claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, dict[str, Any]] = {}
    for family, (probe_path, reentry_path) in _family_paths().items():
        rows = _merged_rows(probe_path, reentry_path)
        if not rows:
            continue
        families[family] = _family_analysis(rows)
    summary = {
        "source_dir": str(DATA_DIR),
        "definition": {
            "reset_failure": "full_trace_correct == True and reentry_exact_correct == False",
            "ordinary_failure": "full_trace_correct == False and reentry_exact_correct == False",
            "matching": "greedy nearest-neighbor matching on step_frac, prefix_tokens_estimated, prompt_tokens within family",
        },
        "families": families,
        "aggregate": _aggregate(families),
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
