from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, cast

import numpy as np

from confidence_tom.infra.paths import project_root, results_root

ROOT = project_root()
RESULTS_DIR = results_root()
OUT_DIR = RESULTS_DIR / "_prefix_predictor_v1"
OUT_JSON = OUT_DIR / "task_structure_analysis.json"
OUT_MD = (
    ROOT
    / "docs"
    / "mainline"
    / "generated"
    / "analysis"
    / "prefix"
    / "prefix_task_structure_analysis.md"
)
OUT_MAP = OUT_DIR / "task_structure_rank_variance_map.png"
OUT_CLUSTER = OUT_DIR / "task_structure_clusters.png"

RUN_NAMES = [
    "qwen_to_openai_50",
    "qwen_to_anthropic_50",
    "llama_to_openai_50",
    "llama_to_anthropic_50",
    "mistral_to_openai_50",
    "mistral_to_anthropic_50",
]

SummaryMap = dict[str, dict[str, Any]]
TaskFeatureRow = dict[str, Any]
ClusterInfo = dict[str, Any]


def _load_summaries() -> SummaryMap:
    out: SummaryMap = {}
    for run_name in RUN_NAMES:
        path = RESULTS_DIR / run_name / "summary.json"
        out[run_name] = cast(dict[str, Any], json.loads(path.read_text()))
    return out


def _task_feature_rows(summaries: SummaryMap) -> list[TaskFeatureRow]:
    all_task_ids: set[str] = set()
    for summary in summaries.values():
        all_task_ids.update(cast(dict[str, Any], summary["per_task_summary"]).keys())

    rows: list[TaskFeatureRow] = []
    for task_id in sorted(all_task_ids):
        pos_fracs: list[float] = []
        neg_fracs: list[float] = []
        any_positives: list[int] = []
        step_counts: list[int] = []
        feature_row: TaskFeatureRow = {"task_id": task_id}

        for run_name in RUN_NAMES:
            summary = summaries[run_name]
            task_summary = cast(dict[str, Any], summary["per_task_summary"])[task_id]
            pos = int(task_summary["positive"])
            zero = int(task_summary["zero"])
            neg = int(task_summary["negative"])
            total = max(1, pos + zero + neg)
            pos_frac = pos / total
            neg_frac = neg / total
            any_pos = int(pos > 0)

            feature_row[f"{run_name}__positive_fraction"] = pos_frac
            feature_row[f"{run_name}__negative_fraction"] = neg_frac
            feature_row[f"{run_name}__any_positive"] = any_pos

            pos_fracs.append(pos_frac)
            neg_fracs.append(neg_frac)
            any_positives.append(any_pos)
            step_counts.append(total)

        mean_pos = float(np.mean(pos_fracs))
        var_pos = float(np.var(pos_fracs))
        mean_neg = float(np.mean(neg_fracs))
        agreement = float(np.mean(any_positives))
        mean_steps = float(np.mean(step_counts))

        feature_row["mean_positive_fraction"] = mean_pos
        feature_row["variance_positive_fraction"] = var_pos
        feature_row["mean_negative_fraction"] = mean_neg
        feature_row["positive_run_count"] = int(sum(any_positives))
        feature_row["agreement_any_positive_rate"] = agreement
        feature_row["mean_step_count"] = mean_steps
        rows.append(feature_row)

    return rows


def _kmeans(
    points: np.ndarray, k: int, seed: int = 0, steps: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(points) < k:
        raise ValueError("k cannot exceed number of points")

    indices = rng.choice(len(points), size=k, replace=False)
    centroids = points[indices].copy()

    for _ in range(steps):
        dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            members = points[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    return labels, centroids


def _cluster_name(centroid: np.ndarray) -> str:
    mean_pos, var_pos, mean_neg, agreement = centroid[:4]
    if agreement >= 0.8 and mean_pos >= 0.35:
        return "stable_recoverable"
    if mean_pos < 0.08 and mean_neg < 0.05:
        return "low_gain_stable"
    if var_pos >= 0.08:
        return "pairing_sensitive"
    if mean_neg >= 0.12 and mean_pos < 0.2:
        return "negative_risk"
    return "mixed_regime"


def _build_clusters(rows: list[TaskFeatureRow], k: int = 4) -> dict[str, Any]:
    feature_names = [
        "mean_positive_fraction",
        "variance_positive_fraction",
        "mean_negative_fraction",
        "agreement_any_positive_rate",
        "mean_step_count",
    ]
    points = np.array(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    stds[stds == 0.0] = 1.0
    z = (points - means) / stds

    labels, centroids_z = _kmeans(z, k=k, seed=0)
    centroids = centroids_z * stds + means

    enriched_rows: list[TaskFeatureRow] = []
    cluster_members: dict[int, list[TaskFeatureRow]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        enriched = dict(row)
        enriched["cluster_id"] = int(label)
        cluster_members[int(label)].append(enriched)
        enriched_rows.append(enriched)

    clusters: list[ClusterInfo] = []
    for cluster_id in range(k):
        centroid = centroids[cluster_id]
        members = cluster_members[cluster_id]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": _cluster_name(centroid),
                "size": len(members),
                "centroid": {
                    feature_names[i]: float(centroid[i]) for i in range(len(feature_names))
                },
                "members": sorted(
                    members,
                    key=lambda row: (
                        -float(row["mean_positive_fraction"]),
                        float(row["variance_positive_fraction"]),
                    ),
                )[:12],
            }
        )

    return {
        "feature_names": feature_names,
        "rows": enriched_rows,
        "clusters": clusters,
    }


def _annotate_top(ax: Any, rows: list[TaskFeatureRow], *, top_k: int = 10) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_positive_fraction"]),
            -float(row["variance_positive_fraction"]),
        ),
    )[:top_k]
    for row in ranked:
        x = float(row["mean_positive_fraction"])
        y = float(row["variance_positive_fraction"])
        label = str(row["task_id"]).replace("olympiadbench_", "")
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)


def _write_plots(rows: list[TaskFeatureRow], clusters: list[ClusterInfo]) -> None:
    import matplotlib.pyplot as plt

    cluster_colors = {
        0: "#4C78A8",
        1: "#F58518",
        2: "#54A24B",
        3: "#E45756",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [float(row["mean_positive_fraction"]) for row in rows]
    ys = [float(row["variance_positive_fraction"]) for row in rows]
    sizes = [40 + 18 * int(row["positive_run_count"]) for row in rows]
    ax.scatter(xs, ys, s=sizes, alpha=0.75, color="#4C78A8", edgecolors="white", linewidths=0.6)
    _annotate_top(ax, rows, top_k=12)
    ax.set_title("Task-Level Rank / Variance Map")
    ax.set_xlabel("Mean Positive Fraction Across 6 Runs")
    ax.set_ylabel("Variance of Positive Fraction Across 6 Runs")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_MAP, dpi=180)
    plt.close(fig)

    cluster_name_by_id = {
        int(cluster["cluster_id"]): str(cluster["cluster_name"]) for cluster in clusters
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in sorted({int(row["cluster_id"]) for row in rows}):
        block = [row for row in rows if int(row["cluster_id"]) == cluster_id]
        xs = [float(row["mean_positive_fraction"]) for row in block]
        ys = [float(row["variance_positive_fraction"]) for row in block]
        ax.scatter(
            xs,
            ys,
            s=70,
            alpha=0.82,
            color=cluster_colors.get(cluster_id, "#777777"),
            edgecolors="white",
            linewidths=0.6,
            label=f"{cluster_id}: {cluster_name_by_id.get(cluster_id, 'cluster')}",
        )
    _annotate_top(ax, rows, top_k=12)
    ax.set_title("Task-Level Structure Clusters")
    ax.set_xlabel("Mean Positive Fraction Across 6 Runs")
    ax.set_ylabel("Variance of Positive Fraction Across 6 Runs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_CLUSTER, dpi=180)
    plt.close(fig)


def _to_markdown(map_rows: list[TaskFeatureRow], clustering: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Prefix Task Structure Analysis")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("這份分析做兩件事：")
    lines.append("")
    lines.append(
        "1. 用 task-level `mean positive fraction` 和 "
        "`variance positive fraction` 建立 rank / variance map。"
    )
    lines.append("2. 用 task-level 統計特徵做簡單 clustering，先看資料自然長出哪些群。")
    lines.append("")
    lines.append("## Rank / Variance Map")
    lines.append("")
    lines.append(
        "| Task | Positive Runs | Mean Positive Fraction | Variance | Mean Negative Fraction |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in sorted(
        map_rows,
        key=lambda row: (
            -float(row["mean_positive_fraction"]),
            float(row["variance_positive_fraction"]),
        ),
    )[:15]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/6 | "
            f"{float(row['mean_positive_fraction']):.3f} | "
            f"{float(row['variance_positive_fraction']):.4f} | "
            f"{float(row['mean_negative_fraction']):.3f} |"
        )
    lines.append("")
    lines.append("### Highest Variance Tasks")
    lines.append("")
    lines.append("| Task | Positive Runs | Mean Positive Fraction | Variance |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in sorted(
        map_rows,
        key=lambda row: float(row["variance_positive_fraction"]),
        reverse=True,
    )[:15]:
        lines.append(
            f"| `{row['task_id']}` | {row['positive_run_count']}/6 | "
            f"{float(row['mean_positive_fraction']):.3f} | "
            f"{float(row['variance_positive_fraction']):.4f} |"
        )
    lines.append("")
    lines.append("## Clustering")
    lines.append("")
    for cluster in sorted(
        cast(list[ClusterInfo], clustering["clusters"]),
        key=lambda item: int(item["cluster_id"]),
    ):
        lines.append(
            f"### Cluster {cluster['cluster_id']}: `{cluster['cluster_name']}` "
            f"(size={cluster['size']})"
        )
        lines.append("")
        lines.append("Centroid:")
        for key, value in cast(dict[str, float], cluster["centroid"]).items():
            lines.append(f"- `{key}`: {value:.3f}")
        lines.append("")
        lines.append("Representative tasks:")
        for row in cluster["members"][:8]:
            lines.append(
                f"- `{row['task_id']}`: "
                f"mean_pos={float(row['mean_positive_fraction']):.3f}, "
                f"var={float(row['variance_positive_fraction']):.4f}, "
                f"neg={float(row['mean_negative_fraction']):.3f}, "
                f"positive_runs={row['positive_run_count']}/6"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze task structure and optionally emit plots."
    )
    parser.add_argument("--skip-plots", action="store_true", help="Write JSON/Markdown only.")
    args = parser.parse_args()

    summaries = _load_summaries()
    map_rows = _task_feature_rows(summaries)
    clustering = _build_clusters(map_rows, k=4)

    report = {
        "task_map_rows": map_rows,
        "clustering": clustering,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(map_rows, clustering), encoding="utf-8")
    if not args.skip_plots:
        _write_plots(
            cast(list[TaskFeatureRow], clustering["rows"]),
            cast(list[ClusterInfo], clustering["clusters"]),
        )
    print(f"Wrote JSON report to {OUT_JSON}")
    print(f"Wrote Markdown report to {OUT_MD}")
    if not args.skip_plots:
        print(f"Wrote rank/variance map to {OUT_MAP}")
        print(f"Wrote cluster plot to {OUT_CLUSTER}")


if __name__ == "__main__":
    main()
