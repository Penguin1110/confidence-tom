from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = ROOT / "results" / "_prefix_predictor_v1" / "task_structure_analysis.json"
OUT_DIR = ROOT / "results" / "_prefix_predictor_v1"
OUT_MAP = OUT_DIR / "task_structure_rank_variance_map.png"
OUT_CLUSTER = OUT_DIR / "task_structure_clusters.png"


CLUSTER_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#E45756",
}


def _load() -> dict:
    return json.loads(INPUT_JSON.read_text())


def _annotate_top(ax: plt.Axes, rows: list[dict], *, top_k: int = 10) -> None:
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


def _plot_rank_variance(rows: list[dict]) -> None:
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


def _plot_clusters(rows: list[dict], clusters: list[dict]) -> None:
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
            color=CLUSTER_COLORS.get(cluster_id, "#777777"),
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


def main() -> None:
    data = _load()
    rows = data["clustering"]["rows"]
    clusters = data["clustering"]["clusters"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_rank_variance(rows)
    _plot_clusters(rows, clusters)
    print(f"Wrote rank/variance map to {OUT_MAP}")
    print(f"Wrote cluster plot to {OUT_CLUSTER}")


if __name__ == "__main__":
    main()
