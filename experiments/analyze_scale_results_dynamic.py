"""Phase 3+4: Analyze scale experiment results.

Computes calibration metrics and generates all key figures:
  - Fig 1: Scale vs Accuracy
  - Fig 2: Scale vs Mean Reported Confidence
  - Fig 3: Scale vs Miscalibration Gap (main figure)
  - Fig 4: Difficulty-Conditioned Gap (faceted)

Usage:
    uv run python experiments/analyze_scale_results.py
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_tom.metrics import (
    DIFFICULTY_LABELS,
    CalibrationReport,
    compute_calibration_report,
    compute_empirical_difficulty,
    stratify_by_difficulty,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Style configuration ----

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 12,
    }
)

SCALE_ORDER = [
    "Gemma-3-4B",
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Qwen-3-8B",
    "Qwen-3-14B",
    "Qwen-3-32B",
    "Qwen-3.5-27B",
    "Qwen-3.5-35B-A3B",
    "Qwen-3.5-122B-A10B",
    "Qwen-3.5-397B-A17B",
    "Llama-4-Scout",
    "Llama-4-Maverick",
    "GPT-OSS-20B",
    "GPT-OSS-120B",
]

SCALE_COLORS = {
    "Gemma-3-4B": "#f97583",
    "Gemma-3-12B": "#79c0ff",
    "Gemma-3-27B": "#56d364",
    "Qwen-3-8B": "#d2a8ff",
    "Qwen-3-14B": "#d2a8ff",
    "Qwen-3-32B": "#d2a8ff",
    "Qwen-3.5-27B": "#d2a8ff",
    "Qwen-3.5-35B-A3B": "#d2a8ff",
    "Qwen-3.5-122B-A10B": "#d2a8ff",
    "Qwen-3.5-397B-A17B": "#d2a8ff",
    "Llama-4-Scout": "#ffa657",
    "Llama-4-Maverick": "#ffa657",
    "GPT-OSS-20B": "#ff7b72",
    "GPT-OSS-120B": "#ff7b72",
}

CATEGORY_COLORS = {
    "math": "#ff7b72",
    "science": "#79c0ff",
    "knowledge": "#d2a8ff",
    "truthfulness": "#ffa657",
}


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load results from all scale model JSON files into a single DataFrame."""
    all_rows = []
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        json_files = sorted(results_dir.glob("*/*.json"))

    for json_file in json_files:
        if json_file.name.endswith(".tmp.json"):
            continue
        if json_file.name == "calibration_summary.json":
            continue
        logger.info(f"Loading {json_file.relative_to(results_dir)}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        benchmark = json_file.parent.name if json_file.parent != results_dir else "unknown"
        for item in data:
            question_id = item.get("question_id", item.get("task_id"))
            question = item.get("question", item.get("instruction", ""))
            correct_answer = item.get("correct_answer", item.get("benchmark_metadata", {}))
            source = item.get("source", benchmark)
            is_correct = item.get("is_correct", item.get("majority_correct"))
            c_rep = item.get("c_rep")
            gap = item.get("gap")
            all_rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "correct_answer": correct_answer,
                    "category": item.get("category", benchmark),
                    "source": source,
                    "external_difficulty": item.get("external_difficulty"),
                    "model_name": item.get("model_name", json_file.stem),
                    "model_label": json_file.stem,
                    "majority_answer": item.get("majority_answer", item.get("majority_correct")),
                    "is_correct": is_correct,
                    "c_beh": item["c_beh"],
                    "c_rep": np.nan if c_rep is None else c_rep,
                    "gap": np.nan if gap is None else gap,
                    "k_samples": item["k_samples"],
                    "benchmark": benchmark,
                }
            )

    df = pd.DataFrame(all_rows)
    logger.info(f"Loaded {len(df)} total results across {df['model_label'].nunique()} models")
    return df


def compute_reports(df: pd.DataFrame) -> dict[str, CalibrationReport]:
    """Compute CalibrationReport for each model."""
    reports = {}
    for label in SCALE_ORDER:
        model_df = df[df["model_label"] == label]
        if model_df.empty:
            continue

        valid_rep = model_df.dropna(subset=["c_rep"])
        if valid_rep.empty:
            logger.warning(f"Skipping calibration report for {label}: no reported confidence in native results")
            continue

        report = compute_calibration_report(
            model_name=label,
            c_reps=valid_rep["c_rep"].to_numpy(dtype=float),
            c_behs=valid_rep["c_beh"].to_numpy(dtype=float),
            is_correct=valid_rep["is_correct"].astype(float).to_numpy(),
        )
        reports[label] = report
        logger.info("\n" + report.display_str())

    return reports


def add_empirical_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Add empirical difficulty and difficulty bucket to DataFrame."""
    # Pivot to get per-question accuracy by model
    question_ids = sorted(df["question_id"].unique())
    model_labels = [m for m in SCALE_ORDER if m in df["model_label"].unique()]

    # Build accuracy matrix
    acc_by_model = {}
    for label in model_labels:
        model_df = df[df["model_label"] == label].set_index("question_id")
        acc_array = np.array(
            [
                float(model_df.loc[qid, "is_correct"]) if qid in model_df.index else 0.0
                for qid in question_ids
            ]
        )
        acc_by_model[label] = acc_array

    difficulties = compute_empirical_difficulty(acc_by_model)
    buckets = stratify_by_difficulty(difficulties)

    # Map back to DataFrame
    qid_to_diff = {
        qid: (float(difficulties[i]), int(buckets[i])) for i, qid in enumerate(question_ids)
    }

    df["empirical_difficulty"] = df["question_id"].map(lambda x: qid_to_diff.get(x, (0.5, 1))[0])
    df["difficulty_bucket"] = df["question_id"].map(
        lambda x: DIFFICULTY_LABELS[qid_to_diff.get(x, (0.5, 1))[1]]
    )

    return df


# ---- Figure 1: Scale vs Accuracy ----


def plot_scale_vs_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart showing accuracy increases with scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_labels = [m for m in SCALE_ORDER if m in df["model_label"].unique()]
    accuracies = [df[df["model_label"] == m]["is_correct"].mean() * 100 for m in model_labels]

    bars = ax.bar(
        model_labels,
        accuracies,
        color=[SCALE_COLORS[m] for m in model_labels],
        edgecolor="#30363d",
        linewidth=1.5,
        width=0.5,
    )

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Figure 1: Scale vs Accuracy", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_scale_vs_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig1_scale_vs_accuracy.png")


# ---- Figure 2: Scale vs Mean Confidence ----


def plot_scale_vs_confidence(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: C_rep vs C_beh vs Accuracy per scale."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_labels = [m for m in SCALE_ORDER if m in df["model_label"].unique()]
    x = np.arange(len(model_labels))
    width = 0.25

    c_reps = []
    for m in model_labels:
        values = df[df["model_label"] == m]["c_rep"].dropna()
        c_reps.append((values.mean() * 100) if not values.empty else np.nan)
    c_behs = [df[df["model_label"] == m]["c_beh"].mean() * 100 for m in model_labels]
    accs = [df[df["model_label"] == m]["is_correct"].mean() * 100 for m in model_labels]

    ax.bar(
        x - width,
        c_reps,
        width,
        label="C_rep (Reported)",
        color="#ffa657",
        edgecolor="#30363d",
    )
    ax.bar(
        x,
        c_behs,
        width,
        label="C_beh (Behavioral)",
        color="#79c0ff",
        edgecolor="#30363d",
    )
    ax.bar(
        x + width,
        accs,
        width,
        label="Accuracy",
        color="#56d364",
        edgecolor="#30363d",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_title("Figure 2: Scale vs Confidence Components", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_scale_vs_confidence.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig2_scale_vs_confidence.png")


# ---- Figure 3: Scale vs Miscalibration Gap (MAIN FIGURE) ----


def plot_scale_vs_gap(df: pd.DataFrame, output_dir: Path) -> None:
    """Main figure: box plot of per-question Gap by scale model."""
    gap_df = df.dropna(subset=["gap"])
    if gap_df.empty:
        logger.warning("  Skipping fig3: no gap values available in these results")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    model_labels = [m for m in SCALE_ORDER if m in gap_df["model_label"].unique()]

    # Box plot of gaps
    plot_data = [gap_df[gap_df["model_label"] == m]["gap"].values * 100 for m in model_labels]

    bp = ax.boxplot(
        plot_data,
        tick_labels=model_labels,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.3, "color": "#8b949e"},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#8b949e"},
        capprops={"color": "#8b949e"},
    )

    for patch, label in zip(bp["boxes"], model_labels):
        patch.set_facecolor(SCALE_COLORS[label])
        patch.set_alpha(0.7)
        patch.set_edgecolor("#30363d")

    # Add mean markers
    for i, label in enumerate(model_labels):
        mean_val = gap_df[gap_df["model_label"] == label]["gap"].mean() * 100
        ax.scatter(
            i + 1,
            mean_val,
            color="white",
            marker="D",
            s=50,
            zorder=5,
            edgecolor="#30363d",
            linewidth=1,
        )
        ax.annotate(
            f"μ={mean_val:+.1f}%",
            (i + 1, mean_val),
            textcoords="offset points",
            xytext=(15, -5),
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.axhline(
        y=0, color="#f0883e", linestyle="--", alpha=0.7, linewidth=1.5, label="Perfect calibration"
    )
    ax.set_ylabel("Miscalibration Gap (%)\n(+ = Overconfident, − = Underconfident)", fontsize=12)
    ax.set_title("Figure 3: Scale vs Miscalibration Gap", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_scale_vs_gap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig3_scale_vs_gap.png")


# ---- Figure 4: Difficulty-Conditioned Gap ----


def plot_difficulty_conditioned_gap(df: pd.DataFrame, output_dir: Path) -> None:
    """Faceted plot: Gap by scale × difficulty bucket × category."""
    if "difficulty_bucket" not in df.columns:
        logger.warning("  Skipping fig4: difficulty_bucket not computed yet")
        return
    if df["gap"].dropna().empty:
        logger.warning("  Skipping fig4: no gap values available in these results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    difficulty_order = ["easy", "medium", "hard"]

    model_labels = [m for m in SCALE_ORDER if m in df["model_label"].unique()]

    for idx, diff in enumerate(difficulty_order):
        ax = axes[idx]
        subset = df[df["difficulty_bucket"] == diff]

        if subset.empty:
            ax.set_title(f"{diff.capitalize()}\n(no data)", fontsize=13, fontweight="bold")
            continue

        for model in model_labels:
            model_subset = subset[subset["model_label"] == model]
            if model_subset.empty:
                continue

            gap_values = model_subset["gap"].dropna()
            if gap_values.empty:
                continue
            mean_gap = gap_values.mean() * 100
            se = gap_values.std() * 100 / (len(gap_values) ** 0.5)

            ax.bar(
                model,
                mean_gap,
                yerr=se,
                color=SCALE_COLORS[model],
                edgecolor="#30363d",
                capsize=5,
                error_kw={"color": "#c9d1d9", "linewidth": 1.5},
                width=0.5,
                alpha=0.8,
            )

            ax.text(
                model,
                mean_gap + se + 1,
                f"{mean_gap:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.axhline(y=0, color="#f0883e", linestyle="--", alpha=0.5)
        ax.set_title(
            f"{diff.capitalize()} Questions\n(D {'<' if diff == 'easy' else '≥'} "
            f"{'0.33' if diff == 'easy' else '0.67' if diff == 'hard' else '0.33–0.67'})",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.set_ylabel("Mean Gap (%)\n(+ = Overconfident)", fontsize=12)

    fig.suptitle(
        "Figure 4: Difficulty-Conditioned Miscalibration Gap",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_difficulty_gap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig4_difficulty_gap.png")


# ---- Figure 4b: Category-Conditioned Gap ----


def plot_category_conditioned_gap(df: pd.DataFrame, output_dir: Path) -> None:
    """Gap by scale × task category (math, science, knowledge, truthfulness)."""
    if df["gap"].dropna().empty:
        logger.warning("  Skipping fig4b: no gap values available in these results")
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    categories = ["math", "science", "knowledge", "truthfulness"]
    model_labels = [m for m in SCALE_ORDER if m in df["model_label"].unique()]

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        subset = df[df["category"] == cat]

        for model in model_labels:
            model_subset = subset[subset["model_label"] == model]
            if model_subset.empty:
                continue

            gap_values = model_subset["gap"].dropna()
            if gap_values.empty:
                continue
            mean_gap = gap_values.mean() * 100
            se = gap_values.std() * 100 / max(1, len(gap_values) ** 0.5)

            ax.bar(
                model,
                mean_gap,
                yerr=se,
                color=SCALE_COLORS[model],
                edgecolor="#30363d",
                capsize=5,
                error_kw={"color": "#c9d1d9", "linewidth": 1.5},
                width=0.5,
                alpha=0.8,
            )

        ax.axhline(y=0, color="#f0883e", linestyle="--", alpha=0.5)
        ax.set_title(f"{cat.capitalize()}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.set_ylabel("Mean Gap (%)", fontsize=12)

    fig.suptitle(
        "Figure 4b: Category-Conditioned Miscalibration Gap", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fig4b_category_gap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig4b_category_gap.png")


# ---- Summary table ----


def save_summary_table(reports: dict[str, CalibrationReport], output_dir: Path) -> None:
    """Save summary metrics as JSON and CSV."""
    rows = [r.to_dict() for r in reports.values()]
    df = pd.DataFrame(rows)

    # Save as JSON
    with open(output_dir / "calibration_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Save as CSV
    df.to_csv(output_dir / "calibration_summary.csv", index=False)
    logger.info("  Saved calibration_summary.json and .csv")

    # Print formatted table
    logger.info("\n📊 Calibration Summary (×100 for display):")
    logger.info(
        f"{'Model':<16} {'Acc%':>6} {'C_rep%':>7} {'C_beh%':>7} "
        f"{'Gap%':>7} {'|Gap|%':>7} {'ECE%':>6} "
        f"{'Brier_a':>8} {'Brier_i':>8} {'OC%':>6}"
    )
    logger.info("-" * 90)
    for r in reports.values():
        logger.info(
            f"{r.model_name:<16} "
            f"{r.accuracy * 100:6.1f} "
            f"{r.mean_reported_confidence * 100:7.1f} "
            f"{r.mean_behavioral_confidence * 100:7.1f} "
            f"{r.mean_gap * 100:+7.1f} "
            f"{r.mean_absolute_gap * 100:7.1f} "
            f"{r.ece * 100:6.1f} "
            f"{r.brier_acc:8.4f} "
            f"{r.brier_internal:8.4f} "
            f"{r.overconfidence_rate * 100:6.1f}"
        )


# ---- Main ----


def main() -> None:
    results_dir = Path("results/scale_dynamic")
    if not results_dir.exists():
        logger.error(
            f"Results directory {results_dir} not found! Run run_scale_generator.py first."
        )
        return

    # Load all data
    df = load_all_results(results_dir)
    if df.empty:
        logger.error("No results found!")
        return

    # Add empirical difficulty
    df = add_empirical_difficulty(df)

    # Compute reports
    reports = compute_reports(df)

    # Create plot directory
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    logger.info("\n📊 Generating figures...")
    plot_scale_vs_accuracy(df, plot_dir)
    plot_scale_vs_confidence(df, plot_dir)
    plot_scale_vs_gap(df, plot_dir)
    plot_difficulty_conditioned_gap(df, plot_dir)
    plot_category_conditioned_gap(df, plot_dir)

    # Save summary
    save_summary_table(reports, results_dir)

    # Save full DataFrame for further analysis
    df.to_csv(results_dir / "full_results.csv", index=False)
    logger.info(f"\n✅ Analysis complete! Results in {results_dir}")


if __name__ == "__main__":
    main()
