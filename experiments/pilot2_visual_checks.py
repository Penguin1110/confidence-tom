import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

SCALE_ORDER = ["Gemma-3-4B", "Gemma-3-12B", "Gemma-3-27B"]
SCALE_COLORS = {"Gemma-3-4B": "#f97583", "Gemma-3-12B": "#79c0ff", "Gemma-3-27B": "#56d364"}


def load_data(results_dir: Path):
    data_by_model = {}
    for label in SCALE_ORDER:
        file_path = results_dir / f"{label}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data_by_model[label] = json.load(f)
    return data_by_model


def plot_confidence_distribution(data_by_model, output_dir: Path):
    fig, axes = plt.subplots(1, len(data_by_model), figsize=(15, 5), sharey=True)
    if len(data_by_model) == 1:
        axes = [axes]

    for ax, label in zip(axes, SCALE_ORDER):
        if label not in data_by_model:
            continue

        data = data_by_model[label]
        # Collect all individual sample confidences
        all_confs = []
        for d in data:
            all_confs.extend(d["sample_confidences"])

        # Multiply by 100 for display
        all_confs = np.array(all_confs) * 100

        bins = np.arange(0, 105, 5)
        ax.hist(all_confs, bins=bins, color=SCALE_COLORS[label], edgecolor="#30363d", alpha=0.8)
        ax.set_title(f"{label}\nC_rep Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Confidence (%)")

        if ax == axes[0]:
            ax.set_ylabel("Frequency (Samples)")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / "check1_confidence_dist.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved check1_confidence_dist.png")


def plot_reliability_diagram(data_by_model, output_dir: Path):
    fig, axes = plt.subplots(1, len(data_by_model), figsize=(15, 5), sharey=True, sharex=True)
    if len(data_by_model) == 1:
        axes = [axes]

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for ax, label in zip(axes, SCALE_ORDER):
        if label not in data_by_model:
            continue

        data = data_by_model[label]
        c_reps = np.array([d["c_rep"] for d in data])
        accs = np.array([float(d["is_correct"]) for d in data])

        bin_accs = []
        bin_confs = []

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            if i == n_bins - 1:
                mask = (c_reps >= lo) & (c_reps <= hi)
            else:
                mask = (c_reps >= lo) & (c_reps < hi)

            if mask.sum() > 0:
                bin_accs.append(accs[mask].mean() * 100)
                bin_confs.append(c_reps[mask].mean() * 100)

        # Perfect calibration line
        ax.plot([0, 100], [0, 100], color="#f0883e", linestyle="--", label="Perfect Calibration")

        # Actual calibration
        ax.plot(
            bin_confs, bin_accs, marker="o", color=SCALE_COLORS[label], linewidth=2, markersize=8
        )

        ax.set_title(f"{label}\nReliability Diagram", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Confidence (%)")
        if ax == axes[0]:
            ax.set_ylabel("Empirical Accuracy (%)")

        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "check2_reliability.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved check2_reliability.png")


def plot_answer_entropy(data_by_model, output_dir: Path):
    fig, axes = plt.subplots(1, len(data_by_model), figsize=(15, 5), sharey=True)
    if len(data_by_model) == 1:
        axes = [axes]

    for ax, label in zip(axes, SCALE_ORDER):
        if label not in data_by_model:
            continue

        data = data_by_model[label]
        entropies = []

        for d in data:
            dist = d["answer_distribution"]
            total = sum(dist.values())
            probs = [v / total for v in dist.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropies.append(entropy)

        ax.hist(entropies, bins=15, color=SCALE_COLORS[label], edgecolor="#30363d", alpha=0.8)
        ax.set_title(f"{label}\nAnswer Entropy (C_beh variance)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Entropy (bits)\n(0 = all 10 same answer)")

        if ax == axes[0]:
            ax.set_ylabel("Frequency (Questions)")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "check3_answer_entropy.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved check3_answer_entropy.png")


def main():
    results_dir = Path("results/scale")
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(results_dir)

    logger.info("Generating Check 1...")
    plot_confidence_distribution(data, output_dir)

    logger.info("Generating Check 2...")
    plot_reliability_diagram(data, output_dir)

    logger.info("Generating Check 3...")
    plot_answer_entropy(data, output_dir)


if __name__ == "__main__":
    main()
