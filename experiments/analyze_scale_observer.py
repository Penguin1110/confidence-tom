"""Aggregates and visualizes Phase 5 (Observer) experimental results.
Computes "Curse of Knowledge" / "Sycophantic Compliance" biases by
quantifying the gap between True State and Observer Estimated State.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Styling ---
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

SUBJECT_ORDER = [
    "Gemma-3-4B",
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Qwen-3.5-27B",
    "Qwen-3.5-35B-A3B",
    "Qwen-3.5-122B-A10B",
    "Qwen-3.5-397B-A17B",
    "Llama-4-Scout",
    "Llama-4-Maverick",
    "GPT-OSS-20B",
    "GPT-OSS-120B",
]
# A cohesive color palette for 11 elements
COLORS = [
    "#f97583",
    "#f97583",
    "#f97583",  # Gemma
    "#79c0ff",
    "#79c0ff",
    "#79c0ff",
    "#79c0ff",  # Qwen
    "#56d364",
    "#56d364",  # Llama
    "#d2a8ff",
    "#d2a8ff",  # GPT
]


def load_observer_results(data_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Returns {observer_model: {subject_model: [records]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for file_path in data_dir.glob("*.json"):
        if file_path.name.endswith(".tmp.json"):
            continue

        parts = file_path.stem.split("_by_")
        if len(parts) != 2:
            continue

        subj, obs = parts

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                data[obs][subj].extend(records)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    return data


def plot_accuracy_blindness(data: dict[str, dict[str, list[dict]]], out_dir: Path):
    """Plots Predicted Accuracy vs True Accuracy."""
    observers = list(data.keys())
    if not observers:
        logger.warning("No observer data found.")
        return

    fig, axes = plt.subplots(1, len(observers), figsize=(5 * len(observers), 5), sharey=True)
    if len(observers) == 1:
        axes = [axes]

    for ax, obs in zip(axes, observers):
        true_accs = []
        pred_accs = []

        for subj, color in zip(SUBJECT_ORDER, COLORS):
            if subj not in data[obs]:
                continue

            records = data[obs][subj]
            t_acc = np.mean([r["truth_is_correct"] for r in records]) * 100
            p_acc = np.mean([r["predicted_correctness"] for r in records]) * 100

            true_accs.append(t_acc)
            pred_accs.append(p_acc)

            ax.plot(
                [subj], [t_acc], marker="o", markersize=10, color=color, label=f"True Acc ({subj})"
            )
            ax.plot(
                [subj],
                [p_acc],
                marker="X",
                markersize=10,
                color=color,
                alpha=0.6,
                label="Observer Predicted Acc",
            )

            # Draw line between them
            ax.plot([subj, subj], [t_acc, p_acc], color=color, linestyle="--", alpha=0.5)

        ax.set_title(f"Observer: {obs}\nAccuracy Blindness", fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")

        ax.grid(axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 100)

    # Deduplicate legend for the first plot
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Simplify legend: we just need True vs Predicted globally
    custom_lines = [
        plt.Line2D([0], [0], color="white", marker="o", linestyle="none", markersize=10),
        plt.Line2D([0], [0], color="white", marker="X", linestyle="none", markersize=10),
    ]
    axes[-1].legend(custom_lines, ["Objective Accuracy", "Observer Estimation"], loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_accuracy_blindness.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved fig5_accuracy_blindness.png")


def plot_confidence_overestimation(data: dict[str, dict[str, list[dict]]], out_dir: Path):
    """Plots True C_rep vs Predicted C_rep to see if Manager gets Fooled by Subject."""
    observers = list(data.keys())
    if not observers:
        return

    fig, axes = plt.subplots(1, len(observers), figsize=(5 * len(observers), 5), sharey=True)
    if len(observers) == 1:
        axes = [axes]

    for ax, obs in zip(axes, observers):
        for subj, color in zip(SUBJECT_ORDER, COLORS):
            if subj not in data[obs]:
                continue

            records = data[obs][subj]
            # Split into correct vs incorrect trials (this is the crux of the curse of knowledge)
            wrong_records = [r for r in records if r["truth_is_correct"] == 0]

            if not wrong_records:
                continue

            true_conf = np.mean([r["truth_c_rep"] for r in wrong_records]) * 100
            pred_conf = np.mean([r["predicted_subject_confidence"] for r in wrong_records]) * 100

            ax.plot([subj], [true_conf], marker="o", markersize=10, color=color)
            ax.plot([subj], [pred_conf], marker="X", markersize=10, color=color, alpha=0.6)
            ax.plot([subj, subj], [true_conf, pred_conf], color=color, linestyle="--", alpha=0.5)

        ax.set_title(f"Observer: {obs}\nConf. Estimation on Wrong Answers", fontsize=13)
        if ax == axes[0]:
            ax.set_ylabel("Confidence (%)")
        ax.grid(axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 100)

    custom_lines = [
        plt.Line2D([0], [0], color="white", marker="o", linestyle="none", markersize=10),
        plt.Line2D([0], [0], color="white", marker="X", linestyle="none", markersize=10),
    ]
    axes[-1].legend(
        custom_lines, ["Subject True C_rep", "Observer Predicted C_rep"], loc="lower right"
    )

    plt.tight_layout()
    plt.savefig(out_dir / "fig6_confidence_sycophancy.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved fig6_confidence_sycophancy.png")


def print_summary(data: dict[str, dict[str, list[dict]]], out_dir: Path):
    summary_data = []

    print("\n" + "=" * 80)
    print("🤖 Phase 5: Observer Bias (Sycophantic Compliance) Summary")
    print("=" * 80)

    for obs, subjs in data.items():
        for subj in SUBJECT_ORDER:
            if subj not in subjs:
                continue
            records = subjs[subj]

            t_acc = np.mean([r["truth_is_correct"] for r in records]) * 100
            p_acc = np.mean([r["predicted_correctness"] for r in records]) * 100
            acc_err = p_acc - t_acc  # Positive means Observer overestimated accuracy

            wrong_records = [r for r in records if r["truth_is_correct"] == 0]
            sycophancy = 0
            if wrong_records:
                # Average predicted correctness ONLY when the subject was actually wrong
                sycophancy = np.mean([r["predicted_correctness"] for r in wrong_records]) * 100

            row = {
                "observer": obs,
                "subject": subj,
                "n_samples": len(records),
                "true_acc": t_acc,
                "pred_acc": p_acc,
                "acc_overestimation": acc_err,
                "sycophancy_on_wrong": sycophancy,
            }
            summary_data.append(row)

            print(f"[{obs}] evaluating [{subj}] (N={len(records)})")
            print(
                f"  True Acc: {t_acc:.1f}%  |  Obs. Pred Acc: {p_acc:.1f}%  |  Acc Bias: {acc_err:+.1f}%"
            )
            print(f"  ⚠️ Sycophancy (Pred Acc when WRONG): {sycophancy:.1f}%")
            print("-" * 60)

    with open(out_dir / "observer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


def main():
    data_dir = Path("results/scale_observer")
    if not data_dir.exists():
        logger.error(f"Directory {data_dir} not found.")
        return

    out_dir = Path("results/scale/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_observer_results(data_dir)
    plot_accuracy_blindness(data, out_dir)
    plot_confidence_overestimation(data, out_dir)
    print_summary(data, out_dir)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
