"""
Analyze Results - Theory of Mind Confidence Prediction Experiment

This script generates comprehensive analysis including:
1. Prediction accuracy by observer group (A/B/C)
2. Hindsight bias measurement
3. Self-assessment calibration
4. Anomaly sample analysis
5. Visualization plots
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from confidence_tom.evaluator import (
    categorize_anomaly_samples,
    compute_group_comparison,
    generate_summary_report,
    quantify_reasoning_explanation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_v3(file_path: Path) -> List[Dict[str, Any]]:
    """Load observer v3 results."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a flat DataFrame for analysis."""
    rows = []
    
    for item in results:
        base_row = {
            "question_id": item["question_id"],
            "task_type": item.get("task_type", "unknown"),
            "ambiguity_level": item.get("ambiguity_level", "unknown"),
            "c_beh": item["c_beh"] * 100,  # Scale to 0-100
            "c_rep": item["c_rep"],
            "is_correct": item.get("is_correct", False),
            "correct_count": item.get("correct_count", 0),
            "k_samples": item.get("k_samples", 10),
            "consistency_rate": item.get("consistency_rate", 0) * 100,
        }
        
        for eval_group in item.get("evaluations_by_group", []):
            group = eval_group["group"]
            group_name = eval_group.get("group_name", group)
            
            for judgment in eval_group.get("judgments", []):
                row = base_row.copy()
                row["group"] = group
                row["group_name"] = group_name
                row["observer_model"] = judgment.get("observer_model", "unknown")
                row["c_pred"] = judgment["predicted_confidence"]
                row["is_overconfident"] = judgment.get("is_overconfident", False)
                row["error"] = judgment["predicted_confidence"] - base_row["c_beh"]
                row["abs_error"] = abs(row["error"])
                
                # Diagnosis info (Group C only)
                diagnosis = judgment.get("diagnosis", {})
                row["luck_factor"] = diagnosis.get("luck_factor", False)
                row["fell_into_trap"] = diagnosis.get("fell_into_trap", False)
                row["reasoning_quality"] = diagnosis.get("reasoning_quality", "N/A")
                
                rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_prediction_accuracy_by_group(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot of prediction errors by observer group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Error distribution
    ax1 = axes[0]
    sns.boxplot(x="group", y="error", data=df, ax=ax1, palette="Set2")
    ax1.axhline(y=0, color="r", linestyle="--", label="Perfect Prediction")
    ax1.set_xlabel("Observer Group")
    ax1.set_ylabel("Prediction Error (C_pred - C_beh)")
    ax1.set_title("Prediction Error by Observer Group")
    ax1.legend()
    
    # Plot 2: Absolute error
    ax2 = axes[1]
    group_mae = df.groupby("group")["abs_error"].mean().reset_index()
    sns.barplot(x="group", y="abs_error", data=group_mae, ax=ax2, palette="Set2")
    ax2.set_xlabel("Observer Group")
    ax2.set_ylabel("Mean Absolute Error (MAE)")
    ax2.set_title("Prediction Accuracy by Observer Group")
    
    # Add reference line for self-assessment MAE
    self_mae = (df["c_rep"] - df["c_beh"]).abs().mean()
    ax2.axhline(y=self_mae, color="orange", linestyle="--", 
                label=f"Subject Self-Assessment MAE: {self_mae:.1f}")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_accuracy_by_group.png", dpi=150)
    plt.close()


def plot_hindsight_bias_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze hindsight bias by comparing Group A (Blind) vs Group B (Informed)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get predictions for correct vs incorrect samples
    df_a = df[df["group"] == "A"]
    df_b = df[df["group"] == "B"]
    
    # Plot 1: Scatter plot of A vs B predictions
    ax1 = axes[0]
    merged = df_a[["question_id", "c_pred"]].merge(
        df_b[["question_id", "c_pred"]], 
        on="question_id", 
        suffixes=("_A", "_B")
    )
    ax1.scatter(merged["c_pred_A"], merged["c_pred_B"], alpha=0.5, s=30)
    ax1.plot([0, 100], [0, 100], "r--", label="y=x (no bias)")
    ax1.set_xlabel("Group A (Blind) Prediction")
    ax1.set_ylabel("Group B (Informed) Prediction")
    ax1.set_title("Hindsight Bias: Blind vs Informed Observer")
    ax1.legend()
    
    # Calculate bias
    bias = (merged["c_pred_B"] - merged["c_pred_A"]).mean()
    ax1.text(0.05, 0.95, f"Avg Bias (B-A): {bias:+.1f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat"))
    
    # Plot 2: Error by correctness
    ax2 = axes[1]
    for_plot = df[df["group"].isin(["A", "B"])].copy()
    for_plot["correct_label"] = for_plot["is_correct"].map({True: "Correct", False: "Incorrect"})
    sns.boxplot(x="group", y="error", hue="correct_label", data=for_plot, ax=ax2, palette="Set1")
    ax2.axhline(y=0, color="gray", linestyle="--")
    ax2.set_xlabel("Observer Group")
    ax2.set_ylabel("Prediction Error")
    ax2.set_title("Prediction Error: Correct vs Incorrect Samples")
    ax2.legend(title="Subject Answer")
    
    plt.tight_layout()
    plt.savefig(output_dir / "hindsight_bias_analysis.png", dpi=150)
    plt.close()


def plot_calibration_curve(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot calibration curves for each observer group."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bins for C_beh
    bins = [0, 20, 40, 60, 80, 100]
    df["c_beh_bin"] = pd.cut(df["c_beh"], bins=bins, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    
    colors = {"A": "blue", "B": "orange", "C": "green"}
    markers = {"A": "o", "B": "s", "C": "^"}
    
    for group in ["A", "B", "C"]:
        group_df = df[df["group"] == group]
        calibration = group_df.groupby("c_beh_bin").agg({
            "c_pred": "mean",
            "c_beh": "mean"
        }).reset_index()
        
        ax.plot(calibration["c_beh"], calibration["c_pred"], 
                marker=markers[group], label=f"Group {group}", 
                color=colors[group], linewidth=2, markersize=8)
    
    # Perfect calibration line
    ax.plot([0, 100], [0, 100], "k--", label="Perfect Calibration")
    
    # Subject self-assessment
    self_cal = df.groupby("c_beh_bin").agg({"c_rep": "mean", "c_beh": "mean"}).reset_index()
    ax.plot(self_cal["c_beh"], self_cal["c_rep"], 
            marker="x", label="Subject Self-Assessment", 
            color="red", linewidth=2, markersize=8, linestyle=":")
    
    ax.set_xlabel("True Behavioral Confidence (C_beh)")
    ax.set_ylabel("Predicted Confidence")
    ax.set_title("Calibration Curves: Observer Predictions vs Ground Truth")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=150)
    plt.close()


def plot_p2_plus_effectiveness(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare Group B vs Group C to measure P2+ (Trap Declaration) effectiveness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter to B and C only
    df_bc = df[df["group"].isin(["B", "C"])]
    
    # Plot 1: MAE comparison by task type
    ax1 = axes[0]
    mae_by_task = df_bc.groupby(["group", "task_type"])["abs_error"].mean().unstack("group")
    mae_by_task.plot(kind="bar", ax=ax1, color=["orange", "green"])
    ax1.set_xlabel("Task Type")
    ax1.set_ylabel("Mean Absolute Error")
    ax1.set_title("P2+ Effectiveness by Task Type")
    ax1.legend(["Group B (Informed)", "Group C (Frame-Aware)"])
    ax1.tick_params(axis="x", rotation=45)
    
    # Plot 2: Trap detection effectiveness
    ax2 = axes[1]
    group_c = df[df["group"] == "C"]
    
    # Compare error when trap detected vs not
    trap_analysis = group_c.groupby("fell_into_trap")["abs_error"].mean()
    if len(trap_analysis) > 0:
        trap_analysis.plot(kind="bar", ax=ax2, color=["lightgreen", "salmon"])
        ax2.set_xlabel("Observer Detected Trap in Subject's Reasoning")
        ax2.set_ylabel("Mean Absolute Error of Prediction")
        ax2.set_title("Prediction Accuracy When Trap Detected")
        ax2.set_xticklabels(["No Trap Detected", "Trap Detected"], rotation=0)
    else:
        ax2.text(0.5, 0.5, "No trap data available", 
                ha="center", va="center", transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "p2_plus_effectiveness.png", dpi=150)
    plt.close()


def plot_luck_factor_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze 'luck factor' cases where correct answer came from flawed reasoning."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to Group C (has luck factor detection)
    group_c = df[df["group"] == "C"]
    
    # Create categories
    group_c = group_c.copy()
    group_c["category"] = "Normal"
    group_c.loc[(group_c["is_correct"]) & (group_c["luck_factor"]), "category"] = "Lucky (Correct + Flawed)"
    group_c.loc[(~group_c["is_correct"]) & (group_c["luck_factor"]), "category"] = "Unlucky (Wrong + Flawed)"
    
    # Plot error distribution by category
    sns.boxplot(x="category", y="error", data=group_c, palette="Set3", ax=ax)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Sample Category")
    ax.set_ylabel("Prediction Error (C_pred - C_beh)")
    ax.set_title("Prediction Error by Luck Factor Detection")
    
    # Add counts
    counts = group_c["category"].value_counts()
    ax.text(0.02, 0.98, f"Counts: {dict(counts)}", 
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"))
    
    plt.tight_layout()
    plt.savefig(output_dir / "luck_factor_analysis.png", dpi=150)
    plt.close()


# ============================================================================
# Main Analysis Function
# ============================================================================

def main() -> None:
    # Try to find results file
    results_file = Path("results/observer_v3_results.json")
    if not results_file.exists():
        results_file = Path("results/observer_v2_recursive_results.json")
    
    if not results_file.exists():
        logger.error(f"Cannot find results file. Run observer experiment first.")
        return

    logger.info(f"Loading results from {results_file}")
    results = load_data_v3(results_file)
    logger.info(f"Loaded {len(results)} evaluated questions")

    # Generate summary report
    logger.info("Generating summary report...")
    report = generate_summary_report(results)
    
    # Save report
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print key findings
    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 60)
    
    summary = report["summary"]
    logger.info(f"Total Samples: {summary['total_samples']}")
    logger.info(f"Subject Accuracy: {summary['accuracy']:.1%}")
    logger.info(f"Average C_beh: {summary['avg_c_beh']:.1f}")
    logger.info(f"Average C_rep: {summary['avg_c_rep']:.1f}")
    
    logger.info("-" * 60)
    logger.info("SELF-ASSESSMENT CALIBRATION")
    self_assess = report["self_assessment"]
    logger.info(f"MAE: {self_assess['mae']:.1f}")
    logger.info(f"Bias: {self_assess['bias']:+.1f} ({self_assess['interpretation']})")
    logger.info(f"Correlation: {self_assess['correlation']:.3f}")
    
    if "group_comparison" in report:
        logger.info("-" * 60)
        logger.info("GROUP COMPARISON")
        gc = report["group_comparison"]
        
        logger.info(f"Group A (Blind) MAE: {gc['group_a_blind']['mae']:.1f}")
        logger.info(f"Group B (Informed) MAE: {gc['group_b_informed']['mae']:.1f}")
        logger.info(f"Group C (Frame-Aware) MAE: {gc['group_c_frame_aware']['mae']:.1f}")
        
        logger.info("-" * 60)
        logger.info(f"Hindsight Bias: {gc['hindsight_bias_indicator']:+.1f}")
        logger.info(f"  {gc['hindsight_interpretation']}")
        logger.info(f"P2+ Improvement: {gc['p2_plus_improvement']:+.1f}")
        logger.info(f"  {gc['p2_plus_interpretation']}")
    
    logger.info("-" * 60)
    logger.info("ANOMALY DETECTION")
    anomalies = report["anomalies"]
    logger.info(f"Total Anomalies: {anomalies['total_count']}")
    for cat, count in anomalies["by_category"].items():
        logger.info(f"  {cat}: {count}")
    
    # Generate plots
    logger.info("-" * 60)
    logger.info("Generating visualizations...")
    
    df = prepare_dataframe(results)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if len(df) > 0:
        plot_prediction_accuracy_by_group(df, plots_dir)
        plot_hindsight_bias_analysis(df, plots_dir)
        plot_calibration_curve(df, plots_dir)
        plot_p2_plus_effectiveness(df, plots_dir)
        plot_luck_factor_analysis(df, plots_dir)
        logger.info(f"Plots saved to {plots_dir}")
    else:
        logger.warning("No data available for plotting")

    # Quantify reasoning
    reasoning_stats = quantify_reasoning_explanation(results)
    logger.info("-" * 60)
    logger.info("REASONING ANALYSIS")
    logger.info(f"Total Judgments: {reasoning_stats['total_judgments']}")
    logger.info(f"Trap Declarations: {reasoning_stats['trap_declarations_made']}")
    logger.info(f"Avg Traps Identified: {reasoning_stats.get('traps_identified_avg', 0):.1f}")
    logger.info(f"Luck Factors Detected: {reasoning_stats['luck_factors_detected']}")
    logger.info(f"Overconfidence Flags: {reasoning_stats['overconfidence_flags']}")
    
    logger.info("=" * 60)
    logger.info(f"Analysis complete. Report saved to {output_dir / 'analysis_report.json'}")


if __name__ == "__main__":
    main()
