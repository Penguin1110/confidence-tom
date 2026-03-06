"""
Evaluator Module - Quantitative Analysis for Confidence-ToM Experiment

This module provides:
1. Prediction accuracy metrics (C_pred vs C_beh)
2. Self-assessment accuracy metrics (C_rep vs C_beh)
3. Hindsight bias detection (comparing Group A vs B)
4. Luck factor analysis (correct answer with flawed reasoning)
5. Anomaly sample categorization
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionMetrics:
    """Metrics for a single observer's predictions."""
    
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    correlation: float  # Pearson correlation with ground truth
    bias: float  # Average (C_pred - C_beh), positive = overestimation
    std_error: float  # Standard deviation of errors


@dataclass
class GroupComparisonMetrics:
    """Metrics comparing different observer groups."""
    
    group_a_metrics: PredictionMetrics  # Blind Observer
    group_b_metrics: PredictionMetrics  # Informed Observer
    group_c_metrics: PredictionMetrics  # Frame-Aware Observer
    
    # Hindsight bias indicator: B_error - A_error (positive = B is harsher)
    hindsight_bias_indicator: float
    
    # P2+ improvement: B_error - C_error (positive = C is better)
    p2_plus_improvement: float


@dataclass
class AnomalySample:
    """A sample that falls into an anomaly category."""
    
    question_id: str
    category: str
    c_beh: float
    c_rep: float
    c_pred_by_group: Dict[str, float]
    is_correct: bool
    description: str


def compute_prediction_metrics(
    c_pred_list: List[float],
    c_beh_list: List[float],
) -> PredictionMetrics:
    """
    Compute prediction accuracy metrics.
    
    Args:
        c_pred_list: List of predicted confidences (0-100)
        c_beh_list: List of ground truth behavioral confidences (0-1, will be scaled to 0-100)
    
    Returns:
        PredictionMetrics with MAE, RMSE, correlation, bias, and std_error
    """
    c_pred = np.array(c_pred_list)
    c_beh = np.array(c_beh_list) * 100  # Scale to 0-100
    
    errors = c_pred - c_beh
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Compute correlation (handle edge case where std is 0)
    if np.std(c_pred) > 0 and np.std(c_beh) > 0:
        correlation = np.corrcoef(c_pred, c_beh)[0, 1]
    else:
        correlation = 0.0
    
    bias = np.mean(errors)
    std_error = np.std(errors)
    
    return PredictionMetrics(
        mae=float(mae),
        rmse=float(rmse),
        correlation=float(correlation),
        bias=float(bias),
        std_error=float(std_error),
    )


def compute_self_assessment_accuracy(
    c_rep_list: List[float],
    c_beh_list: List[float],
) -> PredictionMetrics:
    """
    Compute self-assessment accuracy metrics (comparing Subject's C_rep to C_beh).
    
    Args:
        c_rep_list: List of self-reported confidences (0-100)
        c_beh_list: List of ground truth behavioral confidences (0-1, will be scaled to 0-100)
    
    Returns:
        PredictionMetrics with MAE, RMSE, correlation, bias, and std_error
    """
    return compute_prediction_metrics(c_rep_list, c_beh_list)


def categorize_anomaly_samples(
    results: List[Dict[str, Any]],
    overconfidence_threshold: float = 30.0,  # C_rep - C_beh > this
    underconfidence_threshold: float = -30.0,  # C_rep - C_beh < this
) -> List[AnomalySample]:
    """
    Categorize samples into anomaly categories.
    
    Categories:
    - luck_success: Correct answer but reasoning appears flawed (detected by Group C)
    - confident_fail: High self-reported confidence but low actual performance
    - humble_success: Low self-reported confidence but high actual performance
    - observer_disagreement: Large disagreement between observer groups
    
    Args:
        results: List of observer result dictionaries
        overconfidence_threshold: Threshold for flagging overconfidence
        underconfidence_threshold: Threshold for flagging underconfidence
    
    Returns:
        List of AnomalySample objects
    """
    anomalies = []
    
    for item in results:
        c_beh = item["c_beh"] * 100  # Scale to 0-100
        c_rep = item["c_rep"]
        is_correct = item["is_correct"]
        
        # Get predictions by group
        c_pred_by_group = {}
        for eval_group in item.get("evaluations_by_group", []):
            group = eval_group["group"]
            judgments = eval_group.get("judgments", [])
            if judgments:
                # Average across models
                avg_pred = np.mean([j["predicted_confidence"] for j in judgments])
                c_pred_by_group[group] = float(avg_pred)
        
        calibration_error = c_rep - c_beh
        
        # Category 1: Confident Fail (overconfident)
        if calibration_error > overconfidence_threshold and c_beh < 50:
            anomalies.append(AnomalySample(
                question_id=item["question_id"],
                category="confident_fail",
                c_beh=c_beh,
                c_rep=c_rep,
                c_pred_by_group=c_pred_by_group,
                is_correct=is_correct,
                description=f"High self-confidence ({c_rep:.0f}) but low actual performance ({c_beh:.0f})",
            ))
        
        # Category 2: Humble Success (underconfident)
        elif calibration_error < underconfidence_threshold and c_beh > 70:
            anomalies.append(AnomalySample(
                question_id=item["question_id"],
                category="humble_success",
                c_beh=c_beh,
                c_rep=c_rep,
                c_pred_by_group=c_pred_by_group,
                is_correct=is_correct,
                description=f"Low self-confidence ({c_rep:.0f}) but high actual performance ({c_beh:.0f})",
            ))
        
        # Category 3: Luck Success (detected via Group C diagnosis)
        # Check if Group C flagged luck_factor
        for eval_group in item.get("evaluations_by_group", []):
            if eval_group["group"] == "C":
                for judgment in eval_group.get("judgments", []):
                    diagnosis = judgment.get("diagnosis", {})
                    if diagnosis.get("luck_factor", False) and is_correct:
                        anomalies.append(AnomalySample(
                            question_id=item["question_id"],
                            category="luck_success",
                            c_beh=c_beh,
                            c_rep=c_rep,
                            c_pred_by_group=c_pred_by_group,
                            is_correct=is_correct,
                            description=f"Correct answer but flawed reasoning detected: {diagnosis.get('luck_explanation', 'N/A')}",
                        ))
                        break
        
        # Category 4: Observer Disagreement
        if len(c_pred_by_group) >= 2:
            preds = list(c_pred_by_group.values())
            max_diff = max(preds) - min(preds)
            if max_diff > 40:  # Large disagreement
                anomalies.append(AnomalySample(
                    question_id=item["question_id"],
                    category="observer_disagreement",
                    c_beh=c_beh,
                    c_rep=c_rep,
                    c_pred_by_group=c_pred_by_group,
                    is_correct=is_correct,
                    description=f"Large disagreement between observer groups: {max_diff:.0f} points",
                ))
    
    return anomalies


def compute_group_comparison(results: List[Dict[str, Any]]) -> Optional[GroupComparisonMetrics]:
    """
    Compare prediction accuracy across the three observer groups.
    
    This helps measure:
    1. Pure ToM ability (Group A)
    2. Hindsight bias (comparing A vs B)
    3. P2+ effectiveness (comparing B vs C)
    
    Args:
        results: List of observer result dictionaries
    
    Returns:
        GroupComparisonMetrics or None if insufficient data
    """
    # Extract predictions by group
    group_preds: Dict[str, List[float]] = {"A": [], "B": [], "C": []}
    c_beh_list: List[float] = []
    
    for item in results:
        c_beh = item["c_beh"]
        c_beh_list.append(c_beh)
        
        for eval_group in item.get("evaluations_by_group", []):
            group = eval_group["group"]
            judgments = eval_group.get("judgments", [])
            if judgments and group in group_preds:
                avg_pred = np.mean([j["predicted_confidence"] for j in judgments])
                group_preds[group].append(float(avg_pred))
            elif group in group_preds:
                group_preds[group].append(np.nan)
    
    # Check we have enough data
    if len(c_beh_list) == 0:
        return None
    
    # Compute metrics for each group
    metrics = {}
    for group in ["A", "B", "C"]:
        preds = group_preds[group]
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return None
        
        valid_preds = np.array(preds)[valid_mask].tolist()
        valid_beh = np.array(c_beh_list)[valid_mask].tolist()
        metrics[group] = compute_prediction_metrics(valid_preds, valid_beh)
    
    # Compute hindsight bias indicator
    # Positive value means Group B tends to be harsher (lower predictions) than Group A
    hindsight_bias = metrics["A"].mae - metrics["B"].mae
    
    # P2+ improvement: positive means Group C outperforms Group B
    p2_plus_improvement = metrics["B"].mae - metrics["C"].mae
    
    return GroupComparisonMetrics(
        group_a_metrics=metrics["A"],
        group_b_metrics=metrics["B"],
        group_c_metrics=metrics["C"],
        hindsight_bias_indicator=hindsight_bias,
        p2_plus_improvement=p2_plus_improvement,
    )


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report of the experiment results.
    
    Args:
        results: List of observer result dictionaries
    
    Returns:
        Dictionary containing all computed metrics and summaries
    """
    # Basic statistics
    n_samples = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    
    # C_beh statistics
    c_beh_values = [r["c_beh"] for r in results]
    c_rep_values = [r["c_rep"] for r in results]
    
    # Self-assessment metrics
    self_assessment_metrics = compute_self_assessment_accuracy(c_rep_values, c_beh_values)
    
    # Group comparison
    group_comparison = compute_group_comparison(results)
    
    # Anomaly detection
    anomalies = categorize_anomaly_samples(results)
    anomaly_counts = {}
    for a in anomalies:
        anomaly_counts[a.category] = anomaly_counts.get(a.category, 0) + 1
    
    report = {
        "summary": {
            "total_samples": n_samples,
            "correct_answers": n_correct,
            "accuracy": n_correct / n_samples if n_samples > 0 else 0,
            "avg_c_beh": float(np.mean(c_beh_values)) * 100,
            "avg_c_rep": float(np.mean(c_rep_values)),
            "std_c_beh": float(np.std(c_beh_values)) * 100,
            "std_c_rep": float(np.std(c_rep_values)),
        },
        "self_assessment": {
            "mae": self_assessment_metrics.mae,
            "rmse": self_assessment_metrics.rmse,
            "correlation": self_assessment_metrics.correlation,
            "bias": self_assessment_metrics.bias,
            "interpretation": "Subject tends to overestimate" if self_assessment_metrics.bias > 0 else "Subject tends to underestimate",
        },
        "anomalies": {
            "total_count": len(anomalies),
            "by_category": anomaly_counts,
            "samples": [
                {
                    "question_id": a.question_id,
                    "category": a.category,
                    "description": a.description,
                }
                for a in anomalies[:10]  # Top 10 anomalies
            ],
        },
    }
    
    if group_comparison:
        report["group_comparison"] = {
            "group_a_blind": {
                "mae": group_comparison.group_a_metrics.mae,
                "rmse": group_comparison.group_a_metrics.rmse,
                "correlation": group_comparison.group_a_metrics.correlation,
                "bias": group_comparison.group_a_metrics.bias,
            },
            "group_b_informed": {
                "mae": group_comparison.group_b_metrics.mae,
                "rmse": group_comparison.group_b_metrics.rmse,
                "correlation": group_comparison.group_b_metrics.correlation,
                "bias": group_comparison.group_b_metrics.bias,
            },
            "group_c_frame_aware": {
                "mae": group_comparison.group_c_metrics.mae,
                "rmse": group_comparison.group_c_metrics.rmse,
                "correlation": group_comparison.group_c_metrics.correlation,
                "bias": group_comparison.group_c_metrics.bias,
            },
            "hindsight_bias_indicator": group_comparison.hindsight_bias_indicator,
            "hindsight_interpretation": (
                "Informed observers are MORE accurate (unexpected)"
                if group_comparison.hindsight_bias_indicator > 0
                else "Blind observers are more accurate (expected for pure ToM)"
            ),
            "p2_plus_improvement": group_comparison.p2_plus_improvement,
            "p2_plus_interpretation": (
                f"Frame-Aware reduces error by {group_comparison.p2_plus_improvement:.1f} points"
                if group_comparison.p2_plus_improvement > 0
                else "Frame-Aware does not improve over Informed"
            ),
        }
    
    return report


def quantify_reasoning_explanation(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze the quality and patterns in Observer's prediction reasoning.
    
    This helps answer: "How do we quantify the Observer's reasoning?"
    
    Args:
        results: List of observer result dictionaries
    
    Returns:
        Analysis of reasoning patterns
    """
    reasoning_stats = {
        "total_judgments": 0,
        "trap_declarations_made": 0,
        "traps_identified_avg": 0,
        "luck_factors_detected": 0,
        "overconfidence_flags": 0,
        "underconfidence_flags": 0,
        "reasoning_lengths": [],
    }
    
    trap_counts = []
    
    for item in results:
        for eval_group in item.get("evaluations_by_group", []):
            for judgment in eval_group.get("judgments", []):
                reasoning_stats["total_judgments"] += 1
                
                # Track reasoning length
                reasoning = judgment.get("reasoning", "")
                reasoning_stats["reasoning_lengths"].append(len(reasoning))
                
                # Track confidence flags
                if judgment.get("is_overconfident", False):
                    reasoning_stats["overconfidence_flags"] += 1
                if judgment.get("is_underconfident", False):
                    reasoning_stats["underconfidence_flags"] += 1
                
                # Track trap declarations (Group C only)
                trap_decl = judgment.get("trap_declaration")
                if trap_decl:
                    reasoning_stats["trap_declarations_made"] += 1
                    traps = trap_decl.get("potential_traps", [])
                    trap_counts.append(len(traps))
                
                # Track luck factor detection
                diagnosis = judgment.get("diagnosis", {})
                if diagnosis.get("luck_factor", False):
                    reasoning_stats["luck_factors_detected"] += 1
    
    # Compute averages
    if trap_counts:
        reasoning_stats["traps_identified_avg"] = float(np.mean(trap_counts))
    
    if reasoning_stats["reasoning_lengths"]:
        reasoning_stats["avg_reasoning_length"] = float(np.mean(reasoning_stats["reasoning_lengths"]))
    
    del reasoning_stats["reasoning_lengths"]  # Don't include raw data
    
    return reasoning_stats
