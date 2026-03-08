"""Calibration and miscalibration metrics for the scale experiment.

ALL metrics operate in 0-1 scale internally.
Multiply by 100 only when displaying to humans (plots/tables).

Terminology:
- C_rep: Reported (verbalized) confidence — model's self-assessment [0-1]
- C_beh: Behavioral confidence — K-sample majority fraction [0-1]
- Gap: C_rep - C_beh — positive means overconfident
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---- Per-question metrics ----


def miscalibration_gap(c_rep: float, c_beh: float) -> float:
    """Signed miscalibration gap.

    Gap = C_rep - C_beh (both 0-1).
    Positive → overconfident, Negative → underconfident.
    """
    return c_rep - c_beh


def absolute_gap(c_rep: float, c_beh: float) -> float:
    """Direction-agnostic miscalibration magnitude."""
    return abs(c_rep - c_beh)


def brier_score_question(c_rep: float, c_beh: float) -> float:
    """Internal consistency Brier: (C̄_rep - C_beh)²."""
    return (c_rep - c_beh) ** 2


def brier_score_accuracy(c_rep: float, is_correct: float) -> float:
    """Standard Brier score: (C_rep - is_correct)².

    This is the standard calibration metric from the literature.
    is_correct is 0 or 1.
    """
    return (c_rep - is_correct) ** 2


# ---- Aggregate metrics ----


def expected_calibration_error(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) with M equal-width bins.

    ECE = Σ_{m=1}^{M} (|B_m|/n) × |acc(B_m) - conf(B_m)|

    Args:
        confidences: Array of confidence values in [0, 1].
        accuracies: Array of binary correctness values (0 or 1).
        n_bins: Number of bins (default: 10).

    Returns:
        ECE value in [0, 1].
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]

        # Include right endpoint for the last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def mean_gap(gaps: list[float] | NDArray[np.floating]) -> float:
    """Mean signed miscalibration gap across questions."""
    arr = np.asarray(gaps)
    return float(arr.mean()) if len(arr) > 0 else 0.0


def mean_absolute_gap(gaps: list[float] | NDArray[np.floating]) -> float:
    """Mean absolute miscalibration gap (|Gap|) across questions."""
    arr = np.asarray(gaps)
    return float(np.abs(arr).mean()) if len(arr) > 0 else 0.0


def mean_brier_score(brier_scores: list[float] | NDArray[np.floating]) -> float:
    """Mean Brier score across questions."""
    arr = np.asarray(brier_scores)
    return float(arr.mean()) if len(arr) > 0 else 0.0


def overconfidence_rate(gaps: list[float] | NDArray[np.floating]) -> float:
    """Fraction of questions with Gap > 0 (supplementary metric only)."""
    arr = np.asarray(gaps)
    return float((arr > 0).mean()) if len(arr) > 0 else 0.0


# ---- Difficulty stratification ----


def compute_empirical_difficulty(
    accuracies_by_model: dict[str, NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Compute empirical difficulty for each question across all models.

    D(q) = 1 - (1/|M|) Σ_m Acc(q, m)

    Args:
        accuracies_by_model: Dict mapping model_name → array of per-question correctness (0/1).
                             All arrays must have the same length (same questions in same order).

    Returns:
        Array of difficulty values in [0, 1] per question.
        0 = easiest (everyone got it right), 1 = hardest (nobody got it right).
    """
    acc_matrix = np.stack(list(accuracies_by_model.values()), axis=0)  # (n_models, n_questions)
    mean_acc = acc_matrix.mean(axis=0)  # (n_questions,)
    return 1.0 - mean_acc


def stratify_by_difficulty(
    difficulties: NDArray[np.floating],
    thresholds: tuple[float, float] = (0.33, 0.67),
) -> NDArray[np.int_]:
    """Assign difficulty buckets: 0=easy, 1=medium, 2=hard.

    Args:
        difficulties: Array of empirical difficulty values in [0, 1].
        thresholds: (easy_upper, hard_lower) boundaries.

    Returns:
        Array of bucket indices (0, 1, or 2).
    """
    buckets = np.zeros(len(difficulties), dtype=np.int_)
    buckets[(difficulties >= thresholds[0]) & (difficulties < thresholds[1])] = 1
    buckets[difficulties >= thresholds[1]] = 2
    return buckets


DIFFICULTY_LABELS = {0: "easy", 1: "medium", 2: "hard"}


# ---- Summary report ----


@dataclass
class CalibrationReport:
    """Aggregated calibration metrics for a single model."""

    model_name: str
    n_questions: int
    accuracy: float
    mean_reported_confidence: float
    mean_behavioral_confidence: float
    mean_gap: float
    mean_absolute_gap: float
    ece: float
    brier_acc: float  # Standard: (C_rep - is_correct)²
    brier_internal: float  # Internal: (C_rep - C_beh)²
    overconfidence_rate: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_questions": self.n_questions,
            "accuracy": round(self.accuracy, 4),
            "mean_reported_confidence": round(self.mean_reported_confidence, 4),
            "mean_behavioral_confidence": round(self.mean_behavioral_confidence, 4),
            "mean_gap": round(self.mean_gap, 4),
            "mean_absolute_gap": round(self.mean_absolute_gap, 4),
            "ece": round(self.ece, 4),
            "brier_acc": round(self.brier_acc, 4),
            "brier_internal": round(self.brier_internal, 4),
            "overconfidence_rate": round(self.overconfidence_rate, 4),
        }

    def display_str(self) -> str:
        """Human-readable summary with percentages."""
        return (
            f"{'=' * 50}\n"
            f"Model: {self.model_name}\n"
            f"N = {self.n_questions}\n"
            f"{'=' * 50}\n"
            f"  Accuracy:           {self.accuracy * 100:.1f}%\n"
            f"  Mean C_rep:         {self.mean_reported_confidence * 100:.1f}%\n"
            f"  Mean C_beh:         {self.mean_behavioral_confidence * 100:.1f}%\n"
            f"  Mean Gap:           {self.mean_gap * 100:+.1f}%\n"
            f"  Mean |Gap|:         {self.mean_absolute_gap * 100:.1f}%\n"
            f"  ECE:                {self.ece * 100:.1f}%\n"
            f"  Brier (vs acc):     {self.brier_acc:.4f}\n"
            f"  Brier (vs c_beh):   {self.brier_internal:.4f}\n"
            f"  Overconfidence %:   {self.overconfidence_rate * 100:.1f}%\n"
        )


def compute_calibration_report(
    model_name: str,
    c_reps: NDArray[np.floating],
    c_behs: NDArray[np.floating],
    is_correct: NDArray[np.floating],
) -> CalibrationReport:
    """Compute all calibration metrics for a single model.

    Args:
        model_name: Model identifier.
        c_reps: Reported confidence values (0-1) per question.
        c_behs: Behavioral confidence values (0-1) per question.
        is_correct: Binary correctness (0 or 1) per question.

    Returns:
        CalibrationReport with all metrics.
    """
    n = len(c_reps)
    gaps = c_reps - c_behs

    return CalibrationReport(
        model_name=model_name,
        n_questions=n,
        accuracy=float(is_correct.mean()),
        mean_reported_confidence=float(c_reps.mean()),
        mean_behavioral_confidence=float(c_behs.mean()),
        mean_gap=float(gaps.mean()),
        mean_absolute_gap=float(np.abs(gaps).mean()),
        ece=expected_calibration_error(c_reps, is_correct),
        brier_acc=float(((c_reps - is_correct) ** 2).mean()),
        brier_internal=float(((c_reps - c_behs) ** 2).mean()),
        overconfidence_rate=float((gaps > 0).mean()),
    )
