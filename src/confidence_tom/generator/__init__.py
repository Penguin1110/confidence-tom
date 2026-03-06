"""Generator module - 第一階段：小模型產生數據.

This module provides:
- SubjectGenerator: Runs the subject model K times to compute behavioral confidence (C_beh)
- ConfidenceJustification: Structured explanation for confidence scores
- SolvedInstance: A single generation sample
- SubjectOutputV2: Aggregate output with C_beh computation
"""

from confidence_tom.generator.generator import (
    SubjectGenerator,
    SubjectOutput,
    SubjectOutputWithJustification,
)
from confidence_tom.generator.models import (
    ConfidenceJustification,
    SolvedInstance,
    SubjectOutputV2,
)

__all__ = [
    "SubjectGenerator",
    "SubjectOutput",
    "SubjectOutputWithJustification",
    "ConfidenceJustification",
    "SolvedInstance",
    "SubjectOutputV2",
]
