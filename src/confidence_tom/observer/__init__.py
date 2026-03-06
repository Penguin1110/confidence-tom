"""Observer module - 第二階段：大模型預測信心.

This module provides two experimental observer groups:
- Group A: IntuitionObserver (直覺監考 - 純 ToM)
- Group D: SystematicObserver (系統化監考 - P2+ with Trap Declaration)

Both groups do NOT see the correct answer (blind evaluation).
"""

from confidence_tom.observer.models import (
    CoTDiagnosis,
    EnhancedJudgmentOutput,
    JudgmentOutput,
    RecursiveLevelResult,
    TrapDeclaration,
)
from confidence_tom.observer.observer import (
    IntuitionObserver,
    SystematicObserver,
    RecursiveObserver,
    create_observer,
    select_random_sample_cot,
)
from confidence_tom.observer.protocols import (
    build_blind_observer_context,
    build_systematic_observer_context,
    build_protocol_context,
)

__all__ = [
    # Observer Classes
    "BlindObserver",
    "InformedObserver",
    "FrameAwareObserver",
    "RecursiveObserver",
    "create_observer",
    "select_random_sample_cot",
    # Models
    "JudgmentOutput",
    "EnhancedJudgmentOutput",
    "TrapDeclaration",
    "CoTDiagnosis",
    "RecursiveLevelResult",
    # Protocol Builders
    "build_blind_observer_context",
    "build_informed_observer_context",
    "build_frame_aware_observer_context",
    "build_protocol_context",
]
