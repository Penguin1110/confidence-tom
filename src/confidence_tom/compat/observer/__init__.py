"""Legacy observer implementation for the old ToM confidence pipeline."""

from .dynamic_models import ErrorType as DynamicErrorType
from .dynamic_models import JudgmentOutput as DynamicJudgmentOutput
from .models import (
    CanonicalizedSubjectOutput,
    CoTDiagnosis,
    EnhancedJudgmentOutput,
    ErrorType,
    JudgmentOutput,
    LegacyJudgmentOutput,
    ObserverFrameCheckSelfSolve,
    ObserverSelfSolve,
    RecursiveLevelResult,
    TrapDeclaration,
)
from .observer import RecursiveObserver
from .protocols import (
    build_blind_observer_context,
    build_frame_aware_observer_context,
    build_informed_observer_context,
    build_protocol_context,
)

__all__ = [
    "CanonicalizedSubjectOutput",
    "CoTDiagnosis",
    "DynamicErrorType",
    "DynamicJudgmentOutput",
    "EnhancedJudgmentOutput",
    "ErrorType",
    "JudgmentOutput",
    "LegacyJudgmentOutput",
    "ObserverFrameCheckSelfSolve",
    "ObserverSelfSolve",
    "RecursiveLevelResult",
    "RecursiveObserver",
    "TrapDeclaration",
    "build_blind_observer_context",
    "build_frame_aware_observer_context",
    "build_informed_observer_context",
    "build_protocol_context",
]
