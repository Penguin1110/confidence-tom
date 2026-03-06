from typing import Optional

from confidence_tom.generator.models import SubjectOutputV2
from confidence_tom.observer.models import (
    CanonicalizedSubjectOutput,
    ObserverFrameCheckSelfSolve,
    ObserverSelfSolve,
    TrapDeclaration,
)


# ============================================================================
# Observer Group Type Definitions
# ============================================================================
# Group A: Intuition Observer - No answer, direct judgment (直覺監考)
# Group D: Systematic Observer - No answer, but with P2+ trap analysis (系統化監考)
# ============================================================================


def build_blind_observer_context(subject_output: SubjectOutputV2) -> str:
    """
    Build context for Group A: Blind Observer.
    
    The observer sees:
    - The question
    - Subject's CoT (randomly sampled)
    - Subject's answer
    
    The observer does NOT see:
    - The correct answer
    - Any metadata about correctness
    
    This tests pure Theory of Mind ability.
    """
    return (
        "=== SUBJECT'S RESPONSE (You do NOT know the correct answer) ===\n\n"
        f"Question: {subject_output.question}\n\n"
        f"Subject's Reasoning:\n{subject_output.primary_cot}\n\n"
        f"Subject's Final Answer: {subject_output.majority_answer}\n"
    )


def build_informed_observer_context(subject_output: SubjectOutputV2) -> str:
    """
    Build context for Group B: Informed Observer.
    
    The observer sees:
    - The question
    - The CORRECT answer (Ground Truth)
    - Subject's CoT (randomly sampled)
    - Subject's answer
    
    This establishes the Hindsight Bias baseline.
    WARNING: May lead to overly harsh judgment.
    """
    return (
        "=== GROUND TRUTH ===\n"
        f"Correct Answer: {subject_output.ground_truth}\n\n"
        "=== SUBJECT'S RESPONSE ===\n"
        f"Question: {subject_output.question}\n\n"
        f"Subject's Reasoning:\n{subject_output.primary_cot}\n\n"
        f"Subject's Final Answer: {subject_output.majority_answer}\n"
    )


def build_frame_aware_observer_context(
    subject_output: SubjectOutputV2,
    trap_declaration: TrapDeclaration,
) -> str:
    """
    Build context for legacy Frame-Aware Observer (deprecated).
    Kept for backward compatibility.
    """
    return (
        "=== YOUR PRE-ANALYSIS (Completed BEFORE seeing subject's answer) ===\n"
        f"Question Summary: {trap_declaration.question_summary}\n"
        f"Difficulty for 20B Model: {trap_declaration.difficulty_assessment}\n\n"
        "Potential Traps You Identified:\n"
        + "\n".join(f"  - {trap}" for trap in trap_declaration.potential_traps)
        + "\n\nSuccess Indicators:\n"
        + "\n".join(f"  - {ind}" for ind in trap_declaration.success_indicators)
        + "\n\nFailure Indicators:\n"
        + "\n".join(f"  - {ind}" for ind in trap_declaration.failure_indicators)
        + "\n\n"
        "=== GROUND TRUTH ===\n"
        f"Correct Answer: {subject_output.ground_truth}\n\n"
        "=== SUBJECT'S RESPONSE (Now analyze against your pre-declared traps) ===\n"
        f"Question: {subject_output.question}\n\n"
        f"Subject's Reasoning:\n{subject_output.primary_cot}\n\n"
        f"Subject's Final Answer: {subject_output.majority_answer}\n"
    )


def build_systematic_observer_context(
    subject_output: SubjectOutputV2,
    trap_declaration: TrapDeclaration,
) -> str:
    """
    Build context for Group D: Systematic Observer (系統化監考).
    
    The observer:
    1. Has already declared potential traps BEFORE seeing subject's answer
    2. Now sees the subject's response with the trap analysis in mind
    3. Does NOT see the correct answer (blind evaluation)
    
    This tests whether structured thinking improves ToM ability without hindsight.
    """
    return (
        "=== YOUR PRE-ANALYSIS (Completed BEFORE seeing subject's answer) ===\n"
        f"Question Summary: {trap_declaration.question_summary}\n"
        f"Difficulty for 20B Model: {trap_declaration.difficulty_assessment}\n\n"
        "Potential Traps You Identified:\n"
        + "\n".join(f"  - {trap}" for trap in trap_declaration.potential_traps)
        + "\n\nSuccess Indicators:\n"
        + "\n".join(f"  - {ind}" for ind in trap_declaration.success_indicators)
        + "\n\nFailure Indicators:\n"
        + "\n".join(f"  - {ind}" for ind in trap_declaration.failure_indicators)
        + "\n\n"
        "=== SUBJECT'S RESPONSE (You do NOT know the correct answer) ===\n"
        f"Question: {subject_output.question}\n\n"
        f"Subject's Reasoning:\n{subject_output.primary_cot}\n\n"
        f"Subject's Final Answer: {subject_output.majority_answer}\n"
    )


# ============================================================================
# Legacy Protocol Context Builder (for backward compatibility)
# ============================================================================

def build_protocol_context(
    protocol: str,
    subject_output: SubjectOutputV2,
    observer_self_solve: Optional[ObserverSelfSolve] = None,
    canonicalized_output: Optional[CanonicalizedSubjectOutput] = None,
    observer_frame_check: Optional[ObserverFrameCheckSelfSolve] = None,
    trap_declaration: Optional[TrapDeclaration] = None,
) -> str:
    """
    Formats what the Observer is allowed to see (the Protocol P).
    
    New protocols:
    - "A_blind": Group A - Blind Observer (Pure ToM)
    - "B_informed": Group B - Informed Observer (Hindsight baseline)
    - "C_frame_aware": Group C - Frame-Aware with Trap Declaration (P2+)
    
    Legacy protocols maintained for backward compatibility.
    """
    # New experimental group protocols
    if protocol == "A_blind":
        return build_blind_observer_context(subject_output)
    
    elif protocol == "B_informed":
        return build_informed_observer_context(subject_output)
    
    elif protocol == "C_frame_aware":
        if not trap_declaration:
            raise ValueError("C_frame_aware requires trap_declaration data.")
        return build_frame_aware_observer_context(subject_output, trap_declaration)
    
    # Legacy protocols
    elif protocol == "P0_raw":
        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )
    elif protocol == "P1_final_answer_only":
        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )
    elif protocol == "P1_canonicalize":
        if not canonicalized_output:
            raise ValueError("P1_canonicalize requires canonicalized_output data.")
        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Canonicalized Reasoning: {canonicalized_output.canonical_reasoning}\n"
            f"Subject's Canonicalized Answer: {canonicalized_output.canonical_answer}\n"
        )
    elif protocol == "P2_self_solve":
        if not observer_self_solve:
            raise ValueError("P2_self_solve requires observer_self_solve data.")
        return (
            f"-- OBSERVER'S OWN GROUND-TRUTH RESOLUTION --\n"
            f"Your Reasoning: {observer_self_solve.reasoning}\n"
            f"Your Answer: {observer_self_solve.answer}\n"
            f"Your Confidence: {observer_self_solve.confidence}\n\n"
            f"-- SUBJECT'S OUTPUT TO JUDGE --\n"
            f"Question: {subject_output.question}\n"
            f"Subject's Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )
    elif protocol == "P2_frame_check_self_solve":
        if not observer_frame_check:
            raise ValueError("P2_frame_check_self_solve requires observer_frame_check data.")
        return (
            f"-- OBSERVER'S OWN GROUND-TRUTH RESOLUTION (FRAME-AWARE) --\n"
            f"Epistemic Frame: {observer_frame_check.epistemic_frame}\n"
            f"Frame Analysis: {observer_frame_check.frame_analysis}\n"
            f"Your Reasoning: {observer_frame_check.reasoning}\n"
            f"Your Answer: {observer_frame_check.answer}\n"
            f"Your Confidence: {observer_frame_check.confidence}\n\n"
            f"-- SUBJECT'S OUTPUT TO JUDGE --\n"
            f"Question: {subject_output.question}\n"
            f"Subject's Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )
    elif protocol == "P3_multi_sample":
        consistency_str = "High"
        if subject_output.behavioral_confidence < 0.5:
            consistency_str = "Low"
        elif subject_output.behavioral_confidence < 0.8:
            consistency_str = "Medium"

        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Representative Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
            f"[Behavioral Metadata]: The subject ran this {len(subject_output.samples)} times. "
            f"Their answer consistency was: {consistency_str}.\n"
        )
    else:
        raise ValueError(f"Unknown Protocol: {protocol}")
