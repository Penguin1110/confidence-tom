from typing import Optional

from confidence_tom.generator.models import SubjectOutputV2
from confidence_tom.observer.models import ObserverSelfSolve


def build_protocol_context(
    protocol: str,
    subject_output: SubjectOutputV2,
    observer_self_solve: Optional[ObserverSelfSolve] = None,
) -> str:
    """
    Formats what the Observer is allowed to see (the Protocol P).
    """
    if protocol == "P0_raw":
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
