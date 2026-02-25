from confidence_tom.generator.models import SubjectOutputV2


def build_protocol_context(protocol: str, subject_output: SubjectOutputV2) -> str:
    """
    Formats what the Observer is allowed to see (the Protocol P).
    """
    if protocol == "P0_raw":
        # The most basic: Sees question, one CoT, final majority answer, and that's it
        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )

    elif protocol == "P1_final_answer_only":
        # Removes CoT entirely. Can observer judge confidence just from the final answer text?
        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
        )

    elif protocol == "P3_multi_sample":
        # Insight into behavioral confidence without exposing c_beh number directly
        # E.g., tells the observer the "consistency" of the subject across 5 runs.
        consistency_str = "High"
        if subject_output.behavioral_confidence < 0.5:
            consistency_str = "Low"
        elif subject_output.behavioral_confidence < 0.8:
            consistency_str = "Medium"

        return (
            f"Question: {subject_output.question}\n"
            f"Subject's Representative Reasoning Trace: {subject_output.primary_cot}\n"
            f"Subject's Final Answer: {subject_output.majority_answer}\n"
            f"[Behavioral Metadata]: The subject ran this 5 times. "
            f"Their answer consistency was: {consistency_str}.\n"
        )

    else:
        raise ValueError(f"Unknown Protocol: {protocol}")
