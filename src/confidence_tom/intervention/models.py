from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from confidence_tom.task_models import ApiTrace


class StepRecord(BaseModel):
    step: int = Field(ge=1)
    subgoal: str = Field(default="")
    reasoning: str = Field(default="")
    partial_answer: str = Field(default="")
    step_confidence: int = Field(ge=0, le=100)
    assumptions: list[str] = Field(default_factory=list)
    uncertainty_note: str = Field(default="")
    is_revision: bool = Field(default=False)
    revision_target: str = Field(default="")
    intermediate_result: str = Field(default="")
    verification_status: Literal["none", "partial", "verified", "failed"] = Field(default="none")

    @field_validator("verification_status", mode="before")
    @classmethod
    def normalize_verification_status(cls, value: Any) -> str:
        if value is None:
            return "none"
        text = str(value).strip().lower()
        aliases = {
            "complete": "verified",
            "completed": "verified",
            "done": "verified",
            "success": "verified",
            "ok": "verified",
            "incomplete": "partial",
            "in_progress": "partial",
        }
        return aliases.get(text, text)


class NextStepOutput(BaseModel):
    next_step: StepRecord
    done: bool = Field(default=False)
    final_answer: str = Field(default="")
    final_confidence: int = Field(ge=0, le=100, default=0)
    parse_incomplete: bool = Field(default=False)
    parse_incomplete_note: str = Field(default="")


class StepwiseWorkerOutput(BaseModel):
    steps: list[StepRecord] = Field(default_factory=list)
    final_answer: str = Field(default="")
    final_confidence: int = Field(ge=0, le=100, default=0)
    parse_incomplete: bool = Field(default=False)
    parse_incomplete_note: str = Field(default="")


class InterventionState(BaseModel):
    task_id: str
    step_index: int
    total_steps_available: int
    question: str
    steps_so_far: list[StepRecord] = Field(default_factory=list)
    current_partial_answer: str = Field(default="")
    current_step_confidence: float = Field(default=0.0)


class InterventionFeatureVector(BaseModel):
    task_id: str
    step_index: int
    current_step_confidence: float
    confidence_delta: float = 0.0
    max_confidence_drop_so_far: float = 0.0
    mean_confidence_drop_so_far: float = 0.0
    num_confidence_drops: int = 0
    partial_answer_changed: int = 0
    num_unique_partial_answers: int = 0
    self_correction_depth: float = 0.0
    backtracking_flag: int = 0
    reasoning_length: int = 0
    token_density_ratio: float = 1.0
    hedge_density: float = 0.0
    uncertainty_flag: int = 0
    assumptions_count: int = 0
    verification_status_code: int = 0
    semantic_drift: float = 0.0
    semantic_drift_velocity: float = 0.0
    window_variance: float = 0.0


class InterventionDecision(BaseModel):
    handoff: bool
    score: float = 0.0
    reason: str = Field(default="")
    router_name: str = Field(default="")


class CostBreakdown(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


class InterventionOutcome(BaseModel):
    task_id: str
    benchmark: str
    small_model: str
    large_model: str
    router_name: str
    handoff_step: Optional[int] = None
    handoff_trigger: str = ""
    success_small_only: bool
    success_after_handoff: bool
    small_answer: str = ""
    final_answer: str = ""
    voi_estimate: Optional[float] = None
    voi_realized: Optional[float] = None
    small_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    large_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    router_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    total_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    small_trace: StepwiseWorkerOutput = Field(default_factory=StepwiseWorkerOutput)
    takeover_trace: Optional[StepwiseWorkerOutput] = None
    decisions: list[InterventionDecision] = Field(default_factory=list)
    feature_history: list[InterventionFeatureVector] = Field(default_factory=list)
    small_api_trace: Optional[ApiTrace] = None
    large_api_trace: Optional[ApiTrace] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OracleGainStepResult(BaseModel):
    step_index: int
    prefix_steps: list[StepRecord] = Field(default_factory=list)
    small_continue_answer: str = ""
    small_continue_correct: bool = False
    large_takeover_answer: str = ""
    large_takeover_correct: bool = False
    delta_correctness: float = 0.0
    small_continue_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    large_takeover_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    small_continue_trace: Optional[StepwiseWorkerOutput] = None
    large_takeover_trace: Optional[StepwiseWorkerOutput] = None
    small_continue_api_trace: Optional[ApiTrace] = None
    large_takeover_api_trace: Optional[ApiTrace] = None


class OracleGainTaskResult(BaseModel):
    task_id: str
    benchmark: str
    small_model: str
    large_model: str
    base_small_answer: str = ""
    base_small_correct: bool = False
    base_small_trace: StepwiseWorkerOutput = Field(default_factory=StepwiseWorkerOutput)
    oracle_gain_steps: list[OracleGainStepResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrefixSegment(BaseModel):
    segment_id: str
    index: int = Field(ge=1)
    text: str = Field(default="")


class SegmentedTraceOutput(BaseModel):
    segments: list[PrefixSegment] = Field(default_factory=list)
    final_answer: str = Field(default="")
    parse_incomplete: bool = Field(default=False)
    parse_incomplete_note: str = Field(default="")


class ExtractedFinalAnswerOutput(BaseModel):
    final_answer: str = Field(default="")
    parse_incomplete: bool = Field(default=False)
    parse_incomplete_note: str = Field(default="")


class PrefixOracleGainStepResult(BaseModel):
    prefix_id: str
    parent_prefix_id: str = Field(default="")
    step_index: int
    prefix_segments: list[PrefixSegment] = Field(default_factory=list)
    prefix_text: str = Field(default="")
    small_continue_answer: str = Field(default="")
    small_continue_correct: bool = False
    large_takeover_answer: str = Field(default="")
    large_takeover_correct: bool = False
    delta_correctness: float = 0.0
    small_continue_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    large_takeover_cost: CostBreakdown = Field(default_factory=CostBreakdown)
    small_continue_text: str = Field(default="")
    large_takeover_text: str = Field(default="")
    small_continue_api_trace: Optional[ApiTrace] = None
    large_takeover_api_trace: Optional[ApiTrace] = None


class PrefixOracleGainTaskResult(BaseModel):
    task_id: str
    benchmark: str
    small_model: str
    large_model: str
    trace_id: str
    full_trace_text: str = Field(default="")
    full_trace_answer: str = Field(default="")
    full_trace_correct: bool = False
    full_trace_api_trace: Optional[ApiTrace] = None
    segments: list[PrefixSegment] = Field(default_factory=list)
    prefix_oracle_steps: list[PrefixOracleGainStepResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
