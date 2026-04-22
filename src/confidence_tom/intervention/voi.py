from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from confidence_tom.data.task_models import ApiTrace
from confidence_tom.intervention.models import CostBreakdown


@dataclass
class ModelPricing:
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    reasoning_per_1k: float = 0.0


def trace_to_cost(
    trace: Optional[ApiTrace], pricing: Optional[ModelPricing] = None
) -> CostBreakdown:
    if trace is None:
        return CostBreakdown()
    estimated = None
    if pricing is not None:
        estimated = (
            trace.prompt_tokens / 1000.0 * pricing.input_per_1k
            + trace.completion_tokens / 1000.0 * pricing.output_per_1k
            + trace.reasoning_tokens / 1000.0 * pricing.reasoning_per_1k
        )
    return CostBreakdown(
        input_tokens=trace.prompt_tokens,
        output_tokens=trace.completion_tokens,
        reasoning_tokens=trace.reasoning_tokens,
        total_tokens=trace.total_tokens,
        estimated_cost_usd=estimated,
    )


def combine_costs(*costs: CostBreakdown) -> CostBreakdown:
    estimated = None
    estimated_values = [c.estimated_cost_usd for c in costs if c.estimated_cost_usd is not None]
    if estimated_values:
        estimated = sum(estimated_values)
    return CostBreakdown(
        input_tokens=sum(c.input_tokens for c in costs),
        output_tokens=sum(c.output_tokens for c in costs),
        reasoning_tokens=sum(c.reasoning_tokens for c in costs),
        total_tokens=sum(c.total_tokens for c in costs),
        estimated_cost_usd=estimated,
    )


def estimate_voi(
    p_takeover: float,
    p_continue: float,
    takeover_cost: float,
    continue_cost: float,
    lambda_cost: float,
) -> float:
    return (p_takeover - lambda_cost * takeover_cost) - (p_continue - lambda_cost * continue_cost)
