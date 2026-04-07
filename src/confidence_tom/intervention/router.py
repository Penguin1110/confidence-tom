from __future__ import annotations

from abc import ABC, abstractmethod

from confidence_tom.intervention.models import InterventionDecision, InterventionFeatureVector


class BaseRouter(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def decide(self, features: InterventionFeatureVector) -> InterventionDecision:
        raise NotImplementedError


class ThresholdRouter(BaseRouter):
    def __init__(
        self,
        min_step_confidence: float = 0.45,
        min_drop_intensity: float = 0.2,
        min_token_density_ratio: float = 1.8,
        min_hedge_density: float = 0.03,
        min_semantic_drift: float = 0.35,
    ) -> None:
        super().__init__(name="threshold_router")
        self.min_step_confidence = min_step_confidence
        self.min_drop_intensity = min_drop_intensity
        self.min_token_density_ratio = min_token_density_ratio
        self.min_hedge_density = min_hedge_density
        self.min_semantic_drift = min_semantic_drift

    def decide(self, features: InterventionFeatureVector) -> InterventionDecision:
        score = 0.0
        reasons: list[str] = []

        if features.current_step_confidence < self.min_step_confidence:
            score += 0.35
            reasons.append("low_step_confidence")
        if features.max_confidence_drop_so_far >= self.min_drop_intensity:
            score += 0.2
            reasons.append("confidence_drop")
        if features.token_density_ratio >= self.min_token_density_ratio:
            score += 0.15
            reasons.append("token_burst")
        if features.hedge_density >= self.min_hedge_density:
            score += 0.1
            reasons.append("hedging")
        if features.partial_answer_changed:
            score += 0.1
            reasons.append("answer_changed")
        if features.backtracking_flag:
            score += 0.1
            reasons.append("backtracking")
        if features.semantic_drift >= self.min_semantic_drift:
            score += 0.1
            reasons.append("semantic_drift")
        if features.verification_status_code == 3:
            score += 0.2
            reasons.append("verification_failed")

        handoff = score >= 0.45
        return InterventionDecision(
            handoff=handoff,
            score=min(score, 1.0),
            reason=",".join(reasons) if reasons else "continue",
            router_name=self.name,
        )
