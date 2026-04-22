"""Model configuration for the confidence calibration experiment.

Subject models (small): run tasks K times, self-report confidence.
Observer models (large): judge subject's confidence state.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short identifier used in filenames/results
    api_id: str  # OpenRouter model ID
    display_name: str


# ---------------------------------------------------------------------------
# Subject models (small) — run on dynamic benchmark tasks
# ---------------------------------------------------------------------------

SUBJECT_MODELS: list[ModelSpec] = [
    ModelSpec(
        key="gemma-3-4b",
        api_id="google/gemma-3-4b-it",
        display_name="Gemma-3-4B",
    ),
    ModelSpec(
        key="gemma-3-12b",
        api_id="google/gemma-3-12b-it",
        display_name="Gemma-3-12B",
    ),
    ModelSpec(
        key="gemma-3-27b",
        api_id="google/gemma-3-27b-it",
        display_name="Gemma-3-27B",
    ),
    ModelSpec(
        key="qwen-3-8b",
        api_id="qwen/qwen3-8b",
        display_name="Qwen-3-8B",
    ),
    ModelSpec(
        key="qwen-3-32b",
        api_id="qwen/qwen3-32b",
        display_name="Qwen-3-32B",
    ),
]

# ---------------------------------------------------------------------------
# Observer models (large) — judge subject confidence
# ---------------------------------------------------------------------------

OBSERVER_MODELS: list[ModelSpec] = [
    ModelSpec(
        key="claude_opus_4_6",
        api_id="anthropic/claude-opus-4.6",
        display_name="Claude-Opus-4.6",
    ),
    ModelSpec(
        key="gemini_3_1_pro_preview",
        api_id="google/gemini-3.1-pro-preview",
        display_name="Gemini-3.1-Pro-Preview",
    ),
    ModelSpec(
        key="gpt_5_4",
        api_id="openai/gpt-5.4-pro",
        display_name="GPT-5.4-Pro",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUBJECT_KEYS = {m.key for m in SUBJECT_MODELS}
OBSERVER_KEYS = {m.key for m in OBSERVER_MODELS}


def get_subject(key: str) -> ModelSpec:
    for m in SUBJECT_MODELS:
        if m.key == key:
            return m
    raise KeyError(f"Unknown subject model '{key}'. Known: {sorted(SUBJECT_KEYS)}")


def get_observer(key: str) -> ModelSpec:
    for m in OBSERVER_MODELS:
        if m.key == key:
            return m
    raise KeyError(f"Unknown observer model '{key}'. Known: {sorted(OBSERVER_KEYS)}")
