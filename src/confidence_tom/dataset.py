import logging
import random
from typing import Dict, List, cast

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_gsm8k(split: str = "train", num_samples: int = 100) -> List[Dict[str, str]]:
    """Loads GSM8K examples as L1 Ambiguity (Multi-step deterministic)."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    formatted = []
    for i, item in enumerate(samples):
        answer_text = str(item["answer"])
        gt_answer = answer_text.split("####")[-1].strip()

        formatted.append(
            {
                "id": f"gsm8k_{i}",
                "question": str(item["question"]),
                "ground_truth": gt_answer,
                "ambiguity_level": "L1 (Multi-step deterministic)",
            }
        )
    return formatted


def load_truthfulqa(split: str = "validation", num_samples: int = 100) -> List[Dict[str, str]]:
    """Loads TruthfulQA examples mapped to L3a, L3b, and L4 Ambiguity."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    samples = dataset.shuffle(seed=42)

    # Target counts per taxonomy
    target_each = max(1, num_samples // 3)
    counts = {"L3a": 0, "L3b": 0, "L4": 0}

    formatted = []
    for i, item in enumerate(samples):
        category = str(item["category"]).lower()

        # Taxonomy mapping based on TruthfulQA categories
        if any(
            c in category
            for c in ["science", "history", "health", "law", "finance", "politics", "religion"]
        ):
            level = "L3a (Open-world factual)"
            key = "L3a"
        elif any(
            c in category
            for c in [
                "fiction",
                "myth",
                "superstition",
                "paranormal",
                "conspiracies",
                "misconceptions",
                "mandela",
            ]
        ):
            level = "L3b (False-premise / fictional trap)"
            key = "L3b"
        else:
            level = "L4 (Underspecified / social meaning)"
            key = "L4"

        if counts[key] >= target_each:
            continue

        correct_list = cast(list[str], item["correct_answers"])
        gt_answer = str(correct_list[0]) if correct_list else "Unknown"

        formatted.append(
            {
                "id": f"truthfulqa_{key}_{i}",
                "question": str(item["question"]),
                "ground_truth": gt_answer,
                "ambiguity_level": level,
            }
        )
        counts[key] += 1

        if sum(counts.values()) >= num_samples:
            break

    return formatted


def load_mixed_dataset(num_per_level: int = 5) -> List[Dict[str, str]]:
    """Loads a mixed dataset of GSM8K and TruthfulQA."""
    logger.info(f"Loading {num_per_level} questions per level...")
    l1_qs = load_gsm8k(num_samples=num_per_level)
    l3_qs = load_truthfulqa(num_samples=num_per_level * 3)  # Load 3x since we split into 3 groups

    combined = l1_qs + l3_qs
    random.shuffle(combined)
    return combined
