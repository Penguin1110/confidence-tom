import logging
import random
from typing import Dict, List, cast

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_gsm8k(split: str = "train", num_samples: int = 100) -> List[Dict[str, str]]:
    """Loads GSM8K examples as L1 Ambiguity (Multi-step deterministic)."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    # Shuffle and pick num_samples
    samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    formatted = []
    for i, item in enumerate(samples):
        # Extract ground truth exact answer from GSM8K (it's after '#### ')
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
    """Loads TruthfulQA examples as L3 Ambiguity (Open-world factual)."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    formatted = []
    for i, item in enumerate(samples):
        # TruthfulQA ground truth is often the best answer in 'correct_answers'
        correct_list = cast(list[str], item["correct_answers"])
        gt_answer = str(correct_list[0]) if correct_list else "Unknown"

        formatted.append(
            {
                "id": f"truthfulqa_{i}",
                "question": str(item["question"]),
                "ground_truth": gt_answer,
                "ambiguity_level": "L3 (Open-world factual)",
            }
        )
    return formatted


def load_mixed_dataset(num_per_level: int = 5) -> List[Dict[str, str]]:
    """Loads a mixed dataset of GSM8K and TruthfulQA."""
    logger.info(f"Loading {num_per_level} questions per level...")
    l1_qs = load_gsm8k(num_samples=num_per_level)
    l3_qs = load_truthfulqa(num_samples=num_per_level)

    combined = l1_qs + l3_qs
    random.shuffle(combined)
    return combined
