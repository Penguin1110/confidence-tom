import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.generator.generator import SubjectGenerator
from confidence_tom.generator.styler import StyleTransferer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# sample dataset for testing before integrating with huggingface datasets
SAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "question": (
            "A class has 30 students. 60% are girls. 50% of the girls wear glasses. "
            "How many girls wear glasses?"
        ),
    },
    {
        "id": "q2",
        "question": (
            "If you drive a car at 60 mph for 2.5 hours, how far will you travel "
            "in kilometers? (1 mile = 1.609 km)"
        ),
    },
]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Setup Generator (Level 0 Subject)
    gen_cfg = cfg.generator
    logger.info(f"Initializing SubjectGenerator with model: {gen_cfg.model}")
    subject_gen = SubjectGenerator(model_name=gen_cfg.model, temperature=gen_cfg.temperature)

    # 2. Setup Styler (Style-equivalent Transformer)
    sty_cfg = cfg.styler
    logger.info(f"Initializing StyleTransferer with model: {sty_cfg.model}")
    styler = StyleTransferer(model_name=sty_cfg.model, temperature=sty_cfg.temperature)

    results = []

    # 3. Process Questions
    for q in SAMPLE_QUESTIONS:
        logger.info(f"Processing question {q['id']}...")
        solved = subject_gen.solve(question_id=q["id"], question=q["question"])

        if not solved:
            logger.warning(f"Failed to solve question {q['id']}")
            continue

        logger.info(f"[SOLVED] True Confidence: {solved.true_confidence}%")

        styled_items = styler.restyle(solved)
        if not styled_items:
            logger.warning(f"Failed to restyle question {q['id']}")
            styled_items = []
        else:
            logger.info(f"[STYLED] Successfully generated {len(styled_items)} styles.")

        # Save all results to dictionary format
        results.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "original_answer": solved.answer,
                "true_confidence": solved.true_confidence,
                "original_cot": solved.cot,
                "styled_variants": [s.model_dump() for s in styled_items],
            }
        )

    # 4. Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "generator_level0_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Finished! Results saved to {out_file}")


if __name__ == "__main__":
    main()
