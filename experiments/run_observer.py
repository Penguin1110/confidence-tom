import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.observer.models import RecursiveLevelResult
from confidence_tom.observer.observer import RecursiveObserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    input_file = Path(cfg.output_dir) / "generator_level0_results.json"

    if not input_file.exists():
        logger.error(f"Input file {input_file} not found! Run generator first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        level0_results = json.load(f)

    # Instantiate Observer (using the first model from the config list for now)
    observer_models = cfg.observer.models
    # Let's say we just use the first observer model down the recursive chain for simplicity.
    # Or, we could pass the baton between models!
    # Here, we keep the model fixed for the recursive tower.
    observer = RecursiveObserver(
        model_name=observer_models[0], protocol="full_cot", temperature=cfg.observer.temperature
    )

    # We will compute recursive judgments for levels k=1 to 3
    MAX_LEVELS = 3

    final_results = []

    for item in level0_results:
        question = item["question"]
        logger.info(f"Processing evaluation for Question: {item['question_id']}")

        # We test across all 'styled_variants' to satisfy Milestone 1 (Style-Content Confounding)
        evaluated_variants = []
        for variant in item["styled_variants"]:
            style_name = variant["style_name"]
            cot = variant["styled_cot"]
            answer = variant["answer"]
            true_conf = variant["true_confidence"]

            logger.info(f"  -> Playing Recursive Game for style: [{style_name}]")

            recursive_chain: list[RecursiveLevelResult] = []

            for k in range(1, MAX_LEVELS + 1):
                logger.info(f"      Running Level-{k} observer...")
                # The observer gets the history of all previous levels
                result = observer.evaluate(
                    level=k,
                    question=question,
                    answer=answer,
                    subject_cot=cot,
                    previous_judgments=recursive_chain,
                )

                if result:
                    recursive_chain.append(result)
                else:
                    logger.warning(f"Failed to get Level-{k} judgment.")
                    break

            # Save chain for this variant
            evaluated_variants.append(
                {
                    "style_name": style_name,
                    "styled_cot": cot,
                    "answer": answer,
                    "true_confidence": true_conf,
                    "oversight_chain": [rc.model_dump() for rc in recursive_chain],
                }
            )

        final_results.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "evaluated_variants": evaluated_variants,
            }
        )

    # Save to disk
    output_dir = Path(cfg.output_dir)
    out_file = output_dir / "observer_recursive_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Finished! Recursive observation results saved to {out_file}")


if __name__ == "__main__":
    main()
