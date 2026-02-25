import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.dataset import load_mixed_dataset
from confidence_tom.generator.generator import SubjectGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    gen_cfg = cfg.generator
    logger.info(f"Initializing SubjectGenerator with model: {gen_cfg.model}")
    subject_gen = SubjectGenerator(
        model_name=gen_cfg.model,
        temperature=gen_cfg.temperature,
        k_samples=gen_cfg.k_samples if "k_samples" in gen_cfg else 5,
    )

    questions = load_mixed_dataset(num_per_level=cfg.dataset.num_samples)

    results = []

    for q in questions:
        # For L3b and L4, test framing interventions
        framings = ["standard"]
        if "L3b" in q["ambiguity_level"] or "L4" in q["ambiguity_level"]:
            framings = ["real-world", "in-universe"]

        for framing in framings:
            logger.info(
                f"Processing question {q['id']} [Level: {q['ambiguity_level']}] "
                f"[Framing: {framing}]..."
            )
            solved = subject_gen.solve(
                question_id=f"{q['id']}_{framing}",
                question=q["question"],
                ground_truth=q["ground_truth"],
                ambiguity_level=q["ambiguity_level"],
                framing=framing,
            )

            if not solved:
                logger.warning(f"Failed to solve question {q['id']}")
                continue

            logger.info(
                f"[SOLVED] {q['id']} | "
                f"c_beh: {solved.behavioral_confidence} | "
                f"c_rep: {solved.avg_reported_confidence:.1f}"
            )

            results.append(solved.model_dump())

            # 即時存檔 (Real-time saving)
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / "generator_v2_results.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Finished! Results saved to {out_file}")


if __name__ == "__main__":
    main()
