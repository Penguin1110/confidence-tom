"""
Hard Benchmark Test - Testing Small Models on Expert-Level Datasets

簡化版：只測試正確率

資料集：GPQA Diamond + MATH Level 4-5 混合
- GPQA: 專家級科學推理
- MATH: 多步數學運算

流程：小模型答題 → 大模型批改 → 統計正確率
支援多輪測試
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from confidence_tom.client import LLMClient
from confidence_tom.dataset import load_gpqa_math_mix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


SUBJECT_PROMPT = """Solve this problem step by step.

{question}

Provide your final answer at the end in the format:
ANSWER: [your answer]
"""

GRADER_PROMPT = """Judge if the student's answer is correct.

QUESTION:
{question}

CORRECT ANSWER: {ground_truth}

STUDENT'S ANSWER: {student_answer}

Reply with only: CORRECT or INCORRECT
"""


async def run_benchmark(
    dataset_name: str,
    questions: list[dict],
    subject_model: str,
    observer_model: str,
    output_dir: Path,
) -> dict:
    """Run benchmark and return accuracy statistics."""
    
    subject_client = LLMClient(model=subject_model, temperature=0.7, max_tokens=2048)
    grader_client = LLMClient(model=observer_model, temperature=0, max_tokens=64)
    
    results = []
    correct_count = 0
    
    for i, q in enumerate(questions):
        qid = q["id"]
        logger.info(f"[{dataset_name}] {i+1}/{len(questions)}: {qid}")
        
        # Subject answers
        prompt = SUBJECT_PROMPT.format(question=q["question"])
        response = await subject_client.aclient.chat.completions.create(
            model=subject_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        student_answer = response.choices[0].message.content or ""
        
        # Extract answer
        if "ANSWER:" in student_answer:
            extracted = student_answer.split("ANSWER:")[-1].strip().split("\n")[0]
        else:
            extracted = student_answer.strip().split("\n")[-1]
        
        # Grader judges
        grader_prompt = GRADER_PROMPT.format(
            question=q["question"][:1000],
            ground_truth=q["ground_truth"],
            student_answer=extracted,
        )
        grader_response = await grader_client.aclient.chat.completions.create(
            model=observer_model,
            messages=[{"role": "user", "content": grader_prompt}],
            temperature=0,
            max_tokens=64,
        )
        judgment = grader_response.choices[0].message.content or ""
        is_correct = "CORRECT" in judgment.upper() and "INCORRECT" not in judgment.upper()
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "id": qid,
            "ground_truth": q["ground_truth"],
            "student_answer": extracted[:200],
            "is_correct": is_correct,
        })
        
        logger.info(f"  {'✓' if is_correct else '✗'} | Answer: {extracted[:50]}...")
    
    accuracy = correct_count / len(questions) if questions else 0
    
    # Save results
    output_file = output_dir / f"hard_benchmark_{dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return {
        "dataset": dataset_name,
        "total": len(questions),
        "correct": correct_count,
        "accuracy": accuracy,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config_hard")
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


async def amain(cfg: DictConfig) -> None:
    subject_model = cfg.generator.model
    observer_model = cfg.observer.models[0]
    num_samples = cfg.dataset.get("num_samples", 30)
    num_rounds = cfg.get("num_rounds", 1)  # 支援多輪測試
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("HARD BENCHMARK - GPQA + MATH 正確率測試")
    logger.info("=" * 50)
    logger.info(f"小模型: {subject_model}")
    logger.info(f"批改模型: {observer_model}")
    logger.info(f"每組題數: {num_samples}")
    logger.info(f"測試輪數: {num_rounds}")
    logger.info("=" * 50)
    
    # 載入混合資料集
    questions = load_gpqa_math_mix(num_per_task=num_samples)
    logger.info(f"\n載入 {len(questions)} 題 (GPQA + MATH 混合)")
    
    all_round_stats = []
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"第 {round_num} 輪")
        logger.info("=" * 50)
        
        stats = await run_benchmark(
            dataset_name=f"gpqa_math_mix_round{round_num}",
            questions=questions,
            subject_model=subject_model,
            observer_model=observer_model,
            output_dir=output_dir,
        )
        all_round_stats.append(stats)
        logger.info(f"第 {round_num} 輪正確率: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("總結")
    logger.info("=" * 50)
    
    accuracies = [s["accuracy"] for s in all_round_stats]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    for i, s in enumerate(all_round_stats, 1):
        logger.info(f"  第 {i} 輪: {s['accuracy']:.1%}")
    logger.info(f"  平均: {avg_accuracy:.1%}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "subject_model": subject_model,
        "observer_model": observer_model,
        "num_samples_per_task": num_samples,
        "num_rounds": num_rounds,
        "round_results": all_round_stats,
        "average_accuracy": avg_accuracy,
    }
    with open(output_dir / "hard_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果已儲存至: {output_dir}")


if __name__ == "__main__":
    main()
