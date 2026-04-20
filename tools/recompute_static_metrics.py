import json
from pathlib import Path

from confidence_tom.dataset_models import StaticTask
from confidence_tom.static_evaluators import build_static_evaluator


def recompute_file(path: Path) -> None:
    rows = json.loads(path.read_text())
    changed = 0

    for row in rows:
        task = StaticTask(
            id=row["question_id"],
            question=row.get("question", ""),
            choices=row.get("choices", []) or [],
            correct_answer=row.get("correct_answer", "") or "",
            reference_answer=row.get("reference_answer", "") or "",
            category=row.get("category", ""),
            source=row.get("source", ""),
            answer_format=row.get("answer_format", "multiple_choice"),
            evaluator_name=row.get("evaluator_name", "mc_letter"),
            task_type=row.get("task_type", "QA"),
            environment_context=row.get("environment_context", []) or [],
            metadata=row.get("metadata", {}) or {},
            external_difficulty=row.get("external_difficulty"),
        )
        evaluator = build_static_evaluator(task)
        samples = row.get("sample_traces", [])
        answers = [
            str(s.get("answer", "")).strip() for s in samples if str(s.get("answer", "")).strip()
        ]
        if not answers:
            continue

        correct_flags = [evaluator(answer, task).is_correct for answer in answers]
        c_beh = sum(correct_flags) / len(correct_flags)

        distribution = row.get("answer_distribution", {}) or {}
        majority_count = max(distribution.values()) if distribution else 0
        c_consistency = majority_count / len(answers)

        new_values = {
            "c_beh": round(c_beh, 4),
            "c_consistency": round(c_consistency, 4),
            "gap": round(float(row.get("c_rep", 0.0)) - c_beh, 4),
        }
        for key, value in new_values.items():
            if row.get(key) != value:
                row[key] = value
                changed += 1

    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(path.name, "updated_fields", changed)


def main() -> None:
    base = Path("results/qwen3_thinkoff_k10_olympiad_livebench_50")
    for name in ["Qwen-3-8B.json", "Qwen-3-14B.json", "Qwen-3-32B.json"]:
        recompute_file(base / name)


if __name__ == "__main__":
    main()
