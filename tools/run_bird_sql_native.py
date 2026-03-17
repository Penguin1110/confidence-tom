"""Run BIRD-SQL with benchmark-native execution evaluation.

Generation is still model-driven here, but evaluation is the official
execution-match style used by BIRD.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence_tom.client import LLMClient
from confidence_tom.evaluators import extract_sql


def load_items(split: str):
    base = ROOT / "external" / "birdsql" / "bird" / "llm" / "data" / split
    if (base / f"{split}.json").exists():
        questions_file = base / f"{split}.json"
        db_dir = base / f"{split}_databases"
    else:
        versioned = sorted(base.glob(f"{split}_*"))
        questions_file = versioned[-1] / f"{split}.json"
        db_dir = versioned[-1] / f"{split}_databases"
    return json.loads(questions_file.read_text(encoding="utf-8")), db_dir


def schema_for_db(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    rows = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(rows)


def evaluate(pred_sql: str, gold_sql: str, db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(pred_sql)
        pred_rows = set(map(tuple, cursor.fetchall()))
        cursor.execute(gold_sql)
        gold_rows = set(map(tuple, cursor.fetchall()))
        conn.close()
        return pred_rows == gold_rows
    except Exception:
        return False


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD-SQL with native execution evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="results/bird_sql_native.json")
    args = parser.parse_args()

    client = LLMClient(model=args.model, temperature=0.0, max_tokens=1024)
    items, db_dir = load_items(args.split)
    results = []
    for idx, item in enumerate(items[: args.num_samples]):
        db_id = item["db_id"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        schema = schema_for_db(db_path)
        prompt = (
            f"Database: {db_id}\n\n"
            f"Question: {item['question']}\n\n"
            f"Hint: {item.get('evidence', '')}\n\n"
            f"Schema:\n{schema}\n\n"
            "Output only a valid SQLite SQL query."
        )
        sql_text = await asyncio.to_thread(
            client.generate_text,
            [{"role": "system", "content": "You are a text-to-SQL model."}, {"role": "user", "content": prompt}],
        )
        sql = extract_sql(sql_text) or sql_text.strip()
        results.append(
            {
                "sql_idx": idx,
                "db_id": db_id,
                "predicted_sql": sql,
                "ground_truth_sql": item["SQL"],
                "correct": evaluate(sql, item["SQL"], db_path),
            }
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
