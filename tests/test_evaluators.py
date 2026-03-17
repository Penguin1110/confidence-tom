import sqlite3

from confidence_tom.evaluators import build_evaluator, evaluate_bird_sql, evaluate_plancraft, evaluate_tau_bench, extract_sql
from confidence_tom.task_models import DynamicTask


def test_extract_sql_from_code_fence() -> None:
    text = "```sql\nSELECT 1\n```\nThis works."
    assert extract_sql(text) == "SELECT 1"


def test_bird_sql_accepts_wrapped_sql(tmp_path) -> None:
    db_path = tmp_path / "sample.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE numbers (value INTEGER)")
    conn.execute("INSERT INTO numbers VALUES (1)")
    conn.commit()
    conn.close()

    task = DynamicTask(
        task_id="bird_test",
        benchmark="bird-sql",
        instruction="",
        ground_truth="SELECT value FROM numbers;",
        metadata={"db_path": str(db_path)},
    )
    assert evaluate_bird_sql("SQL Query: ```sql\nSELECT value FROM numbers;\n```", task)


def test_plancraft_rejects_contradiction_even_with_target_answer() -> None:
    task = DynamicTask(
        task_id="plancraft_test",
        benchmark="plancraft",
        instruction=(
            "You are playing Minecraft. Craft the following item: cake.\n"
            "Your current inventory: milk_bucket x3, egg x1, wheat x3, sugar_cane x2.\n"
            "List the crafting steps needed and state your final answer as the target item name."
        ),
        ground_truth="cake",
        metadata={"target_item": "cake"},
    )
    evidence = "Cannot craft cake - insufficient wheat."
    assert not evaluate_plancraft("cake", task, evidence)


def test_plancraft_accepts_reachable_target() -> None:
    task = DynamicTask(
        task_id="plancraft_test",
        benchmark="plancraft",
        instruction=(
            "You are playing Minecraft. Craft the following item: sugar.\n"
            "Your current inventory: sugar_cane x2.\n"
            "List the crafting steps needed and state your final answer as the target item name."
        ),
        ground_truth="sugar",
        metadata={"target_item": "sugar", "initial_inventory": {"sugar_cane": 2}},
    )
    assert evaluate_plancraft("sugar", task, "Craft sugar from sugar cane.")


def test_tau_bench_uses_outputs_and_full_evidence() -> None:
    task = DynamicTask(
        task_id="tau_test",
        benchmark="tau-bench",
        instruction="",
        ground_truth={
            "actions": [
                {"name": "find_user_id_by_name_zip", "kwargs": {"zip": "19122"}},
                {"name": "return_delivered_order_items", "kwargs": {"order_id": "#W2378156"}},
            ],
            "outputs": ["10"],
        },
        metadata={},
    )
    final_answer = "Returned the requested items."
    evidence = (
        "Action: find_user_id_by_name_zip\n"
        "Action: return_delivered_order_items\n"
        "Summary: There are 10 tshirt options available."
    )
    assert evaluate_tau_bench(final_answer, task, evidence)


def test_tau_bench_rejects_missing_required_output() -> None:
    task = DynamicTask(
        task_id="tau_test",
        benchmark="tau-bench",
        instruction="",
        ground_truth={
            "actions": [{"name": "return_delivered_order_items", "kwargs": {"order_id": "#W2378156"}}],
            "outputs": ["10"],
        },
        metadata={},
    )
    assert not evaluate_tau_bench("There are 35 options.", task, "Action: return_delivered_order_items")


def test_build_evaluator_returns_callable() -> None:
    assert callable(build_evaluator("plancraft"))
