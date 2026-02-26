import json
import os
from collections import defaultdict


def main() -> None:
    input_file = "results/generator_v2_results.json"
    output_file = "results/generator_v2_results_balanced.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by level and framing
    groups = defaultdict(list)
    for item in data:
        level = item.get("ambiguity_level", "").split(" ")[0]
        framing = item.get("framing", "standard")
        groups[(level, framing)].append(item)

    # Find minimum size among all configurations
    min_size = min(len(v) for v in groups.values())
    print(f"Minimum group size is {min_size}. Balancing all groups to {min_size}...")

    balanced_data = []
    for (level, framing), items in groups.items():
        # Sort by question ID to ensure we pick deterministically
        sorted_items = sorted(items, key=lambda x: x["question_id"])
        selected = sorted_items[:min_size]
        balanced_data.extend(selected)
        print(f"Selected {len(selected)} items for Level: {level}, Framing: {framing}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)

    print(f"\\Saved balanced dataset with {len(balanced_data)} total items to {output_file}")


if __name__ == "__main__":
    main()
