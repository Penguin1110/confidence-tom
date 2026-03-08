import json


def main():
    try:
        with open("results/generator_v2_results_balanced.json", "r") as f:
            gen_data = json.load(f)

        gen_map = {item["question_id"]: item for item in gen_data}

        with open("results/observer_v2_recursive_results.json", "r") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if item.get("ambiguity_level", "").startswith("L3b") or item.get(
                "ambiguity_level", ""
            ).startswith("L4"):
                framing = item.get("framing", "")
                qid = item.get("question_id")
                gen_item = gen_map.get(qid, {})

                p0_eval = None
                p2_eval = None

                for p_chain in item.get("evaluations_by_protocol", []):
                    if p_chain["protocol"] == "P0_raw" and len(p_chain["oversight_chain"]) > 0:
                        p0_eval = p_chain["oversight_chain"][0]["judgment"]
                    if (
                        p_chain["protocol"] == "P2_frame_check_self_solve"
                        and len(p_chain["oversight_chain"]) > 0
                    ):
                        p2_eval = p_chain["oversight_chain"][0]["judgment"]

                if (
                    p0_eval
                    and p2_eval
                    and p0_eval["is_overconfident"] != p2_eval["is_overconfident"]
                ):
                    print("\n" + "=" * 50)
                    print(f"Question [{item['ambiguity_level']}] ({framing}): {item['question']}")
                    print(
                        f"Subject Answer:\n{gen_item.get('primary_cot', '')[:200]}... \n=> {gen_item.get('majority_answer', '')}\n"
                    )

                    print("--- P0 (Raw) Judgment ---")
                    print(f"Is Overconfident? {p0_eval['is_overconfident']}")
                    print(f"Reasoning: {p0_eval['reasoning']}")

                    print("\n--- P2+ (Frame-Check) Judgment ---")
                    print(f"Is Overconfident? {p2_eval['is_overconfident']}")
                    print(f"Reasoning: {p2_eval['reasoning']}")

                    count += 1
                    if count >= 3:
                        break
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
