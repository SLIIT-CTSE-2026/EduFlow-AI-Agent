import os
from config import STATE_AG03

def evaluate_agent_03():
    print("STARTING EVALUATION: AGENT 03 (QUIZ GENERATOR)\n")

    if not os.path.exists(STATE_AG03):
        print(f"FAILED: Quiz file '{STATE_AG03}' not found.")
        return

    with open(STATE_AG03, "r", encoding="utf-8") as f:
        content = f.read().strip()

    print("Testing quiz file...\n")

    # Property 1: Not empty
    if content:
        print("PROPERTY PASSED: Quiz content is not empty.")
    else:
        print("PROPERTY FAILED: Quiz content is empty.")

    # Property 2: Question count
    q_count = content.count("Q")
    if q_count >= 5:
        print(f"PROPERTY PASSED: {q_count} questions detected.")
    else:
        print(f"PROPERTY FAILED: Only {q_count} questions found.")

    # Property 3: Options exist
    if all(opt in content for opt in ["A)", "B)", "C)"]):
        print("PROPERTY PASSED: Required options A/B/C exist.")
    else:
        print("PROPERTY FAILED: Missing MCQ options.")

    # Property 4: Answers exist
    if "Answer:" in content:
        print("PROPERTY PASSED: Answer lines present.")
    else:
        print("PROPERTY FAILED: Missing answers.")

    # Property 5: No JSON
    if "{" not in content and "}" not in content:
        print("PROPERTY PASSED: No JSON contamination.")
    else:
        print("PROPERTY FAILED: JSON detected in output.")

    # Property 6: Length check
    if len(content) > 200:
        print("PROPERTY PASSED: Content length is valid.")
    else:
        print("PROPERTY FAILED: Content too short.")

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    evaluate_agent_03()