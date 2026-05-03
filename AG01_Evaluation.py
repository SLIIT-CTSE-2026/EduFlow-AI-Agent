import os
from config import STATE_AG01, STUDY_MATERIALS_PATH

def evaluate_agent_01():
    print("STARTING EVALUATION: AGENT 01 (DOCUMENT SEARCHER)\n")
    
    # Load the result from the state file
    if not os.path.exists(STATE_AG01):
        print(f"FAILED: State file '{STATE_AG01}' not found.")
        return

    with open(STATE_AG01, "r") as f:
        file_path = f.read().strip()

    print(f"Testing path: {file_path}")

    # Existence
    if os.path.exists(file_path):
        print("PROPERTY PASSED: File physically exists on disk.")
    else:
        print("PROPERTY FAILED: Path exists in state but file is missing on disk.")

    # File Type Integrity
    if file_path.lower().endswith('.pdf'):
        print("PROPERTY PASSED: Resource is a valid PDF format.")
    else:
        print("PROPERTY FAILED: Resource is not a PDF (Handoff to Agent 2 may fail).")

    # Security Constraint - not accessed files outside the allowed folder
    clean_materials_path = os.path.abspath(STUDY_MATERIALS_PATH)
    clean_file_path = os.path.abspath(file_path)
    
    if clean_file_path.startswith(clean_materials_path):
        print("SECURITY PASSED: Path is confined to the authorized './study_materials' directory.")
    else:
        print("SECURITY FAILED: Agent outputted a path outside authorized directory!")

    print("\nEVALUATION COMPLETE")

if __name__ == "__main__":
    evaluate_agent_01()