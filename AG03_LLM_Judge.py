import os
from langchain_ollama import ChatOllama
from config import LOCAL_LLM, STATE_AG02, STATE_AG03

def run_llm_judge():
    print("STARTING LLM JUDGE EVALUATION\n")

    # ── Load Summary ─────────────────────
    if not os.path.exists(STATE_AG02):
        print("ERROR: Summary file missing")
        return

    with open(STATE_AG02, "r", encoding="utf-8") as f:
        summary = f.read().strip()

    # ── Load Quiz ────────────────────────
    if not os.path.exists(STATE_AG03):
        print("ERROR: Quiz file missing")
        return

    with open(STATE_AG03, "r", encoding="utf-8") as f:
        quiz = f.read().strip()

    # ── Initialize LLM ───────────────────
    model_name = LOCAL_LLM.split("/")[-1]  
    llm = ChatOllama(model=model_name)

    # ── Judge Prompt ─────────────────────
    prompt = f"""
You are a strict university exam evaluator.

Evaluate the following quiz based ONLY on the summary.

SUMMARY:
{summary}

QUIZ:
{quiz}

SCORING CRITERIA (Total = 10 marks):
- Format correctness (2 marks)
- Question relevance to summary (3 marks)
- Answer correctness (3 marks)
- Clarity and quality (2 marks)

Rules:
- Be strict
- Do NOT explain too much
- Do NOT rewrite the quiz

Return EXACTLY in this format:

Score: X/10
Status: PASS or FAIL
Issues:
- issue 1
- issue 2
"""

    #  Call LLM 
    response = llm.invoke(prompt)

    #  Extract response content 
    if hasattr(response, "content"):
        result = response.content
    else:
        result = str(response)

    print("\n=== LLM JUDGE RESULT ===\n")
    print(result)

    #  Save report 
    with open("AG03_Judge_Report.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("\nJudge report saved → AG03_Judge_Report.txt")


if __name__ == "__main__":
    run_llm_judge()