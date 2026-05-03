import os
import logging
from crewai import Agent, Task, Crew, Process
from config import LOCAL_LLM, STATE_AG02, STATE_AG03

# Import tools
from AG03_Custom_Tools import QuizSaverTool, QuizValidatorTool

# ── Initialize Tools ───────────────────────────────────────
quiz_saver_tool = QuizSaverTool()
quiz_validator_tool = QuizValidatorTool()

# ── Logging Setup ──────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── Agent Definition (SIMPLIFIED & FIXED) ───────────────────
agent = Agent(
    role="Examiner",
    goal="Generate exactly 5 MCQ questions from the provided summary.",
    backstory=(
        "You are a university exam setter specializing in agriculture and IoT. "
        "You create clear MCQs strictly based on given content. "
        "Do NOT add explanations or extra text."
    ),
    tools=[],  # ❗ Tools removed (manual execution instead)
    llm=LOCAL_LLM,
    allow_delegation=False,
    verbose=True,
    max_iter=3
)

# ── Main Execution ─────────────────────────────────────────
if __name__ == "__main__":

    # ── Load Summary ───────────────────────────────────────
    if not os.path.exists(STATE_AG02):
        print(f"[ERROR] Summary file not found: {STATE_AG02}")
        exit(1)

    with open(STATE_AG02, "r", encoding="utf-8") as f:
        summary: str = f.read().strip()

    if not summary:
        print("[ERROR] Summary file is empty.")
        exit(1)

    logging.info(f"[AG03 START] Summary loaded | {len(summary)} chars")
    print("\n### AGENT 03: QUIZ GENERATOR STARTING ###")

    # ── Task Definition ────────────────────────────────────
    task = Task(
        description=f"""
Summary:
{summary}

Generate exactly 5 MCQ questions.

STRICT FORMAT:

Q1. ...
A) ...
B) ...
C) ...
Answer: ...

Q2. ...
...

Rules:
- No explanations
- No extra text
- Only the quiz
""",
        expected_output="5 MCQ questions in strict format.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    # ── Run Agent ─────────────────────────────────────────
    result = crew.kickoff()
    quiz_text: str = str(result).strip()

    print("\n[AG03] Generated Quiz\n")
    print(quiz_text)

    logging.info("[AG03 OUTPUT] Quiz generated")

    # ── TOOL 1: SAVE QUIZ ─────────────────────────────────
    logging.info("[AG03 TOOL] Calling QuizSaverTool")
    save_result = quiz_saver_tool.run(content=quiz_text)
    print("\n[SAVE RESULT]")
    print(save_result)
    logging.info(save_result)

    # ── TOOL 2: VALIDATE QUIZ ─────────────────────────────
    logging.info("[AG03 TOOL] Calling QuizValidatorTool")
    validate_result = quiz_validator_tool.run(file_path=STATE_AG03)
    print("\n[VALIDATION RESULT]")
    print(validate_result)
    logging.info(validate_result)

    # ── Final Output ──────────────────────────────────────
    logging.info(f"[AG03 COMPLETE] Saved & validated {STATE_AG03}")

    print("\n=== FINAL QUIZ OUTPUT ===")
    print(quiz_text)