import os
import re
import logging
from crewai import Agent, Task, Crew, Process
from config import LOCAL_LLM, STATE_AG02, STATE_AG03

from AG03_Custom_Tools import QuizSaverTool, QuizValidatorTool

# ── Logging ───────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── Tools ────────────────────────────────────────────────
quiz_saver = QuizSaverTool()
quiz_validator = QuizValidatorTool()


# ══════════════════════════════════════════════════════════
# CLEAN FUNCTION (STRONG VERSION)
# ══════════════════════════════════════════════════════════

def clean_output(text: str) -> str:
    """
    Clean LLM output:
    - Remove JSON/tool calls
    - Keep only MCQ section
    - Fix answer format
    """

    # Remove JSON blocks
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)

    # Keep from Q1 only
    match = re.search(r'(Q1\..*)', text, re.DOTALL)
    if match:
        text = match.group(1)

    # Fix weird answer formats
    text = re.sub(
        r'Answer:\s*.*?([A-D])\b',
        r'Answer: \1',
        text
    )

    # Remove extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ══════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════

agent = Agent(
    role="Strict Exam Generator",
    goal="Generate exactly 10 MCQ questions ONLY from given summary.",
    backstory=(
        "You are a STRICT university examiner.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Use ONLY exact facts from the summary.\n"
        "2. DO NOT infer, assume, or add knowledge.\n"
        "3. Every correct answer MUST be directly supported by the summary.\n"
        "4. If unsure → SKIP that idea.\n"
        "5. NEVER generate incorrect answers.\n"
        "6. Output MUST strictly follow format.\n\n"

        "FORMAT RULE:\n"
        "Answer must be EXACTLY one letter:\n"
        "Answer: A\n"
        "Answer: B\n"
        "Answer: C\n"
        "Answer: D\n\n"

        "NO brackets like (A) or A)\n"
        "NO explanations\n"
        "NO JSON\n"
    ),
    tools=[],
    llm=LOCAL_LLM,
    verbose=True,
    allow_delegation=False
)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    if not os.path.exists(STATE_AG02):
        print("[ERROR] Summary file missing")
        exit(1)

    with open(STATE_AG02, "r", encoding="utf-8") as f:
        summary = f.read().strip()

    if not summary:
        print("[ERROR] Empty summary")
        exit(1)

    logging.info(f"[AG03 START] Summary loaded ({len(summary)} chars)")

    print("\n### AGENT 03 START ###")

    task = Task(
    description=f"""
Summary:
{summary}

Generate EXACTLY 10 MCQ questions.

STRICT RULES:

- Use ONLY information explicitly in the summary
- DO NOT create questions from assumptions
- Each correct answer MUST match the summary EXACTLY
- Distractors (wrong options) must be clearly incorrect
- If content is insufficient → repeat concepts differently

STRICT FORMAT:

Q1. Question
A) Option
B) Option
C) Option
D) Option
Answer: X

Repeat until Q10.

CRITICAL:
- Answer must be ONLY A/B/C/D
- No brackets like A)
- No explanations
- No JSON
- Output ONLY quiz
""",
    agent=agent,
    expected_output="Strict MCQ format"
)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    # ── CLEAN OUTPUT ──────────────────────
    quiz_text = clean_output(str(result))

    print("\n=== CLEAN QUIZ ===\n")
    print(quiz_text)

    logging.info("[AG03 OUTPUT] Quiz generated")

    # ── SAVE ─────────────────────────────
    save = quiz_saver.run(content=quiz_text)
    print("\n[SAVE]", save)
    logging.info(save)

    # ── VALIDATE ─────────────────────────
    validate = quiz_validator.run(file_path=STATE_AG03)
    print("\n[VALIDATION]", validate)
    logging.info(validate)

    print("\n### DONE ###")