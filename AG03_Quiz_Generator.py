# AG03_Quiz_Generator.py - FIXED

import os
import logging
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from config import LOCAL_LLM, STATE_AG02, STATE_AG03

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class QuizSaverInput(BaseModel):
    """Input schema for QuizSaverTool."""
    content: str = Field(
        ...,
        description="The full plain text of the 5 MCQ questions with A) B) C) options."
    )


class QuizSaverTool(BaseTool):
    """
    Tool for saving generated MCQ quiz content to the local filesystem.

    Writes the complete quiz text to the configured STATE_AG03 output file.
    Used by the Examiner agent as the final step after quiz generation.
    """

    name: str = "quiz_saver_tool"
    description: str = (
        "Saves the complete MCQ quiz text to a local file. "
        "Call this tool with the full quiz text as 'content'. "
        "Do not describe the tool — just call it with the quiz."
    )
    args_schema: Type[BaseModel] = QuizSaverInput

    def _run(self, content: str) -> str:
        """
        Write quiz content to the configured state output file.

        Args:
            content (str): Complete MCQ quiz text with all 5 questions
                          and A), B), C) answer options.

        Returns:
            str: Success confirmation with file path and byte size,
                 or structured error string if write fails.

        Raises:
            Does not raise — all exceptions caught and returned as strings.

        Example:
            >>> tool = QuizSaverTool()
            >>> result = tool._run("Q1. What is IoT?\\nA) Option1\\nB) Option2\\nC) Option3")
            >>> "successfully" in result
            True
        """
        logging.info(f"[TOOL CALL] QuizSaverTool | Preview: {content[:100]}")

        if not content or not content.strip():
            return "[ERROR] Cannot save empty content."

        if len(content.strip()) < 50:
            return "[ERROR] Quiz too short. Provide all 5 questions."

        try:
            with open(STATE_AG03, "w", encoding="utf-8") as f:
                f.write(content)

            file_size: int = os.path.getsize(STATE_AG03)
            logging.info(f"[TOOL SUCCESS] Saved to {STATE_AG03} | {file_size} bytes")
            return f"Quiz saved successfully to {STATE_AG03} ({file_size} bytes)."

        except PermissionError:
            return f"[ERROR] Permission denied writing to {STATE_AG03}."
        except OSError as e:
            return f"[ERROR] File system error: {str(e)}"
        except Exception as e:
            return f"[ERROR] Unexpected error: {str(e)}"


# ── Agent ──────────────────────────────────────────────────
agent = Agent(
    role="Examiner",
    goal=(
        "Generate exactly 5 MCQ questions from the summary. "
        "Format each question as:\n"
        "Q1. [Question]\nA) [Option]\nB) [Option]\nC) [Option]\nAnswer: [Letter]\n\n"
        "Then CALL quiz_saver_tool with the complete quiz text. "
        "Your Final Answer is the quiz text only."
    ),
    backstory=(
        "You are a university exam setter. You write plain text MCQs only. "
        "You NEVER output JSON or code blocks. "
        "You ALWAYS physically call quiz_saver_tool — never just describe it. "
        "Calling the tool means using it as an action, not writing its name in text."
    ),
    tools=[QuizSaverTool()],
    llm=LOCAL_LLM,
    allow_delegation=False,
    verbose=True,
    max_iter=4
)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.path.exists(STATE_AG02):
        print(f"[ERROR] {STATE_AG02} not found.")
        exit(1)

    with open(STATE_AG02, "r", encoding="utf-8") as f:
        summary: str = f.read().strip()

    logging.info(f"[AG03 START] Summary loaded | {len(summary)} chars")

    task = Task(
        description=(
            f"Here is the summary:\n\n{summary}\n\n"
            "Step 1: Write 5 MCQ questions in this exact format:\n"
            "Q1. [Question]\nA) [Option]\nB) [Option]\nC) [Option]\nAnswer: [Letter]\n\n"
            "Step 2: Call quiz_saver_tool with the complete quiz as 'content'.\n"
            "Step 3: Return the quiz text as Final Answer.\n"
            "IMPORTANT: Actually execute the tool. Do not write JSON."
        ),
        expected_output="5 plain text MCQ questions saved to file.",
        agent=agent
    )

    # Use Crew to properly handle tool calls
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    logging.info(f"[AG03 COMPLETE] Saved to {STATE_AG03}")
    print(f"\nSTATE SAVED TO: {STATE_AG03}")
    print("\n=== QUIZ OUTPUT ===")
    print(result)