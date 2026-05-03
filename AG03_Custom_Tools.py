import os
import logging
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ── Logging Setup ──────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ══════════════════════════════════════════════════════════
# TOOL 1 — Quiz Saver Tool
# ══════════════════════════════════════════════════════════

class QuizSaverInput(BaseModel):
    """Input schema for QuizSaverTool."""
    content: str = Field(..., description="Full MCQ quiz text")


class QuizSaverTool(BaseTool):
    """
    Saves MCQ quiz content to local file.
    """

    name: str = "quiz_saver_tool"
    description: str = "Saves quiz text to a local file"
    args_schema: Type[BaseModel] = QuizSaverInput

    def _run(self, content: str) -> str:
        from config import STATE_AG03

        logging.info(f"[TOOL CALL] QuizSaverTool")

        # ── Validation ─────────────────────────
        if not content or not content.strip():
            return "[ERROR] Empty quiz content"

        if len(content) < 50:
            return "[ERROR] Quiz content too short"

        try:
            with open(STATE_AG03, "w", encoding="utf-8") as f:
                f.write(content)

            size = os.path.getsize(STATE_AG03)

            success_msg = f"[SUCCESS] Quiz saved → {STATE_AG03} ({size} bytes)"
            logging.info(success_msg)
            return success_msg

        except Exception as e:
            error = f"[ERROR] Save failed: {str(e)}"
            logging.error(error)
            return error


# ══════════════════════════════════════════════════════════
# TOOL 2 — Quiz Validator Tool
# ══════════════════════════════════════════════════════════

class QuizValidatorInput(BaseModel):
    """Input schema for QuizValidatorTool."""
    file_path: str = Field(..., description="Path to quiz file")


class QuizValidatorTool(BaseTool):
    """
    Validates structure of saved quiz file.
    """

    name: str = "quiz_validator_tool"
    description: str = "Validates quiz format"
    args_schema: Type[BaseModel] = QuizValidatorInput

    def _run(self, file_path: str) -> str:

        logging.info(f"[TOOL CALL] QuizValidatorTool")

        if not os.path.exists(file_path):
            return f"[ERROR] File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # ── Checks ─────────────────────────
            q_count = content.count("Q")
            has_a = "A)" in content
            has_b = "B)" in content
            has_c = "C)" in content
            has_ans = "Answer:" in content

            issues = []

            if q_count < 5:
                issues.append(f"Only {q_count} questions found")
            if not (has_a and has_b and has_c):
                issues.append("Missing A/B/C options")
            if not has_ans:
                issues.append("Missing answers")

            if issues:
                error_msg = f"[FAILED] {', '.join(issues)}"
                logging.warning(error_msg)
                return error_msg

            success_msg = "[PASSED] Quiz format is valid"
            logging.info(success_msg)
            return success_msg

        except Exception as e:
            error = f"[ERROR] Validation failed: {str(e)}"
            logging.error(error)
            return error