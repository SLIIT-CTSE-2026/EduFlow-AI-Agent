import os
import logging
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ── Logging ───────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_trace.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ══════════════════════════════════════════════════════════
# TOOL 1 — Quiz Saver
# ══════════════════════════════════════════════════════════

class QuizSaverInput(BaseModel):
    content: str = Field(..., description="Full MCQ quiz text")


class QuizSaverTool(BaseTool):
    name: str = "quiz_saver_tool"
    description: str = "Save quiz to local file"
    args_schema: Type[BaseModel] = QuizSaverInput

    def _run(self, content: str) -> str:
        from config import STATE_AG03

        logging.info("[TOOL] QuizSaverTool called")

        if not content.strip():
            return "[ERROR] Empty quiz"

        if "Q1." not in content:
            return "[ERROR] Invalid format"

        try:
            with open(STATE_AG03, "w", encoding="utf-8") as f:
                f.write(content)

            return "[SUCCESS] Quiz saved"

        except Exception as e:
            return f"[ERROR] {str(e)}"


# ══════════════════════════════════════════════════════════
# TOOL 2 — Quiz Validator
# ══════════════════════════════════════════════════════════

class QuizValidatorInput(BaseModel):
    file_path: str = Field(...)


class QuizValidatorTool(BaseTool):
    name: str = "quiz_validator_tool"
    description: str = "Validate quiz format"
    args_schema: Type[BaseModel] = QuizValidatorInput

    def _run(self, file_path: str) -> str:

        logging.info("[TOOL] QuizValidatorTool called")

        if not os.path.exists(file_path):
            return "[ERROR] File not found"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        issues = []

        if content.count("Q") < 10:
            issues.append("Not enough questions")

        if "A)" not in content or "B)" not in content or "C)" not in content:
            issues.append("Missing options")

        if "Answer:" not in content:
            issues.append("Missing answers")

        if issues:
            return "[FAILED] " + ", ".join(issues)

        return "[PASSED] Quiz valid"