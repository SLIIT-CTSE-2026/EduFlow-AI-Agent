"""
AG02 — Content Summarizer (Teaching Assistant agent).

Reads the document path produced by AG01 (STATE_AG01), extracts text via a
custom tool, and writes a five-point plain-text summary to STATE_AG02 for
downstream agents (AG03 quiz, AG04 report). Observability and optional
offline evaluation are included for AgentOps / rubric compliance.

Individual rubric mapping (this student’s deliverables)
--------------------------------------------------------
1) **Agent** — ``agent`` below: persona (role/backstory), goals, and
   ``system_template`` constraints tuned for a local SLM (Ollama).
2) **Tool** — ``ReadFileTool`` + ``ReadDocumentInput``: typed Pydantic args,
   docstrings, bounded file I/O for PDF/TXT.
3) **Testing / evaluation** — ``evaluate_summary_output`` (deterministic
   accuracy + security heuristics) and the separate harness
   ``AG02_evaluation.py`` (unittest + optional Hypothesis property tests +
   optional local LLM-as-a-Judge via Ollama). Install optional deps::

       pip install -r requirements-ag02.txt

   Run::

       python AG02_evaluation.py
       python AG02_evaluation.py --judge    # after a successful AG02 run; needs Ollama
       python AG02_Content_Summarizer.py --run-tests
"""

# PEP 563: must be the first non-docstring statement in the module.
from __future__ import annotations

# Public API for imports / report cross-references.
__all__ = [
    "ReadDocumentInput",
    "ReadFileTool",
    "agent",
    "evaluate_summary_output",
    "main",
]

# --- Standard library -------------------------------------------------------
import argparse  # CLI: normal run vs --evaluate-only / --run-tests.
import logging  # Structured traces for AgentOps / debugging.
import os  # Filesystem checks (state files exist).
import re  # Regex for evaluation patterns (bullets, JSON leak).
import subprocess  # Delegate to AG02_evaluation.py without import cycles.
import sys  # stderr handler for logs; exit codes via main().
from pathlib import Path  # Safe path normalization and existence checks.
from typing import Final, List, Tuple, Type  # Strict typing for tool schema and APIs.

# --- Third party (CrewAI + LangChain + Pydantic) ----------------------------
from crewai import Agent, Task  # Agent definition and single-task execution.
from crewai.tools import BaseTool  # Base class for custom Python tools.
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Load PDF/txt into LC Documents.
from pydantic import BaseModel, Field, field_validator  # Tool argument schema + validation.

# --- Project config (shared with other agents; do not rename these keys) ---
from config import CHAR_LIMIT, LOCAL_LLM, TOPIC, STATE_AG01, STATE_AG02

# ---------------------------------------------------------------------------
# Observability — scoped logger (inputs, tool I/O, agent output)
# ---------------------------------------------------------------------------
# One named logger per agent module avoids duplicate handlers on re-import (e.g. tests).
_LOG: Final[logging.Logger] = logging.getLogger("eduflow.ag02")
if not _LOG.handlers:
    # Stream to stderr so normal stdout prints (e.g. "STATE SAVED") stay readable.
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s [AG02] %(message)s")
    )
    _LOG.addHandler(_handler)
# INFO captures lifecycle + tool metrics; use DEBUG locally if you need more noise.
_LOG.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Tool input schema (structured args for the LLM / CrewAI tool router)
# ---------------------------------------------------------------------------
class ReadDocumentInput(BaseModel):
    """Validated arguments for reading a local study document (tool router schema)."""

    # Required string; CrewAI passes tool args as kwargs matching field names.
    file_path: str = Field(
        ...,
        min_length=1,
        description="Absolute or workspace-relative path to a .pdf or .txt file.",
    )

    @field_validator("file_path")
    @classmethod
    def strip_quotes(cls, value: str) -> str:
        """Normalize path string from LLM output (quotes, whitespace)."""
        # LLMs often wrap paths in quotes; strip so Path() resolves correctly.
        return value.strip().strip("'\"")


# ---------------------------------------------------------------------------
# Custom tool — Agent 2’s real-world interaction surface
# ---------------------------------------------------------------------------
class ReadFileTool(BaseTool):
    """
    Loads plain text from a local PDF or text file for summarization.

    This tool is the agent’s primary perception channel: it turns on-disk
    materials into bounded context the SLM can safely reason over.
    """

    # Stable tool name: appears in ReAct traces and must match system prompt wording.
    name: str = "read_file_tool"
    # Shown to the model when choosing tools; keep aligned with CHAR_LIMIT in config.
    description: str = (
        "Reads a local .pdf or .txt file and returns its text, truncated to a "
        f"safe context window (max {CHAR_LIMIT} characters). "
        "Always pass the exact path you were given in the task; do not invent paths."
    )
    # Pydantic model constrains and documents tool parameters for the router.
    args_schema: Type[BaseModel] = ReadDocumentInput

    def _run(self, file_path: str) -> str:
        """
        Read and normalize document text from disk.

        Parameters
        ----------
        file_path
            User- or orchestrator-supplied path after schema validation.

        Returns
        -------
        str
            Concatenated page/line content, truncated to ``CHAR_LIMIT``, or a
            clear error string if the file cannot be read.
        """
        # Defensive strip even after Pydantic (some callers may bypass schema in tests).
        cleaned = file_path.strip().strip("'\"")
        try:
            # expanduser: allow ~/... ; resolve: canonical path; strict=False: allow missing until is_file check.
            resolved = Path(cleaned).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            # Broken symlinks, permission errors during resolve, etc.
            _LOG.warning("Path resolution failed for %r: %s", cleaned, exc)
            return f"Error: invalid path ({exc})"

        if not resolved.is_file():
            # Directory path, typo, or AG01 race: return a string the LLM can reason about.
            _LOG.warning("Not a file or missing: %s", resolved)
            return f"Error: not a readable file: {resolved}"

        # Gate file types we know how to parse; avoids passing binaries to TextLoader.
        suffix = resolved.suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            _LOG.warning("Unsupported extension %s for %s", suffix, resolved)
            return f"Error: unsupported type '{suffix}' (use .pdf or .txt)."

        _LOG.info("Tool read_file_tool: loading %s", resolved)
        try:
            if suffix == ".pdf":
                # PyPDFLoader yields one Document per page; join into one context block.
                docs = PyPDFLoader(str(resolved)).load()
                text = "\n".join(doc.page_content or "" for doc in docs)
            else:
                # Explicit UTF-8 avoids Windows default encoding surprises on study notes.
                text = "\n".join(
                    doc.page_content or ""
                    for doc in TextLoader(str(resolved), encoding="utf-8").load()
                )
        except Exception as exc:  # noqa: BLE001 — surface loader errors to the agent
            # Corrupt PDF, encoding issues, etc.: log stack trace for humans, short message for model.
            _LOG.exception("Document load failed: %s", resolved)
            return f"Error reading file: {exc}"

        # Normalize whitespace-only extracts to empty string for consistent handling.
        text = (text or "").strip()
        if not text:
            _LOG.warning("Empty extract from %s", resolved)
            return "Error: file produced no extractable text."

        if len(text) > CHAR_LIMIT:
            # Hard cap keeps the local SLM within context; trailing marker shows truncation happened.
            text = text[:CHAR_LIMIT] + "\n[...truncated to CHAR_LIMIT for the local SLM...]"

        _LOG.info(
            "Tool read_file_tool: extracted %d chars (cap=%d)",
            len(text),
            CHAR_LIMIT,
        )
        return text


# ---------------------------------------------------------------------------
# Evaluation harness (automated checks on this agent’s final artifact)
# ---------------------------------------------------------------------------
# (?m) = multiline ^/$ ; match bullet or numbered list lines the rubric expects.
_POINT_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^\s*(?:[-*•]|\d{1,2}[\.)])\s+\S.{0,2000}$"
)


def evaluate_summary_output(summary: str, topic: str) -> Tuple[bool, List[str]]:
    """
    Deterministic checks on AG02 output: structure, safety, and usefulness.

    This is intentionally lightweight (no cloud LLM judge) so CI and local
    grading stay zero-cost while still validating constraints from the brief.

    Parameters
    ----------
    summary
        Final plain-text summary written for AG03.
    topic
        Course topic string; used only for a soft topicality hint.

    Returns
    -------
    tuple[bool, list[str]]
        ``(passed, messages)`` where messages collect human-readable findings.
    """
    # Collect human-readable verdict lines; any line starting "FAIL:" fails the run.
    messages: List[str] = []
    text = (summary or "").strip()
    # Pull all lines that look like list items (bullets or 1–99 numbered points).
    points = _POINT_LINE.findall(text)
    if len(points) < 5:
        # Hard requirement for downstream quiz quality and assignment spec.
        messages.append(
            "FAIL: need at least 5 bullet/numbered lines "
            "(e.g. '- point' or '1. point')."
        )
    elif len(text) < 60:
        # Soft signal: model may have padded with ultra-short bullets.
        messages.append("WARN: five points detected but overall text is very terse.")

    # Block obvious injection / exfil patterns if the model echoes untrusted content.
    suspicious = ("```", "__import__", "os.system", "<script", "javascript:")
    for token in suspicious:
        if token.lower() in text.lower():
            messages.append(f"FAIL: suspicious token {token!r} in output (security).")

    # CrewAI ReAct sometimes leaks tool JSON; treat as format failure for AG03 parser safety.
    if re.search(r"\{\s*\"Action\"", text, re.IGNORECASE):
        messages.append("FAIL: looks like raw ReAct/JSON leaked into final output.")

    # Heuristic: long topic words should appear unless the summary uses synonyms only.
    topic_words = [w for w in re.split(r"\W+", topic.lower()) if len(w) > 3]
    lowered = text.lower()
    if topic_words and not any(w in lowered for w in topic_words[:3]):
        messages.append(
            "WARN: summary does not obviously echo the configured topic keywords "
            "(may still be valid if synonyms were used)."
        )

    if len(text) > 20000:
        # Avoid huge artifacts bloating AG03/AG04; warn but do not auto-fail.
        messages.append("WARN: unusually long summary; downstream agents may truncate.")

    # PASS only if no FAIL lines exist (WARN lines are allowed).
    passed = not any(m.startswith("FAIL:") for m in messages)
    if passed:
        # Prepend success banner so humans see green status first in console logs.
        messages.insert(0, "PASS: structural and basic security checks satisfied.")
    return passed, messages


# ---------------------------------------------------------------------------
# Agent definition — tuned for a local SLM via Ollama
# ---------------------------------------------------------------------------
# Single shared tool instance: stateless reads, safe to reuse across tasks.
_READ_TOOL: Final[ReadFileTool] = ReadFileTool()

# Exported as `agent` for Orchestrator: from AG02_Content_Summarizer import agent as analyst
agent = Agent(
    # Persona: instructional designer + TA tone for student-facing bullets.
    role="Senior Teaching Assistant (Content Synthesis)",
    # Goal ties behavior to TOPIC from config (same variable AG04 uses in headings).
    goal=(
        f"Turn the study document into exactly five accurate, student-friendly "
        f"takeaways about {TOPIC}, grounded only in tool-read text."
    ),
    # Backstory steers the SLM away from hallucinating sources not in the tool output.
    backstory=(
        "You specialize in distilling dense readings into five memorable points. "
        "You never invent citations or facts not present in the loaded text. "
        "You write for undergraduates: clear language, no fluff, no markup."
    ),
    tools=[_READ_TOOL],
    # Enough ReAct iterations for: think → tool → observe → final five lines.
    max_iter=4,
    llm=LOCAL_LLM,
    verbose=True,
    # Prevents this worker from delegating to other agents in the Crew definition.
    allow_delegation=False,
    # Ensures system_template is injected into the prompt stack CrewAI builds.
    use_system_prompt=True,
    # Hard constraints for small models: output shape, no JSON, single tool call policy.
    system_template=(
        "You are AG02 in a local multi-agent study pipeline (EduFlow).\n"
        "CONSTRAINTS (must all hold):\n"
        "C1 — Call read_file_tool exactly once using the path from the user task; "
        "never guess paths.\n"
        "C2 — Every bullet must be faithful to the tool output; if the tool returns "
        "an Error: line, output only: Error: could not read source document.\n"
        "C3 — No JSON, YAML, Markdown code fences, XML, or pasted tool traces in "
        "the final answer.\n"
        "C4 — Final answer is exactly five lines: either five '- ' bullets OR "
        "lines '1.' through '5.'; no title line, no closing 'Summary' line.\n"
        "C5 — Plain text only; no HTML or script.\n\n"
        f"Topic context for framing (not for inventing facts): {TOPIC}."
    ),
)


def _run_agent_pipeline(source_path: str) -> str:
    """Execute the CrewAI task and return raw agent output string."""
    # Task bundles: user-facing instructions + schema hint for expected_output.
    task = Task(
        description=(
            # Pin the path literally so the model does not "guess" a different file.
            f"The document path to read is exactly:\n{source_path}\n\n"
            f"Use read_file_tool with that path, then summarize the material for "
            f"students learning about: {TOPIC}.\n"
            # Output contract must stay plain text for AG03 string consumption.
            "Your final answer must be ONLY five concise points about the document "
            "(not about filenames or tools), each on its own line, using either "
            "'- ' bullets or '1. ' … '5. ' numbering. No JSON, no code fences, "
            "no preamble or closing remarks."
        ),
        expected_output=(
            "Exactly five plain-text bullet or numbered lines; nothing else."
        ),
        agent=agent,
    )
    _LOG.info("Invoking agent task (path length=%d)", len(source_path))
    # Blocking call into Ollama-backed LLM via CrewAI; may take tens of seconds.
    result = agent.execute_task(task)
    # CrewAI may return non-str types depending on version; normalize for file write.
    out = str(result).strip()
    _LOG.info("Agent finished; output length=%d", len(out))
    return out


def main(argv: List[str] | None = None) -> int:
    """Entry point: full pipeline (default), disk evaluation, or full test harness."""
    parser = argparse.ArgumentParser(description="AG02 Content Summarizer")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Read STATE_AG02 and run structural/security evaluation (no LLM).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run AG02_evaluation.py (unittest + fuzz + optional Hypothesis; no agent).",
    )
    args = parser.parse_args(argv)

    if args.run_tests:
        # Subprocess keeps evaluation dependencies optional and avoids import cycles.
        script = Path(__file__).resolve().parent / "AG02_evaluation.py"
        proc = subprocess.run([sys.executable, str(script)], check=False)
        return int(proc.returncode)

    if args.evaluate_only:
        # Fast path for CI / marker scripts: validates last written summary on disk.
        if not os.path.isfile(STATE_AG02):
            _LOG.error("Evaluation requested but %s is missing.", STATE_AG02)
            return 1
        with open(STATE_AG02, "r", encoding="utf-8") as handle:
            body = handle.read()
        ok, msgs = evaluate_summary_output(body, TOPIC)
        for line in msgs:
            print(line)
        # Shell exit code: 0 pass, 1 fail (usable in automated test harnesses).
        return 0 if ok else 1

    # Normal pipeline: upstream agent must have written the PDF path string first.
    if not os.path.isfile(STATE_AG01):
        _LOG.error("Missing state file %s — run AG01 first.", STATE_AG01)
        return 1

    with open(STATE_AG01, "r", encoding="utf-8") as handle:
        source_path = handle.read().strip()

    if not source_path:
        # Empty state would send the LLM down a hallucination path; fail fast.
        _LOG.error("State file %s is empty.", STATE_AG01)
        return 1

    _LOG.info("Loaded source path from %s", STATE_AG01)
    result = _run_agent_pipeline(source_path)

    # Global handoff to AG03 / AG04: UTF-8 text file, one artifact per pipeline run.
    with open(STATE_AG02, "w", encoding="utf-8") as handle:
        handle.write(result)

    # Post-run gate: logs findings; does not block saving (instructors can inspect bad runs).
    ok, msgs = evaluate_summary_output(result, TOPIC)
    for line in msgs:
        _LOG.info("Post-run eval: %s", line)
    if not ok:
        _LOG.warning("Post-run evaluation reported failures; artifact still saved.")

    print(f"\nSTATE SAVED TO: {STATE_AG02}")
    return 0


# Script execution (Orchestrator calls: python AG02_Content_Summarizer.py).
if __name__ == "__main__":
    raise SystemExit(main())
