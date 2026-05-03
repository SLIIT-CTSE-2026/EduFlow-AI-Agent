from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import Field
from config import LOCAL_LLM, TOPIC, STATE_AG02, STATE_AG03, FINAL_REPORT

import os
import re
import json
import logging
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOOL 1 — Table of Contents Generator
# ─────────────────────────────────────────────

class TableOfContentsGeneratorTool(BaseTool):
    """
    Parses Markdown headings (# H1, ## H2, ### H3) from the report content
    and injects a formatted anchor-linked Table of Contents block
    immediately after the document title.
    """

    name: str = "toc_generator_tool"
    description: str = (
        "Parses Markdown headings from report content and injects a formatted "
        "Table of Contents (TOC) after the title. Returns the updated Markdown "
        "string with the TOC inserted. Input must be a Markdown string."
    )

    def _run(self, markdown_content: str) -> str:
        """
        Inject a Table of Contents into a Markdown document.

        Args:
            markdown_content (str): Full Markdown report content.

        Returns:
            str: Markdown with TOC block inserted after the first H1.
        """
        if not markdown_content or not markdown_content.strip():
            return "ERROR: Markdown content is empty — cannot generate TOC."

        lines                           = markdown_content.split("\n")
        heading_pattern                 = re.compile(r"^(#{1,3})\s+(.+)")
        headings: list[tuple[int, str]] = []
        title_line_index: int           = -1

        for i, line in enumerate(lines):
            match = heading_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                text  = match.group(2).strip()
                if level == 1 and title_line_index == -1:
                    title_line_index = i
                    continue
                headings.append((level, text))

        if not headings:
            return markdown_content

        def slugify(text: str) -> str:
            return re.sub(r"[^\w\-]", "", text.lower().replace(" ", "-"))

        toc_lines: list[str] = ["## Table of Contents\n"]
        for level, text in headings:
            indent = "  " * (level - 2)
            slug   = slugify(text)
            toc_lines.append(f"{indent}- [{text}](#{slug})")

        toc_block = "\n".join(toc_lines) + "\n\n---\n"
        insert_at = title_line_index + 1 if title_line_index != -1 else 0
        lines.insert(insert_at, "\n" + toc_block)

        logger.info("TOC generated with %d entries.", len(headings))
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL 2 — PDF Export Tool
# ─────────────────────────────────────────────

class PDFExportTool(BaseTool):
    """
    Converts a Markdown-formatted study report into a professionally styled
    PDF with EduFlow AI Agent branding on every page.
    """

    name: str = "pdf_export_tool"
    description: str = (
        "Converts a Markdown report string into a branded EduFlow AI Agent PDF. "
        "Returns the output PDF path on success or an error message on failure."
    )
    output_path: str = Field(default=FINAL_REPORT.replace(".md", ".pdf"))

    @staticmethod
    def _clean(line: str) -> str:
        """Strip leading '* ' markers the SLM sometimes injects."""
        return re.sub(r"^\*\s+", "", line)

    @staticmethod
    def _md_inline_to_rl(text: str) -> str:
        """
        Convert Markdown inline markers to ReportLab XML tags.

        Args:
            text (str): A single line of Markdown text.

        Returns:
            str: Line with **bold** and *italic* substituted, & escaped.
        """
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
        text = re.sub(r"&(?!amp;|lt;|gt;|quot;)", "&amp;", text)
        return text

    def _build_styles(self) -> dict:
        """
        Build named ReportLab ParagraphStyles with EduFlow branding.

        Returns:
            dict[str, ParagraphStyle]: Style map for all document elements.
        """
        base         = getSampleStyleSheet()
        BRAND_DARK   = colors.HexColor("#1B2A4A")
        BRAND_MID    = colors.HexColor("#2E6DA4")
        BODY_GRAY    = colors.HexColor("#333333")

        return {
            "DocTitle": ParagraphStyle(
                "DocTitle", parent=base["Title"],
                fontSize=20, leading=26,
                alignment=TA_CENTER, spaceAfter=6,
                textColor=colors.white,
            ),
            "SectionHeading": ParagraphStyle(
                "SectionHeading", parent=base["Heading1"],
                fontSize=13, leading=17,
                spaceBefore=14, spaceAfter=5,
                textColor=BRAND_DARK,
            ),
            "SubHeading": ParagraphStyle(
                "SubHeading", parent=base["Heading2"],
                fontSize=11, leading=15,
                spaceBefore=8, spaceAfter=3,
                textColor=BRAND_MID,
            ),
            "Body": ParagraphStyle(
                "Body", parent=base["Normal"],
                fontSize=10, leading=15,
                spaceAfter=5, alignment=TA_LEFT,
                textColor=BODY_GRAY,
            ),
            "BulletItem": ParagraphStyle(
                "BulletItem", parent=base["Normal"],
                fontSize=10, leading=14,
                leftIndent=16, spaceAfter=2,
                textColor=BODY_GRAY,
            ),
            "QuestionText": ParagraphStyle(
                "QuestionText", parent=base["Normal"],
                fontSize=10, leading=15,
                spaceBefore=10, spaceAfter=3,
                textColor=BRAND_DARK,
                fontName="Helvetica-Bold",
            ),
            "OptionText": ParagraphStyle(
                "OptionText", parent=base["Normal"],
                fontSize=10, leading=13,
                leftIndent=20, spaceAfter=1,
                textColor=BODY_GRAY,
            ),
            "_BRAND_DARK": BRAND_DARK,
            "_BRAND_MID":  BRAND_MID,
        }

    def _header_footer(self, canvas, doc) -> None:
        """
        Draw EduFlow AI Agent branded header and footer on every page.

        Args:
            canvas: ReportLab canvas object.
            doc:    ReportLab document object.
        """
        canvas.saveState()
        width, height = A4

        # Header band
        canvas.setFillColor(colors.HexColor("#1B2A4A"))
        canvas.rect(0, height - 28*mm, width, 28*mm, fill=1, stroke=0)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(20*mm, height - 14*mm, "EduFlow AI Agent")

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#A8C8E8"))
        canvas.drawString(20*mm, height - 21*mm, f"Study Report  ·  {TOPIC}")

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - 20*mm, height - 14*mm, f"Page {doc.page}")

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#A8C8E8"))
        canvas.drawRightString(
            width - 20*mm, height - 21*mm,
            datetime.now().strftime("%B %d, %Y"),
        )

        # Footer band
        canvas.setFillColor(colors.HexColor("#F0F4F8"))
        canvas.rect(0, 0, width, 12*mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawCentredString(
            width / 2, 4*mm,
            "Generated by EduFlow AI Agent  ·  SE4010 CTSE Assignment 2  "
            "·  Sri Lanka Institute of Information Technology",
        )
        canvas.restoreState()

    def _run(self, markdown_content: str) -> str:
        """
        Parse the Markdown string and render a branded PDF via ReportLab.

        Args:
            markdown_content (str): Full Markdown report to convert.

        Returns:
            str: Success message with PDF path/size, or error description.
        """
        if not markdown_content or not markdown_content.strip():
            return "ERROR: Markdown content is empty — PDF not created."

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            styles         = self._build_styles()
            story:         list      = []
            bullet_buffer: list[str] = []
            numbered_buf:  list[str] = []

            # Quiz state
            in_quiz         = False
            q_number        = 0
            current_q_text  = ""
            current_options: list[str] = []

            def flush_bullets() -> None:
                if bullet_buffer:
                    items = [
                        ListItem(
                            Paragraph(self._md_inline_to_rl(b), styles["BulletItem"]),
                            bulletColor=styles["_BRAND_MID"],
                        )
                        for b in bullet_buffer
                    ]
                    story.append(ListFlowable(items, bulletType="bullet", leftIndent=20))
                    story.append(Spacer(1, 4))
                    bullet_buffer.clear()

            def flush_numbered() -> None:
                if numbered_buf:
                    items = [
                        ListItem(
                            Paragraph(self._md_inline_to_rl(n), styles["BulletItem"]),
                            bulletColor=styles["_BRAND_MID"],
                        )
                        for n in numbered_buf
                    ]
                    story.append(ListFlowable(items, bulletType="1", leftIndent=20))
                    story.append(Spacer(1, 4))
                    numbered_buf.clear()

            def flush_question() -> None:
                """Render buffered quiz question with options only — no answers."""
                nonlocal current_q_text, current_options, q_number
                if not current_q_text.strip():
                    return
                story.append(Paragraph(
                    self._md_inline_to_rl(f"Q{q_number}. {current_q_text}"),
                    styles["QuestionText"],
                ))
                for opt in current_options:
                    story.append(Paragraph(
                        self._md_inline_to_rl(opt), styles["OptionText"]
                    ))
                story.append(Spacer(1, 8))
                current_q_text  = ""
                current_options = []

            for raw_line in markdown_content.split("\n"):
                line = self._clean(raw_line.rstrip())

                # ── Detect Quiz section heading ──
                if re.match(r"^#{1,4}\s+Quiz", line, re.IGNORECASE):
                    flush_bullets(); flush_numbered(); flush_question()
                    in_quiz = True
                    story.append(Spacer(1, 6))
                    story.append(Paragraph("Quiz", styles["SectionHeading"]))
                    story.append(HRFlowable(
                        width="100%", thickness=1,
                        color=colors.HexColor("#2E6DA4"), spaceAfter=6,
                    ))
                    continue

                # ── H3 ──
                if line.startswith("### "):
                    flush_bullets(); flush_numbered(); flush_question()
                    story.append(Paragraph(
                        self._md_inline_to_rl(line[4:]), styles["SubHeading"]
                    ))

                # ── H2 ──
                elif line.startswith("## "):
                    flush_bullets(); flush_numbered(); flush_question()
                    heading_text = line[3:].strip()
                    if heading_text.lower() == "table of contents":
                        continue
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(
                        self._md_inline_to_rl(heading_text), styles["SectionHeading"]
                    ))
                    story.append(HRFlowable(
                        width="100%", thickness=1,
                        color=colors.HexColor("#2E6DA4"), spaceAfter=4,
                    ))

                # ── H1 ──
                elif line.startswith("# "):
                    flush_bullets(); flush_numbered(); flush_question()
                    story.append(Paragraph(
                        self._md_inline_to_rl(line[2:]), styles["DocTitle"]
                    ))
                    story.append(Spacer(1, 4))

                # ── Horizontal rule ──
                elif re.match(r"^-{3,}$", line.strip()):
                    flush_bullets(); flush_numbered(); flush_question()
                    story.append(HRFlowable(
                        width="100%", thickness=0.5,
                        color=colors.HexColor("#CCCCCC"), spaceAfter=6,
                    ))

                # ── Quiz: numbered question ──
                elif in_quiz and re.match(r"^\d+[\.\)]\s+", line):
                    flush_question()
                    q_number += 1
                    current_q_text = re.sub(r"^\d+[\.\)]\s+", "", line).strip()

                # ── Quiz: bullet question ──
                elif in_quiz and re.match(r"^[•]\s+", line):
                    flush_question()
                    q_number += 1
                    current_q_text = re.sub(r"^[•]\s+", "", line).strip()

                # ── Quiz: option line A) B) C) D) ──
                elif in_quiz and re.match(r"^[A-Da-d][\.\)]\s+", line):
                    current_options.append(line.strip())

                # ── Quiz: continuation of question text ──
                elif in_quiz and current_q_text and line.strip():
                    current_q_text += " " + line.strip()

                # ── Regular bullet ──
                elif not in_quiz and re.match(r"^[-•]\s+", line):
                    flush_numbered()
                    bullet_buffer.append(re.sub(r"^[-•]\s+", "", line))

                # ── Regular numbered list ──
                elif not in_quiz and re.match(r"^\d+\.\s+", line):
                    flush_bullets()
                    numbered_buf.append(re.sub(r"^\d+\.\s+", "", line))

                # ── HTML comment ──
                elif line.strip().startswith("<!--"):
                    continue

                # ── Blank line ──
                elif not line.strip():
                    if not in_quiz:
                        flush_bullets(); flush_numbered()
                        story.append(Spacer(1, 5))

                # ── Plain paragraph ──
                else:
                    if not in_quiz:
                        flush_bullets(); flush_numbered()
                        story.append(Paragraph(
                            self._md_inline_to_rl(line), styles["Body"]
                        ))

            # Flush remaining
            flush_bullets()
            flush_numbered()
            flush_question()

            doc = SimpleDocTemplate(
                self.output_path, pagesize=A4,
                leftMargin=20*mm, rightMargin=20*mm,
                topMargin=32*mm, bottomMargin=18*mm,
                title=f"{TOPIC} Study Report",
                author="EduFlow AI Agent",
            )
            doc.build(
                story,
                onFirstPage=self._header_footer,
                onLaterPages=self._header_footer,
            )

            file_size = os.path.getsize(self.output_path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("PDF written to %s (%d bytes).", self.output_path, file_size)
            return (
                f"SUCCESS: PDF exported to '{self.output_path}' "
                f"({file_size} bytes) at {timestamp}."
            )

        except PermissionError:
            return f"ERROR: Permission denied — '{self.output_path}'."
        except OSError as e:
            return f"ERROR: OS error — {str(e)}"
        except Exception as e:
            logger.exception("Unexpected error during PDF export.")
            return f"ERROR: PDF generation failed — {str(e)}"


# ─────────────────────────────────────────────
# TOOL 3 — Markdown File Writer
# ─────────────────────────────────────────────

class ReportCompilerTool(BaseTool):
    """
    Persists the final Markdown report string to a local .md file.
    Validates content, creates output directories, injects a timestamp header.
    """

    name: str = "report_compiler_tool"
    description: str = (
        "Writes the final formatted Markdown report to a local .md file. "
        "Returns the output file path on success or an error message on failure."
    )
    output_path: str = Field(default=FINAL_REPORT)

    def _run(self, report_content: str) -> str:
        """
        Write Markdown content to the configured output path.

        Args:
            report_content (str): Non-empty Markdown string to persist.

        Returns:
            str: Success message with path and size, or descriptive error string.
        """
        if not report_content or not report_content.strip():
            return "ERROR: Report content is empty."
        if len(report_content.strip()) < 50:
            return "ERROR: Report content is too short."

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_content = (
                f"<!-- Generated by EduFlow AI Agent: {timestamp} -->\n\n"
                + report_content
            )

            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            file_size = os.path.getsize(self.output_path)
            logger.info("Markdown saved to %s (%d bytes).", self.output_path, file_size)
            return f"SUCCESS: Report saved to '{self.output_path}' ({file_size} bytes)."

        except PermissionError:
            return f"ERROR: Permission denied — '{self.output_path}'."
        except OSError as e:
            return f"ERROR: Could not write — {str(e)}"


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

agent = Agent(
    role="Report Coordinator",
    goal=(
        "Synthesize a research summary and quiz into a well-structured "
        "Markdown study report, inject a Table of Contents, save the .md file, "
        "and export a branded EduFlow AI Agent PDF."
    ),
    backstory=(
        "You are a meticulous academic editor for EduFlow AI Agent. "
        "You never invent facts. You always produce valid Markdown. "
        "Your final output is always both a .md file and a .pdf file."
    ),
    llm=LOCAL_LLM,
    tools=[
        TableOfContentsGeneratorTool(),
        PDFExportTool(),
        ReportCompilerTool(),
    ],
    verbose=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_raw(text: str) -> str:
    """Strip '* ' prefixes the SLM injects on every line."""
    return "\n".join(re.sub(r"^\*\s+", "", l) for l in text.split("\n"))


def build_markdown_from_scratch(summary: str, quiz: str) -> str:
    """
    Build a clean Markdown report directly from the state files.

    Args:
        summary (str): Raw summary text from STATE_AG02.
        quiz    (str): Raw quiz text from STATE_AG03.

    Returns:
        str: Well-structured Markdown report string.
    """
    lines = [f"# {TOPIC} Study Report", "", "## Summary", ""]

    for line in summary.split("\n"):
        line = re.sub(r"^\*\s+", "", line.strip())
        if line:
            lines.append(line)

    lines += ["", "---", "", "## Quiz", ""]

    q_num = 0
    for line in quiz.split("\n"):
        line = re.sub(r"^\*\s+", "", line.strip())
        if not line:
            continue
        if re.match(r"^(\d+[\.\)]|[•])\s+", line):
            q_num += 1
            text = re.sub(r"^(\d+[\.\)]|[•])\s+", "", line)
            lines.append(f"\n{q_num}. {text}")
        elif re.match(r"^[A-Da-d][\.\)]\s+", line):
            lines.append(f"   {line}")
        else:
            lines.append(line)

    return "\n".join(lines)


def extract_markdown(raw: str) -> str:
    """
    Extract clean Markdown from whatever the agent returned.

    Args:
        raw (str): Raw agent output string.

    Returns:
        str: Clean Markdown string.
    """
    raw = clean_raw(raw.strip())

    if raw.startswith("#"):
        return raw

    try:
        clean  = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        data   = json.loads(clean)
        params = data.get("parameters", data)

        title   = clean_raw(params.get("title",   f"# {TOPIC} Study Report"))
        summary = [clean_raw(s) for s in params.get("summary", [])]
        quiz    = [clean_raw(q) for q in params.get("quiz",    [])]

        lines = [title, ""] + summary + ["", "---", "", "## Quiz", ""] + quiz
        logger.info("Reconstructed Markdown from agent JSON.")
        return "\n".join(lines)

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    logger.warning("Falling back to plain wrap.")
    return f"# {TOPIC} Study Report\n\n{raw}"


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(STATE_AG02, "r", encoding="utf-8") as f:
        summary = f.read().strip()

    with open(STATE_AG03, "r", encoding="utf-8") as f:
        quiz = f.read().strip()

    combined_prompt = f"""
You are given a topic summary and a quiz. Follow these steps IN ORDER:

STEP 1 — Draft a Markdown report:
  # {TOPIC} Study Report
  ## Summary
  (summary as bullet points)
  ## Quiz
  (numbered MCQs, each option on its own line)

STEP 2 — Call toc_generator_tool with the Markdown.
STEP 3 — Call report_compiler_tool with the TOC-enriched Markdown.
STEP 4 — Call pdf_export_tool with the same Markdown.

RULES: Do not invent facts. Number every quiz question. One option per line.

TOPIC: {TOPIC}
SUMMARY: {summary}
QUIZ: {quiz}
    """

    task = Task(
        description=(
            f"Produce the '{TOPIC}' study report: "
            "draft Markdown → inject TOC → save .md → export PDF."
        ),
        expected_output=(
            "A .md file and a branded EduFlow PDF both confirmed saved on disk."
        ),
        agent=agent,
    )

    # ── Run agent (best effort) ──────────────────────────────────────────
    raw_output = agent.execute_task(task, context=combined_prompt)

    # ── GUARANTEED FALLBACK — always runs regardless of agent behaviour ──
    logger.info("Running guaranteed fallback tool pipeline...")

    final_md = extract_markdown(raw_output)

    if len(final_md.strip()) < 100:
        logger.warning("Agent output too short — rebuilding from state files.")
        final_md = build_markdown_from_scratch(summary, quiz)

    # Step 1: Inject TOC
    toc_tool   = TableOfContentsGeneratorTool()
    toc_result = toc_tool._run(final_md)
    final_md   = toc_result if not toc_result.startswith("ERROR") else final_md

    # Step 2: Save .md
    md_tool   = ReportCompilerTool()
    md_result = md_tool._run(final_md)
    print("\n✅", md_result)

    # Step 3: Export PDF
    pdf_tool   = PDFExportTool()
    pdf_result = pdf_tool._run(final_md)
    print("✅", pdf_result)