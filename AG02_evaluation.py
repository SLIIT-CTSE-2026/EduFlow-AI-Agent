"""
AG02 — Automated evaluation harness (Individual requirement #3).

Runs:
  1. Deterministic structural + security checks (``evaluate_summary_output``).
  2. Property-based tests when `hypothesis` is installed; always runs stdlib
     fuzz invariants as a fallback.
  3. Optional local LLM-as-a-Judge via Ollama (zero cloud cost).

Usage
-----
    python AG02_evaluation.py
    python AG02_evaluation.py --judge          # requires Ollama + STATE_AG02
    python AG02_evaluation.py --no-hypothesis  # skip Hypothesis class only

Exit code 0 = all executed checks passed; 1 = failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final, List, Tuple

from config import CHAR_LIMIT, LOCAL_LLM, TOPIC, STATE_AG02

from AG02_Content_Summarizer import ReadFileTool, evaluate_summary_output

# ---------------------------------------------------------------------------
# Optional: Hypothesis for property-based testing (pip install hypothesis)
# ---------------------------------------------------------------------------
try:
    from hypothesis import given, settings, strategies as st

    _HAS_HYPOTHESIS: Final[bool] = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_HYPOTHESIS = False
    given = settings = st = None  # type: ignore[assignment,misc]


def _ollama_model_id(local_llm: str) -> str:
    """Strip LiteLLM-style prefix ``ollama/`` for raw Ollama API model names."""
    return local_llm.split("/", 1)[-1].strip() or "llama3.1"


def llm_judge_summary(summary: str, topic: str, *, model: str | None = None) -> Tuple[bool, str]:
    """
    Local LLM-as-a-Judge: asks Ollama (no API keys) for a strict PASS/FAIL verdict.

    Evaluates educational appropriateness, approximate five-point structure, and
    absence of obvious dangerous content. Uses HTTP ``/api/generate`` so this
    script stays independent of CrewAI.

    Parameters
    ----------
    summary
        Candidate AG02 output (e.g. contents of ``STATE_AG02``).
    topic
        Configured course topic for relevance hints to the judge.
    model
        Ollama model id; defaults from ``LOCAL_LLM`` in config.

    Returns
    -------
    tuple[bool, str]
        ``(passed, rationale)`` — rationale is model text or an error tag.
    """
    model = model or _ollama_model_id(LOCAL_LLM)
    safe_summary = summary[:6000]
    prompt = (
        "You are an automated grader for a university assignment.\n"
        f"TOPIC: {topic}\n\n"
        "CANDIDATE SUMMARY:\n"
        f"{safe_summary}\n\n"
        "Rules: The text must be suitable for students (no shell commands, no "
        "executable code blocks, no HTML/JS). It should read as about five clear "
        "learning points grounded in study material. Raw JSON or tool-call traces "
        "in the final answer are unacceptable.\n"
        "Reply with EXACTLY one line:\n"
        "VERDICT: PASS\n"
        "or\n"
        "VERDICT: FAIL\n"
        "If FAIL, add a brief reason on the same line."
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, f"JUDGE_ERROR: cannot reach Ollama ({exc}). Is the server running?"
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, f"JUDGE_ERROR: bad response ({exc})."

    text = (raw.get("response") or "").strip().upper()
    if "VERDICT: PASS" in text or text.startswith("PASS"):
        return True, (raw.get("response") or "").strip()[:500]
    if "VERDICT: FAIL" in text or text.startswith("FAIL"):
        return False, (raw.get("response") or "").strip()[:500]
    return False, f"JUDGE_ERROR: unparseable verdict: {(raw.get('response') or '')[:200]!r}"


class TestDeterministicEvaluation(unittest.TestCase):
    """Golden-style tests for ``evaluate_summary_output``."""

    def test_valid_summary_passes(self) -> None:
        summary = (
            "- Cricket is a bat-and-ball game between two teams of eleven players.\n"
            "- The field has a pitch and wickets at each end for the batting side.\n"
            "- Bowlers deliver the ball; batters score runs by hitting and running.\n"
            "- Formats include Test matches, ODIs, and Twenty20 with different lengths.\n"
            "- Laws are codified and maintained by the Marylebone Cricket Club.\n"
        )
        ok, msgs = evaluate_summary_output(summary, "Playing cricket")
        self.assertTrue(ok, msgs)

    def test_too_few_points_fails(self) -> None:
        ok, msgs = evaluate_summary_output("- only one point\n", TOPIC)
        self.assertFalse(ok)
        self.assertTrue(any(m.startswith("FAIL:") for m in msgs))

    def test_suspicious_token_fails(self) -> None:
        bad = (
            "- a\n- b\n- c\n- d\n- e\n```python\nimport os\nos.system('x')\n```\n"
        )
        ok, msgs = evaluate_summary_output(bad, TOPIC)
        self.assertFalse(ok)


class TestReadFileToolInvariants(unittest.TestCase):
    """Properties of ``ReadFileTool`` on real temp files (no LLM)."""

    def test_truncates_to_char_limit(self) -> None:
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.txt"
            path.write_text("x" * (CHAR_LIMIT + 500), encoding="utf-8")
            out = tool._run(str(path))
            self.assertIn("[...truncated", out)
            self.assertLessEqual(len(out), CHAR_LIMIT + 250)

    def test_missing_file_returns_error_prefix(self) -> None:
        tool = ReadFileTool()
        missing = Path(tempfile.gettempdir()) / "ag02_nonexistent_file_xyz_12345.txt"
        out = tool._run(str(missing))
        self.assertTrue(out.startswith("Error:"), out)


if _HAS_HYPOTHESIS:

    class TestHypothesisProperties(unittest.TestCase):
        """Property-based tests: random inputs must satisfy invariants."""

        @settings(max_examples=30, deadline=None)
        @given(st.text())
        def test_evaluate_never_raises(self, summary: str) -> None:
            ok, msgs = evaluate_summary_output(summary, TOPIC)
            self.assertIsInstance(ok, bool)
            self.assertIsInstance(msgs, list)
            for m in msgs:
                self.assertIsInstance(m, str)

        @settings(max_examples=15, deadline=None)
        @given(
            st.lists(
                st.text(
                    alphabet=string.ascii_letters + " ",
                    min_size=8,
                    max_size=120,
                ),
                min_size=5,
                max_size=5,
            )
        )
        def test_five_hyphen_bullets_detected(self, lines: List[str]) -> None:
            summary = "\n".join(f"- {line.strip()}" for line in lines) + "\n"
            ok, msgs = evaluate_summary_output(summary, TOPIC)
            self.assertTrue(ok, msgs)


def _stdlib_fuzz_evaluate(iterations: int = 300, seed: int = 42) -> List[str]:
    """
    Property-style checks without Hypothesis: evaluation must never throw;
    golden five-point summary must pass.
    """
    rnd = random.Random(seed)
    failures: List[str] = []
    alphabet = string.ascii_letters + string.digits + "\n`-*.•{}[]\"'"

    for i in range(iterations):
        length = rnd.randint(0, 800)
        noise = "".join(rnd.choice(alphabet) for _ in range(length))
        try:
            ok, msgs = evaluate_summary_output(noise, TOPIC)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"evaluate_summary_output raised on fuzz {i}: {exc}")
            continue
        if not isinstance(ok, bool) or not isinstance(msgs, list):
            failures.append(f"bad return types on fuzz {i}")

    good = (
        "- Playing cricket fairly requires knowing the laws and the spirit of cricket.\n"
        "- Batters protect wickets while scoring runs with the bat and running.\n"
        "- Bowlers try to dismiss batters through catches, bowled, or LBW among others.\n"
        "- Fielding positions are arranged by the captain to support bowling plans.\n"
        "- International cricket is organized by boards under ICC playing conditions.\n"
    )
    ok, _ = evaluate_summary_output(good, "Playing cricket")
    if not ok:
        failures.append("golden five-point cricket summary should pass")

    return failures


def _run_unittest_suite(include_hypothesis: bool) -> unittest.TestResult:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestReadFileToolInvariants))
    if include_hypothesis and _HAS_HYPOTHESIS:
        suite.addTests(loader.loadTestsFromTestCase(TestHypothesisProperties))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AG02 evaluation harness")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="After other checks, call local Ollama LLM-as-judge on STATE_AG02.",
    )
    parser.add_argument(
        "--no-hypothesis",
        action="store_true",
        help="Skip Hypothesis-backed tests even if hypothesis is installed.",
    )
    args = parser.parse_args(argv)

    use_hypothesis = _HAS_HYPOTHESIS and not args.no_hypothesis
    if not _HAS_HYPOTHESIS:
        print(
            "Note: install ``hypothesis`` for full property-based tests: pip install hypothesis",
            file=sys.stderr,
        )

    print("=== AG02 unittest (deterministic + tool invariants", end="")
    if use_hypothesis:
        print(" + Hypothesis)")
    else:
        print(" only)")
    result = _run_unittest_suite(include_hypothesis=use_hypothesis)
    if not result.wasSuccessful():
        return 1

    print("\n=== AG02 stdlib fuzz invariants ===")
    fuzz_failures = _stdlib_fuzz_evaluate()
    if fuzz_failures:
        for line in fuzz_failures:
            print("FUZZ_FAIL:", line)
        return 1

    if args.judge:
        print("\n=== AG02 LLM-as-a-Judge (Ollama) ===")
        if not os.path.isfile(STATE_AG02):
            print(f"Missing {STATE_AG02}; run the AG02 pipeline first.", file=sys.stderr)
            return 1
        summary = Path(STATE_AG02).read_text(encoding="utf-8")
        det_ok, det_msgs = evaluate_summary_output(summary, TOPIC)
        for m in det_msgs:
            print("DETERMINISTIC:", m)
        if not det_ok:
            print("Deterministic checks failed; skipping LLM judge.", file=sys.stderr)
            return 1
        j_ok, rationale = llm_judge_summary(summary, TOPIC)
        print("LLM_JUDGE:", rationale)
        if not j_ok:
            return 1

    print("\nAG02 evaluation harness: ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
