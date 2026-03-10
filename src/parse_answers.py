"""
Extract single answer letter A/B/C/D from model output. FR-6.x.
Strict rule: first occurrence of A, B, C, or D in a defined pattern; else invalid.
"""

import re

# Strict rule: look for "Answer: X" or "(X)" or standalone A/B/C/D at start of line or after colon.
# We take the first occurrence of a single letter A, B, C, or D that is clearly the answer.
ANSWER_PATTERN = re.compile(
    r"\b([ABCD])\b",
    re.IGNORECASE,
)


def parse_answer(raw_output: str) -> tuple[str | None, bool]:
    """
    Extract a single answer letter from model output.
    Returns (letter, is_valid). Letter is "A"|"B"|"C"|"D" or None if invalid.
    FR-6.1, FR-6.2: strict rule; no valid A/B/C/D -> invalid.
    """
    if not raw_output or not isinstance(raw_output, str):
        return None, False
    text = raw_output.strip()
    # Prefer "Answer: X" or "answer is X" style
    for pattern in [
        r"(?:answer|Answer)\s*[:\s]+\s*([ABCD])",
        r"([ABCD])\s*[.)]\s*(?:\s|$)",
        r"\b([ABCD])\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in ("A", "B", "C", "D"):
                return letter, True
    # Fallback: first A/B/C/D in text (e.g. "The answer is A.")
    m = ANSWER_PATTERN.search(text)
    if m:
        letter = m.group(1).upper()
        return letter, True
    return None, False
