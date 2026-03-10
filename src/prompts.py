"""
Standardized prompt generation for MMLU (zero-shot, answer-only).
FR-4.1, FR-4.2, FR-4.3: Single documented format, identical for all models, no CoT.
"""

# Documented prompt template. Used identically for every model (FR-4.2).
PROMPT_TEMPLATE = """Question: {question}

Choices:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer with one letter: A, B, C, or D."""


def build_prompt(item: dict) -> str:
    """
    Build one prompt per MMLU item. item must have keys: question, choices (list of 4).
    """
    question = item.get("question", "")
    choices = item.get("choices", ["", "", "", ""])
    choice_a = choices[0] if len(choices) > 0 else ""
    choice_b = choices[1] if len(choices) > 1 else ""
    choice_c = choices[2] if len(choices) > 2 else ""
    choice_d = choices[3] if len(choices) > 3 else ""
    return PROMPT_TEMPLATE.format(
        question=question,
        choice_a=choice_a,
        choice_b=choice_b,
        choice_c=choice_c,
        choice_d=choice_d,
    )
