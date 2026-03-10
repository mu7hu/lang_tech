# Limitations of MMLU as a Benchmark

*(Use this section in your written report and reflection to show critical thinking about evaluation methodology.)*

## Multiple-choice bias

MMLU measures the ability to **choose** among four options, not to generate explanations or justify answers. Models may exploit surface cues, keyword overlap, or guessing. Conclusions from this project should be framed as performance **under this multiple-choice benchmark**, not as general reasoning or knowledge.

## Memorization vs reasoning

Performance may reflect **memorization** of similar items or overlap with training data, rather than true reasoning. The report should discuss this limitation and avoid overinterpreting accuracy as “understanding.”

## Ambiguity and benchmark quality

Some MMLU items may be ambiguous or contain errors. A subset of model “errors” could be due to **benchmark quality** rather than model failure. Manual error analysis (using `error_analysis.csv`) can help identify such cases.

## Limits of benchmark-based evaluation

MMLU does not cover all capabilities (e.g., long-form generation, coding, safety, multilingual use). Conclusions from this project apply to **this benchmark and setting** (zero-shot, local models, MMLU subset) and should not be generalized to “general intelligence” or other tasks.

---

*In your report, dedicate a subsection to these points and phrase conclusions as “under this benchmark” to support a stronger grade.*
