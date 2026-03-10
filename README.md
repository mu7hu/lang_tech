# LLM MMLU Evaluation Pipeline

University Language Technology course project: **evaluating free open-source LLMs on a balanced subset of the MMLU benchmark** with a fully local, reproducible, zero-cost pipeline.

## Goals

- Compare three configurable open-source models (default: Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, Mistral 7B Instruct) on MMLU.
- Report overall, per-subject, and per-domain accuracy; invalid output rate; and error analysis.
- Run entirely for **free** (no paid APIs, no usage-based services). Inference via **Ollama** (or compatible local API).

## Requirements

- **Python 3.10+**
- **Ollama** installed and running locally ([ollama.ai](https://ollama.ai))
- Pull the models you want to evaluate, e.g.:
  ```bash
  ollama pull llama3.1:8b-instruct
  ollama pull qwen2.5:7b-instruct
  ollama pull mistral:7b-instruct
  ```

## Setup

```bash
cd /path/to/Langtechproj
pip install -r requirements.txt
```

## Configuration

- **`config/models.yaml`** — Model names (Ollama identifiers), optional system prompt, inference timeout/retries.
- **`config/subjects.yaml`** — MMLU subjects to include, `max_items_per_subject`, and random `seed` for reproducibility.
- **`config/subject_to_domain.yaml`** — Mapping from subjects to domains (STEM, Humanities, Social Science, Other).

Edit these to change models, subset size, or domain grouping.

## Running the Pipeline

**Full run** (load MMLU → run all models → metrics → export → charts):

```bash
python run_pipeline.py
```

**Reload MMLU from source** (no cache):

```bash
python run_pipeline.py --no-cache
```

**Recompute metrics and charts from existing raw results** (no inference):

```bash
python run_pipeline.py --skip-inference --raw-results results/raw_results.csv
```

Outputs are written to **`results/`**:

- `raw_results.csv` — Per-item, per-model: item_id, subject, domain, correct answer, model, raw output, parsed answer, correct.
- `summary_metrics.csv` — Overall accuracy, invalid rate, per-subject and per-domain accuracy.
- `error_analysis.csv` — Incorrect (but parseable) items for manual error categorization. Open in a spreadsheet or notebook to add a column (e.g. `error_category`: wrong reasoning, ambiguous question, format confusion) and summarize patterns in your report (see [docs/benchmark_limitations.md](docs/benchmark_limitations.md)).
- `figures/` — `overall_accuracy.png`, `per_subject_accuracy.png`, `per_domain_accuracy.png`, `invalid_rate.png`.

## Methodology

- **Zero-shot** multiple-choice; identical prompt format for all models; no chain-of-thought.
- **Strict answer parsing**: only A/B/C/D; unparseable outputs count as invalid (reported separately).
- **Balanced subset**: deterministic sampling per subject with fixed seed for reproducibility.

## Zero-Cost Note

This project uses only free, local tools: Hugging Face `datasets` (public MMLU), Ollama (local inference), and standard Python libraries. No API keys or paid services are required.

## License and Course Use

This repository is for educational use as part of a university Language Technology course. MMLU is used under its respective license (see dataset documentation).
