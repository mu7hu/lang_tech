#!/usr/bin/env python3
"""
Orchestrate full evaluation: load -> prompt -> infer -> parse -> metrics -> export -> visualize.
Single entrypoint to run the pipeline (FR-5.4, M5).
"""

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_mmlu import load_subjects_config, load_mmlu_subset
from src.prompts import build_prompt
from src.inference import load_models_config, complete
from src.parse_answers import parse_answer
from src.metrics import compute_metrics
from src.export import export_summary_metrics, export_error_analysis
from src.visualize import generate_all_charts


def run_pipeline(
    use_cache: bool = True,
    results_dir: Path | str | None = None,
    skip_inference: bool = False,
    raw_results_path: Path | str | None = None,
) -> None:
    """
    Run the full pipeline. If skip_inference is True, load existing raw results from
    raw_results_path and only recompute metrics, export, and charts.
    """
    results_dir = Path(results_dir or PROJECT_ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if skip_inference and raw_results_path and Path(raw_results_path).exists():
        # Load existing raw CSV and recompute metrics + export + charts
        rows = []
        with open(raw_results_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["correct"] = r.get("correct", "").strip().lower() in ("true", "1", "yes")
                r["is_valid"] = (r.get("parsed_answer") or "").strip().upper() in ("A", "B", "C", "D")
                rows.append(r)
        metrics = compute_metrics(rows)
        export_summary_metrics(metrics, results_dir / "summary_metrics.csv")
        export_error_analysis(rows, results_dir / "error_analysis.csv")
        generate_all_charts(metrics, figures_dir)
        print("Done (metrics + export + charts from existing raw results).")
        return

    # Load MMLU subset
    subj_cfg = load_subjects_config()
    if use_cache:
        from src.load_mmlu import get_mmlu_subset_cached
        items = get_mmlu_subset_cached(
            subjects=subj_cfg.get("subjects"),
            max_items_per_subject=subj_cfg.get("max_items_per_subject", 50),
            seed=subj_cfg.get("seed", 42),
        )
    else:
        items = load_mmlu_subset(
            subjects=subj_cfg.get("subjects"),
            max_items_per_subject=subj_cfg.get("max_items_per_subject", 50),
            seed=subj_cfg.get("seed", 42),
        )
    print(f"Loaded {len(items)} MMLU items.")

    models_cfg = load_models_config()
    models = models_cfg.get("models", [])
    if not models:
        raise ValueError("No models in config/models.yaml")
    system_prompt = models_cfg.get("system_prompt")

    # Stream raw results to CSV so that partial progress is saved even if the run is interrupted.
    rows: list[dict] = []
    raw_path = results_dir / "raw_results.csv"
    fieldnames = [
        "item_id",
        "subject",
        "domain",
        "correct_answer",
        "model",
        "raw_output",
        "parsed_answer",
        "correct",
    ]

    with open(raw_path, "w", newline="", encoding="utf-8") as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=fieldnames)
        writer.writeheader()

        for model in models:
            print(f"Running model: {model}")
            for i, item in enumerate(items):
                prompt = build_prompt(item)
                try:
                    raw_output = complete(prompt, model, system_prompt=system_prompt)
                except Exception as e:
                    raw_output = ""
                    print(f"  Item {item.get('item_id')} error: {e}")
                parsed, is_valid = parse_answer(raw_output)
                correct_answer = item.get("correct_answer", "")
                correct = is_valid and parsed == correct_answer
                row = {
                    "item_id": item.get("item_id"),
                    "subject": item.get("subject"),
                    "domain": item.get("domain"),
                    "correct_answer": correct_answer,
                    "model": model,
                    "raw_output": raw_output,
                    "parsed_answer": parsed if is_valid else "",
                    "is_valid": is_valid,
                    "correct": correct,
                }
                rows.append(row)
                writer.writerow(
                    {
                        "item_id": row["item_id"],
                        "subject": row["subject"],
                        "domain": row["domain"],
                        "correct_answer": row["correct_answer"],
                        "model": row["model"],
                        "raw_output": row["raw_output"],
                        "parsed_answer": row["parsed_answer"],
                        "correct": row["correct"],
                    }
                )
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(items)}")
                # Ensure results are flushed regularly in case the run is interrupted.
                raw_f.flush()

    # Compute metrics and exports from all collected rows.
    metrics = compute_metrics(rows)
    export_summary_metrics(metrics, results_dir / "summary_metrics.csv")
    export_error_analysis(rows, results_dir / "error_analysis.csv")
    generate_all_charts(metrics, figures_dir)
    print(f"Results written to {results_dir}")
    print("Overall accuracy:", {m: f"{metrics['overall'][m]['accuracy']:.4f}" for m in metrics["overall"]})


def main():
    p = argparse.ArgumentParser(description="Run LLM MMLU evaluation pipeline")
    p.add_argument("--no-cache", action="store_true", help="Reload MMLU from source (do not use cached subset)")
    p.add_argument("--results-dir", type=Path, default=None, help="Output directory for results (default: results/)")
    p.add_argument("--skip-inference", action="store_true", help="Skip inference; recompute from existing raw_results.csv")
    p.add_argument("--raw-results", type=Path, default=None, help="Path to raw_results.csv when using --skip-inference")
    args = p.parse_args()
    run_pipeline(
        use_cache=not args.no_cache,
        results_dir=args.results_dir,
        skip_inference=args.skip_inference,
        raw_results_path=args.raw_results or (Path(args.results_dir or PROJECT_ROOT / "results") / "raw_results.csv"),
    )


if __name__ == "__main__":
    main()
