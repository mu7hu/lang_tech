"""
Export raw results, summary metrics, and error-analysis CSVs. FR-8.x, FR-10.1.
"""

import csv
from pathlib import Path


def export_raw_results(rows: list[dict], out_path: Path | str) -> None:
    """
    FR-8.1: item_id, subject, domain, question (or id), correct answer, model name,
    model output (raw), parsed answer, correct (boolean).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "item_id", "subject", "domain", "correct_answer", "model",
        "raw_output", "parsed_answer", "correct",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames}
            row["correct"] = bool(r.get("correct")) if r.get("correct") is not None else ""
            w.writerow(row)


def export_summary_metrics(metrics: dict, out_path: Path | str) -> None:
    """
    FR-8.2: model, overall_accuracy, invalid_rate, per-subject and per-domain accuracy.
    We write: one row per model with overall + invalid_rate; then separate tables/sections
    for per_subject and per_domain (long format: model, subject_or_domain, accuracy).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overall = metrics.get("overall", {})
    invalid_rate = metrics.get("invalid_rate", {})
    per_subject = metrics.get("per_subject", {})
    per_domain = metrics.get("per_domain", {})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "overall_accuracy", "invalid_rate", "valid_count", "total"])
        for model in overall:
            o = overall[model]
            w.writerow([
                model,
                f"{o['accuracy']:.4f}",
                f"{invalid_rate.get(model, 0):.4f}",
                o.get("valid_count", ""),
                o.get("total", ""),
            ])
        f.write("\n")
        w.writerow(["model", "subject", "accuracy"])
        for model, subj_acc in per_subject.items():
            for subj, acc in subj_acc.items():
                w.writerow([model, subj, f"{acc:.4f}"])
        f.write("\n")
        w.writerow(["model", "domain", "accuracy"])
        for model, dom_acc in per_domain.items():
            for dom, acc in dom_acc.items():
                w.writerow([model, dom, f"{acc:.4f}"])


def export_error_analysis(rows: list[dict], out_path: Path | str) -> None:
    """
    FR-8.3, FR-10.1: only incorrect items; item_id, subject, domain, correct answer,
    model(s), parsed answer(s), optional raw output snippet for manual categorization.
    rows: raw result rows; we filter to correct==False and is_valid==True (wrong but parseable).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors = [r for r in rows if r.get("is_valid") and r.get("correct") is False]
    if not errors:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("item_id,subject,domain,correct_answer,model,parsed_answer,raw_output_snippet\n")
        return
    fieldnames = [
        "item_id", "subject", "domain", "correct_answer", "model",
        "parsed_answer", "raw_output_snippet",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in errors:
            snippet = (r.get("raw_output") or "")[:500]
            w.writerow({
                "item_id": r.get("item_id"),
                "subject": r.get("subject"),
                "domain": r.get("domain"),
                "correct_answer": r.get("correct_answer"),
                "model": r.get("model"),
                "parsed_answer": r.get("parsed_answer"),
                "raw_output_snippet": snippet,
            })
