"""
Generate bar charts from summary metrics. FR-9.x.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_overall_accuracy(metrics: dict, out_path: Path | str) -> None:
    """Bar chart of overall accuracy by model (FR-9.1a)."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    overall = metrics.get("overall", {})
    if not overall:
        return
    models = list(overall.keys())
    print(f"DEBUG: Plotting Overall Accuracy. Models found: {list(overall.keys())}")

    accs = [overall[m]["accuracy"] for m in models]
    x = np.arange(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar(x, accs, color=["#2ecc71", "#3498db", "#9b59b6"][: len(models)], edgecolor="black", linewidth=0.5)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Overall accuracy by model")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_subject_accuracy(metrics: dict, out_path: Path | str) -> None:
    """Grouped or grouped bar chart: per-subject accuracy by model (FR-9.1b)."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    per_subject = metrics.get("per_subject", {})
    if not per_subject:
        return
    models = list(per_subject.keys())
    # Collect all subjects
    subjects = sorted(set().union(*(per_subject[m].keys() for m in models)))
    if not subjects:
        return
    n_subj = len(subjects)
    n_models = len(models)
    x = np.arange(n_subj)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(10, n_subj * 0.4), 6))
    for i, model in enumerate(models):
        accs = [per_subject[model].get(s, 0) for s in subjects]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-subject accuracy by model")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_domain_accuracy(metrics: dict, out_path: Path | str) -> None:
    """Bar chart of per-domain accuracy by model (FR-9.1c)."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    per_domain = metrics.get("per_domain", {})
    if not per_domain:
        return
    models = list(per_domain.keys())
    domains = sorted(set().union(*(per_domain[m].keys() for m in models)))
    if not domains:
        return
    n_dom = len(domains)
    n_models = len(models)
    x = np.arange(n_dom)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        accs = [per_domain[model].get(d, 0) for d in domains]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-domain accuracy by model")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_invalid_rate(metrics: dict, out_path: Path | str) -> None:
    """Bar chart of invalid output rate by model (FR-9.1d optional)."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    invalid_rate = metrics.get("invalid_rate", {})
    if not invalid_rate:
        return
    models = list(invalid_rate.keys())
    rates = [invalid_rate[m] for m in models]
    x = np.arange(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar(x, rates, color=["#e74c3c"] * len(models), edgecolor="black", linewidth=0.5)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("Invalid output rate")
    plt.title("Invalid output rate by model")
    y_max = max(rates) if rates else 0
    plt.ylim(0, max(y_max * 1.2, 0.1) if y_max > 0 else 0.1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_charts(metrics: dict, figures_dir: Path | str) -> None:
    """Generate all required charts into figures_dir (FR-9.2, FR-9.3)."""
    figures_dir = Path(figures_dir)
    plot_overall_accuracy(metrics, figures_dir / "overall_accuracy.png")
    plot_per_subject_accuracy(metrics, figures_dir / "per_subject_accuracy.png")
    plot_per_domain_accuracy(metrics, figures_dir / "per_domain_accuracy.png")
    plot_invalid_rate(metrics, figures_dir / "invalid_rate.png")
