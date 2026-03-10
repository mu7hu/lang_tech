"""
Load MMLU from Hugging Face datasets, apply subject filter and balanced sampling.
FR-1.x: Load from free source; configurable split; expose question, choices, correct answer, subject.
FR-2.x: Balanced subset by subject; deterministic sampling with fixed seed.
FR-3.x: Domain mapping applied in metrics/export; this module only loads subject.
"""

from pathlib import Path

import yaml

# Optional: use datasets from Hugging Face (free, no API key required for public datasets)
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def _load_config(name: str) -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_subjects_config() -> dict:
    """Load config/subjects.yaml (subjects list, max_items_per_subject, seed)."""
    return _load_config("subjects")


def load_subject_to_domain() -> dict[str, str]:
    """Load config/subject_to_domain.yaml and return subject -> domain mapping."""
    raw = _load_config("subject_to_domain")
    mapping = {}
    for domain, subject_list in raw.items():
        if isinstance(subject_list, list):
            for s in subject_list:
                mapping[s] = domain
        elif isinstance(subject_list, str):
            mapping[subject_list] = domain
    return mapping


def load_mmlu_subset(
    dataset_name: str = "cais/mmlu",
    split: str = "test",
    subjects: list[str] | None = None,
    max_items_per_subject: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Load a balanced subset of MMLU.
    Each subject is loaded separately (Hugging Face MMLU uses per-subject configs), then
    we sample up to max_items_per_subject per subject with a fixed seed and concatenate.
    Returns list of items with keys: item_id, question, choices (list of 4 strings),
    correct_answer (letter A/B/C/D), subject.
    """
    if not HAS_DATASETS:
        raise RuntimeError(
            "Package 'datasets' is required. Install with: pip install datasets"
        )

    subjects_config = load_subjects_config()
    if subjects is None:
        subjects = subjects_config.get("subjects", [])
    if not subjects:
        raise ValueError("No subjects specified in config or arguments")
    max_n = max_items_per_subject or subjects_config.get("max_items_per_subject", 50)
    seed = seed or subjects_config.get("seed", 42)

    all_items = []
    item_idx = 0

    for subj in subjects:
        try:
            ds = load_dataset(dataset_name, subj, split=split, trust_remote_code=True)
        except Exception as e:
            # Some MMLU variants use different config names; try with "all" or skip
            raise RuntimeError(
                f"Failed to load MMLU subject '{subj}' from {dataset_name}: {e}"
            ) from e

        n_total = len(ds)
        if n_total == 0:
            continue

        # Deterministic sample: take up to max_n items
        rng = __import__("random").Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        selected = indices[:max_n]

        for i in selected:
            row = ds[int(i)]
            # MMLU cais/mmlu: question, choices (list of 4), answer (int 0-3 for A-D)
            question = row.get("question", "")
            choices = row.get("choices", [])
            if len(choices) != 4:
                choices = (choices + [""] * 4)[:4]
            ans_idx = row.get("answer", 0)
            if isinstance(ans_idx, str):
                correct_letter = str(ans_idx).strip().upper()[:1]
            else:
                correct_letter = ["A", "B", "C", "D"][int(ans_idx) % 4]

            all_items.append({
                "item_id": f"{subj}_{item_idx}",
                "question": question,
                "choices": list(choices),
                "correct_answer": correct_letter,
                "subject": subj,
            })
            item_idx += 1

    # Attach domain from config (FR-3.2)
    try:
        subj_to_domain = load_subject_to_domain()
        for it in all_items:
            it["domain"] = subj_to_domain.get(it["subject"], "Other")
    except FileNotFoundError:
        for it in all_items:
            it["domain"] = "Other"

    return all_items


def get_mmlu_subset_cached(
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
    **kwargs,
) -> list[dict]:
    """
    Load MMLU subset, optionally cache to disk under data/ for faster reruns.
    kwargs are passed to load_mmlu_subset (dataset_name, split, subjects, etc.).
    """
    

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "mmlu_subset.pt"

    if not force_reload and cache_file.exists():
        import pickle
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    items = load_mmlu_subset(**kwargs)
    try:
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(items, f)
    except Exception:
        pass
    return items


if __name__ == "__main__":
    cfg = load_subjects_config()
    items = load_mmlu_subset(
        subjects=cfg.get("subjects"),
        max_items_per_subject=cfg.get("max_items_per_subject", 5),
        seed=cfg.get("seed", 42),
    )
    print(f"Loaded {len(items)} items across {len(set(i['subject'] for i in items))} subjects.")
    if items:
        print("Sample keys:", list(items[0].keys()))
