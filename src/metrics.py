"""
Compute overall, per-subject, per-domain accuracy and invalid output rate. FR-7.x.
"""

from collections import defaultdict


def compute_metrics(rows: list[dict]) -> dict:
    """
    rows: list of dicts with keys: model, subject, domain, correct_answer, parsed_answer, is_valid (bool).
    Returns dict with:
      - overall: { model -> { accuracy, valid_count, total } }
      - per_subject: { model -> { subject -> accuracy } }  (only over valid)
      - per_domain: { model -> { domain -> accuracy } }
      - invalid_rate: { model -> invalid_count / total }
    """
    by_model = defaultdict(lambda: {
        "correct": 0,
        "valid": 0,
        "total": 0,
        "by_subject": defaultdict(lambda: {"correct": 0, "valid": 0}),
        "by_domain": defaultdict(lambda: {"correct": 0, "valid": 0}),
    })

    for r in rows:
        model = r.get("model", "")
        by_model[model]["total"] += 1
        if r.get("is_valid"):
            by_model[model]["valid"] += 1
            if r.get("correct"):
                by_model[model]["correct"] += 1
            subj = r.get("subject", "Other")
            dom = r.get("domain", "Other")
            by_model[model]["by_subject"][subj]["valid"] += 1
            by_model[model]["by_domain"][dom]["valid"] += 1
            if r.get("correct"):
                by_model[model]["by_subject"][subj]["correct"] += 1
                by_model[model]["by_domain"][dom]["correct"] += 1

    result = {
        "overall": {},
        "per_subject": {},
        "per_domain": {},
        "invalid_rate": {},
    }
    for model, data in by_model.items():
        v = data["valid"]
        result["overall"][model] = {
            "accuracy": data["correct"] / v if v else 0.0,
            "valid_count": v,
            "total": data["total"],
        }
        result["invalid_rate"][model] = (data["total"] - v) / data["total"] if data["total"] else 0.0
        result["per_subject"][model] = {
            s: (info["correct"] / info["valid"] if info["valid"] else 0.0)
            for s, info in data["by_subject"].items()
        }
        result["per_domain"][model] = {
            dom: (info["correct"] / info["valid"] if info["valid"] else 0.0)
            for dom, info in data["by_domain"].items()
        }
    return result
