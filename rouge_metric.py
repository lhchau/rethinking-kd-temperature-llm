"""ROUGE-L and exact match against a list of references (Super-NaturalInstructions evaluation)."""
import string

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_punctuation = set(string.punctuation)


def normalize_answer(s):
    """Lowercase, drop punctuation and collapse whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in _punctuation)
    return " ".join(s.split())


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge(prediction, ground_truth):
    return _scorer.score(prediction=prediction, target=ground_truth)["rougeL"].fmeasure


def compute_metrics(predictions, references):
    """predictions: list[str]; references: list[list[str]]. Scores are in [0, 100]."""
    n = min(len(predictions), len(references))
    em = sum(max(exact_match(p, g) for g in gold) for p, gold in zip(predictions[:n], references[:n]))
    rouge_l = sum(max(rouge(p, g) for g in gold) for p, gold in zip(predictions[:n], references[:n]))
    return {"exact_match": round(100.0 * em / n, 4), "rougeL": round(100.0 * rouge_l / n, 4)}
