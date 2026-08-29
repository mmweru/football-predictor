"""
Evaluation metrics for match outcome predictions.

Accuracy alone is misleading for probabilistic predictions (see the
roadmap notes on calibration) — log loss and Brier score are the primary
metrics used throughout this project. All three are computed here so every
model (Elo baseline, naive baseline, XGBoost) is scored identically and
comparably.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from app.baselines import OUTCOME_LABELS


def probabilities_to_matrix(prob_dicts: Sequence[dict]) -> np.ndarray:
    """Converts a list of {'H': .., 'D': .., 'A': ..} dicts into an (n, 3) array, columns in OUTCOME_LABELS order."""
    return np.array([[d[label] for label in OUTCOME_LABELS] for d in prob_dicts])


def brier_score_multiclass(y_true: Sequence[str], prob_matrix: np.ndarray) -> float:
    """
    Multiclass Brier score: mean squared distance between the predicted
    probability vector and the one-hot true outcome, averaged over samples.
    Lower is better; 0 is a perfect, fully-confident-and-correct model.

    sklearn's brier_score_loss is binary-only, so this is implemented
    directly rather than misusing that function.
    """
    n = len(y_true)
    one_hot = np.zeros((n, len(OUTCOME_LABELS)))
    for i, label in enumerate(y_true):
        one_hot[i, OUTCOME_LABELS.index(label)] = 1.0
    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def evaluate_predictions(y_true: Sequence[str], prob_dicts: Sequence[dict]) -> dict:
    """
    Scores a set of predictions (as probability dicts) against true outcomes.
    Returns log_loss, brier_score, and accuracy (using the argmax of each
    prediction as the "hard" predicted class for accuracy).
    """
    prob_matrix = probabilities_to_matrix(prob_dicts)
    predicted_labels = [OUTCOME_LABELS[i] for i in np.argmax(prob_matrix, axis=1)]

    return {
        "log_loss": log_loss(y_true, prob_matrix, labels=OUTCOME_LABELS),
        "brier_score": brier_score_multiclass(y_true, prob_matrix),
        "accuracy": accuracy_score(y_true, predicted_labels),
        "n_samples": len(y_true),
    }


def print_evaluation(name: str, metrics: dict) -> None:
    print(
        f"  {name:20s}  log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier_score']:.4f}  accuracy={metrics['accuracy']:.3f}  "
        f"(n={metrics['n_samples']})"
    )
