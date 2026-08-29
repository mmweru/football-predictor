"""
Tests for app.evaluation — verified against hand-computed expected values.
"""

import numpy as np
import pytest

from app.evaluation import (
    brier_score_multiclass,
    evaluate_predictions,
    probabilities_to_matrix,
)


def test_probabilities_to_matrix_column_order():
    dicts = [{"H": 0.5, "D": 0.3, "A": 0.2}]
    matrix = probabilities_to_matrix(dicts)
    # OUTCOME_LABELS = ["A", "D", "H"] — columns must follow that order.
    assert matrix[0].tolist() == pytest.approx([0.2, 0.3, 0.5])


def test_brier_score_perfect_prediction_is_zero():
    y_true = ["H"]
    prob_matrix = np.array([[0.0, 0.0, 1.0]])  # [A, D, H] — fully confident, correct
    assert brier_score_multiclass(y_true, prob_matrix) == pytest.approx(0.0)


def test_brier_score_maximally_wrong_prediction():
    y_true = ["H"]
    prob_matrix = np.array([[1.0, 0.0, 0.0]])  # fully confident in A, but actual was H
    # Squared distance: (1-0)^2 + (0-0)^2 + (0-1)^2 = 2.0
    assert brier_score_multiclass(y_true, prob_matrix) == pytest.approx(2.0)


def test_brier_score_uniform_uncertain_prediction():
    y_true = ["H"]
    prob_matrix = np.array([[1 / 3, 1 / 3, 1 / 3]])
    # (1/3)^2 + (1/3)^2 + (2/3)^2 = 1/9 + 1/9 + 4/9 = 6/9 = 0.6667
    assert brier_score_multiclass(y_true, prob_matrix) == pytest.approx(6 / 9)


def test_evaluate_predictions_perfect_model():
    y_true = ["H", "D", "A"]
    prob_dicts = [
        {"H": 1.0, "D": 0.0, "A": 0.0},
        {"H": 0.0, "D": 1.0, "A": 0.0},
        {"H": 0.0, "D": 0.0, "A": 1.0},
    ]
    metrics = evaluate_predictions(y_true, prob_dicts)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["log_loss"] < 0.01  # near zero, not exactly zero due to sklearn's internal epsilon clipping
    assert metrics["n_samples"] == 3


def test_evaluate_predictions_worse_model_has_worse_metrics():
    y_true = ["H", "H", "H"]
    good_preds = [{"H": 0.8, "D": 0.1, "A": 0.1}] * 3
    bad_preds = [{"H": 0.2, "D": 0.4, "A": 0.4}] * 3

    good_metrics = evaluate_predictions(y_true, good_preds)
    bad_metrics = evaluate_predictions(y_true, bad_preds)

    assert good_metrics["log_loss"] < bad_metrics["log_loss"]
    assert good_metrics["brier_score"] < bad_metrics["brier_score"]
    assert good_metrics["accuracy"] > bad_metrics["accuracy"]
