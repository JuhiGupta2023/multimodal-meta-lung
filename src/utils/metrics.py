# src/utils/metrics.py
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) for binary/multiclass (works with max-prob).
    probs: numpy array [N, C] of softmax probs
    labels: numpy array [N] int
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0: continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return ece

def compute_brier(probs, labels):
    # multiclass brier: sum over classes
    N, C = probs.shape
    y_onehot = np.eye(C)[labels]
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

def compute_nll(probs, labels, eps=1e-12):
    return float(log_loss(labels, probs, eps=eps))
