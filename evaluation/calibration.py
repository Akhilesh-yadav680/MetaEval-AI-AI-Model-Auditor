from sklearn.metrics import brier_score_loss

def calibration_score(y_true, probabilities):
    brier = brier_score_loss(y_true, probabilities)
    score = (1 - brier) * 100
    return round(max(score, 0), 2)
