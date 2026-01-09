from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def overall_reliability(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def reliability_score(metrics):
    return round(sum(metrics.values()) / len(metrics) * 100, 2)
