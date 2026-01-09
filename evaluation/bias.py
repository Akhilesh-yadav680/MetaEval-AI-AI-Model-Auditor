from sklearn.metrics import accuracy_score

def group_accuracy(df, group_col):
    results = {}
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        acc = accuracy_score(
            subset["Loan_Status"],
            subset["prediction"]
        )
        results[group] = acc
    return results

def bias_gap(group_results):
    return round(max(group_results.values()) - min(group_results.values()), 3)

def fairness_score(gap):
    score = (1 - gap) * 100
    return round(max(score, 0), 2)
