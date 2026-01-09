from scipy.stats import ks_2samp

def detect_drift(col_train, col_new, alpha=0.05):
    _, p_value = ks_2samp(col_train, col_new)
    return p_value < alpha

def drift_score(train_df, new_df):
    drifted = 0
    total = len(train_df.columns)

    for col in train_df.columns:
        if detect_drift(train_df[col], new_df[col]):
            drifted += 1

    return round((1 - drifted / total) * 100, 2)
