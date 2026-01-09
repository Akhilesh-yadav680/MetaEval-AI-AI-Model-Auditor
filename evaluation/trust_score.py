def compute_trust_score(reliability, fairness, calibration, drift):
    score = (
        0.40 * reliability +
        0.25 * fairness +
        0.20 * calibration +
        0.15 * drift
    )
    return round(score, 2)

def trust_verdict(score):
    if score >= 85:
        return "🟢 Safe to Deploy"
    elif score >= 70:
        return "🟡 Monitor Closely"
    else:
        return "🔴 High Risk – Do Not Deploy"
