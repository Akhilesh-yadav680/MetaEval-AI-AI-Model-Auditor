import streamlit as st
import pandas as pd
import joblib

from evaluation.reliability import overall_reliability, reliability_score
from evaluation.bias import group_accuracy, bias_gap, fairness_score
from evaluation.calibration import calibration_score
from evaluation.drift import drift_score
from evaluation.trust_score import compute_trust_score, trust_verdict

st.set_page_config(page_title="MetaEval-AI", layout="wide")
st.title("🧠 MetaEval-AI — AI Model Auditor")

st.sidebar.header("Upload Inputs")

data_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])

if data_file and model_file:
    df = pd.read_csv(data_file).dropna()
    model = joblib.load(model_file)

    # Drop ID column
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Encode categorical features (same as training)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    if "Married" in df.columns:
        df["Married"] = df["Married"].map({"Yes": 1, "No": 0})

    if "Education" in df.columns:
        df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})

    if "Self_Employed" in df.columns:
        df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})

    if "Property_Area" in df.columns:
        df["Property_Area"] = df["Property_Area"].map(
            {"Urban": 2, "Semiurban": 1, "Rural": 0}
        )

    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Encode target
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    df["prediction"] = y_pred

    # Reliability
    metrics = overall_reliability(y, y_pred)
    reliability = reliability_score(metrics)

    # Fairness
    gender_acc = group_accuracy(df, "Gender")
    gap = bias_gap(gender_acc)
    fairness = fairness_score(gap)

    # Calibration
    calibration = calibration_score(y, y_prob)

    # Drift
    train = X.sample(frac=0.7, random_state=42)
    new = X.sample(frac=0.3, random_state=1)
    drift = drift_score(train, new)

    # Trust Score
    trust = compute_trust_score(reliability, fairness, calibration, drift)
    verdict = trust_verdict(trust)

    # Dashboard UI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reliability", f"{reliability}%")
    col2.metric("Fairness", f"{fairness}%")
    col3.metric("Calibration", f"{calibration}%")
    col4.metric("Drift Stability", f"{drift}%")

    st.divider()
    st.subheader("Final Trust Assessment")
    st.metric("Trust Score", f"{trust}%")
    st.success(verdict)

else:
    st.info("Upload dataset and model to start evaluation")
