import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/loan_data.csv")
df = df.dropna()

# Drop ID column (not useful for prediction)
df = df.drop("Loan_ID", axis=1)

# Encode categorical columns
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})

# Convert Dependents column
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Encode target column
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/base_model.pkl")

print("✅ Base model trained and saved successfully")
