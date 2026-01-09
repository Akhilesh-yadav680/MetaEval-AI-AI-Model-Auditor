import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/house_data.csv")

# Drop ID column
df = df.drop("Id", axis=1)

# Encode categorical columns
encoder = LabelEncoder()

df["Location"] = encoder.fit_transform(df["Location"])
df["Condition"] = encoder.fit_transform(df["Condition"])
df["Garage"] = encoder.fit_transform(df["Garage"])

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/house_price_model.pkl")

print("✅ House Price Prediction model trained and saved successfully")
