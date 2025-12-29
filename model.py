import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\Abhirami R\Desktop\diabetes_prediction_dataset.csv")

# Lowercase categorical values (VERY IMPORTANT)
df["gender"] = df["gender"].str.lower()
df["smoking_history"] = df["smoking_history"].str.lower()

# Features & target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model (handles imbalance)
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Save model, scaler & columns
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns), f)

print("✅ Model trained & saved successfully")
