from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler & columns
with open("diabetes_model.pkl", "rb") as f:
    model, scaler, model_columns = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input
        data = {
            "gender": request.form["gender"].lower(),
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"].lower(),
            "bmi": float(request.form["bmi"]),
            "HbA1c_level": float(request.form["HbA1c_level"]),
            "blood_glucose_level": float(request.form["blood_glucose_level"])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode
        input_df = pd.get_dummies(input_df)

        # Align columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        probability = model.predict_proba(input_scaled)[0][1]

        # Lower threshold (medical use)
        prediction = 1 if probability >= 0.4 else 0

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
