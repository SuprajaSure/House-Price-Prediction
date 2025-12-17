from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model & scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Extract values in same order as FEATURES
            values = [float(request.form[feat]) for feat in FEATURES]
            X = scaler.transform([values])
            prediction = model.predict(X)[0]
        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
