from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load model, scaler, and feature names
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scale.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.json", "r") as f:
    feature_names = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form data
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        holiday = request.form['holiday']
        weather = request.form['weather']

        date = request.form['date']
        time = request.form['time']

        # Convert date and time to components
        from datetime import datetime
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")

        features = {
            "temp": temp,
            "rain": rain,
            "snow": snow,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
        }

        # One-hot encoding for weather
        for col in feature_names:
            if col.startswith("weather_"):
                features[col] = 1 if col == f"weather_{weather}" else 0

        # One-hot encoding for holiday
        for col in feature_names:
            if col.startswith("holiday_"):
                features[col] = 1 if col == f"holiday_{holiday}" else 0

        # Make sure final input vector is in correct order
        final_input = [features.get(col, 0) for col in feature_names]
        final_scaled = scaler.transform([final_input])
        prediction = model.predict(final_scaled)[0]

        return render_template("result.html", result=f"Estimated Traffic Volume: {int(prediction)} vehicles/hour")

    except Exception as e:
        return render_template("result.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
