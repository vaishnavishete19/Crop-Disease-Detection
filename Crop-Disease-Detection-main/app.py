from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("crop_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        soil_moisture = float(request.form["soil_moisture"])
        ph = float(request.form["ph"])

        data = np.array([[temperature, humidity, soil_moisture, ph]])
        result = model.predict(data)

        if result[0] == 1:
            prediction = "Crop is Diseased ❌"
        else:
            prediction = "Crop is Healthy ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
