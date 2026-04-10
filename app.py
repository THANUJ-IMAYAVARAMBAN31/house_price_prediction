from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd  

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_names = [
            "OverallQual", "GrLivArea", "GarageCars",
            "TotalBsmtSF", "1stFlrSF", "YearBuilt",
            "FullBath", "TotRmsAbvGrd", "GarageArea",
            "LotArea", "YearRemodAdd"
        ]

        values = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["GarageCars"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["1stFlrSF"]),
            float(request.form["YearBuilt"]),
            float(request.form["FullBath"]),
            float(request.form["TotRmsAbvGrd"]),
            float(request.form["GarageArea"]),
            float(request.form["LotArea"]),
            float(request.form["YearRemodAdd"]),
        ]

        input_df = pd.DataFrame([values], columns=feature_names)

        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ₹ {round(prediction,2)}"
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)