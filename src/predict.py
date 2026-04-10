import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
data_path = os.path.join(BASE_DIR, "content", "test.csv")
output_path = os.path.join(BASE_DIR, "submission.csv")

model = joblib.load(model_path)

df = pd.read_csv(data_path)

features = [
    "OverallQual", "GrLivArea", "GarageCars", 
    "TotalBsmtSF", "1stFlrSF", "YearBuilt", 
    "FullBath", "TotRmsAbvGrd", "GarageArea", 
    "LotArea", "YearRemodAdd"
]

X_test = df[features]

preds = model.predict(X_test)

submission = pd.DataFrame({
    "Id": df["Id"],
    "SalePrice": preds
})

submission.to_csv(output_path, index=False)

print("Submission file created:", output_path)

