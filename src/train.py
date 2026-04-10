import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("../content/train.csv")
features = ["OverallQual", "GrLivArea", "GarageCars", 
            "TotalBsmtSF", "1stFlrSF", "YearBuilt", "FullBath", 
            "TotRmsAbvGrd", "GarageArea", "LotArea", "YearRemodAdd"]

X = df[features]
y = df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, features)
    ]
)
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

params_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [5, 10],
    "model__min_samples_leaf": [2, 4],
    "model__max_features": [ "sqrt", "log2"],
    "model__bootstrap": [True, False],
}
grid_search = GridSearchCV(pipe, params_grid, cv=5, scoring="neg_mean_squared_error", verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_    

joblib.dump(best_model, "../models/best_model.pkl")