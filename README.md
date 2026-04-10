#  House Price Prediction App

A machine learning web application built using Flask that predicts house prices based on user input features.

#  House Price Prediction App

-> Live Demo: https://house-price-prediction-0txy.onrender.com
-> Note: App may take ~30 seconds to load on first visit (free hosting).

##  Features
- Predict house prices using a trained ML model
- Simple and interactive web interface
- Built with Flask and Scikit-learn pipeline

##  Model Details
- Algorithm: Random Forest Regressor
- Preprocessing:
  - Missing value handling (SimpleImputer)
  - Feature scaling (StandardScaler)
- Hyperparameter tuning using GridSearchCV

##  Input Features
- OverallQual
- GrLivArea
- GarageCars
- TotalBsmtSF
- 1stFlrSF
- YearBuilt
- FullBath
- TotRmsAbvGrd
- GarageArea
- LotArea
- YearRemodAdd

##  Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

##  Project Structure
project/
│
├── app.py
├── predict.py
├── requirements.txt
├── Procfile
│
├── models/
│ └── best_model.pkl
│
├── templates/
│ └── index.html


##  Installation

```bash
git clone https://github.com/THANUJ-IMAYAVARAMBAN31/house_price_prediction
cd house_price_prediction
pip install -r requirements.txt
python app.py

http://127.0.0.1:5000/