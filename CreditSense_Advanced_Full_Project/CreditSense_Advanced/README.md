# CreditSense Advanced

Put the supplied `loan-data.csv` inside `dataset/`.

## Setup
py -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt

## Train
python train.py

The training script compares Logistic Regression, Random Forest and Gradient Boosting, evaluates accuracy, precision, recall, F1 and ROC-AUC, and saves the selected pipeline to `models/credit_model.joblib`.

## Run
python app.py
Open http://127.0.0.1:5000

## Features
- Flask web application
- Registration/login with hashed passwords
- SQLite prediction history
- Dashboard
- Risk probability
- Model comparison and analytics
- Batch CSV prediction
- Printable prediction report
- Input validation

## Dataset note
The current supplied dataset does not contain credit score or debt-to-income ratio. The model therefore does not use those fields. `Current` loans are excluded because the final outcome is unknown, and `total_pymnt` is excluded to avoid post-outcome leakage.
