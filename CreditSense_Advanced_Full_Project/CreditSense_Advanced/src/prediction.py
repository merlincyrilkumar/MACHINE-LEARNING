from pathlib import Path
import joblib,pandas as pd
BASE=Path(__file__).resolve().parents[1]
def predict(data):
 model=joblib.load(BASE/'models'/'credit_model.joblib'); x=pd.DataFrame([data]); return int(model.predict(x)[0]),float(model.predict_proba(x)[0][1])
