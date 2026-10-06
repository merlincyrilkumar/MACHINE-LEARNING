from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
def load_dataset(): return pd.read_csv(BASE/'dataset'/'loan-data.csv')
def summary():
 d=load_dataset(); return {'rows':len(d),'columns':len(d.columns),'missing':int(d.isna().sum().sum()),'duplicates':int(d.duplicated().sum())}
