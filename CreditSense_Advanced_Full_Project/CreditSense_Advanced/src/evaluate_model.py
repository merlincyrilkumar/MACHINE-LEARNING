from pathlib import Path
import joblib
p=Path(__file__).resolve().parents[1]/'models'/'metrics.joblib'
def load_metrics(): return joblib.load(p) if p.exists() else {}
