from pathlib import Path
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE=Path(__file__).resolve().parent
DATA=BASE/'dataset'/'loan-data.csv'; MODEL_DIR=BASE/'models'; MODEL_DIR.mkdir(exist_ok=True)
FEATURES=['loan_amnt','term_months','int_rate','installment','annual_inc','home_ownership','verification_status','purpose']
NUMERIC=['loan_amnt','term_months','int_rate','installment','annual_inc']; CATEGORICAL=['home_ownership','verification_status','purpose']

def prepare():
    df=pd.read_csv(DATA)
    required=['loan_status']+['loan_amnt','term','int_rate','installment','annual_inc','home_ownership','verification_status','purpose']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError('Missing dataset columns: '+', '.join(missing))
    df=df[df.loan_status.isin(['Fully Paid','Charged Off'])].copy()
    df['term_months']=df.term.astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['int_rate']=df.int_rate.astype(str).str.replace('%','',regex=False).astype(float)
    df['default']=(df.loan_status=='Charged Off').astype(int)
    return df[FEATURES],df.default,df

def preprocessor():
    n=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    c=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num',n,NUMERIC),('cat',c,CATEGORICAL)])

def main():
    if not DATA.exists(): raise FileNotFoundError(f'Put loan-data.csv in {DATA}')
    X,y,df=prepare(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={
      'Logistic Regression':LogisticRegression(max_iter=1000,class_weight='balanced'),
      'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1),
      'Gradient Boosting':GradientBoostingClassifier(n_estimators=150,learning_rate=.05,max_depth=3,random_state=42)}
    metrics={}; trained={}
    for name,est in models.items():
        pipe=Pipeline([('preprocessor',preprocessor()),('model',est)]); pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        metrics[name]={'accuracy':round(accuracy_score(yte,pred),4),'precision':round(precision_score(yte,pred,zero_division=0),4),'recall':round(recall_score(yte,pred,zero_division=0),4),'f1':round(f1_score(yte,pred,zero_division=0),4),'roc_auc':round(roc_auc_score(yte,prob),4),'confusion_matrix':confusion_matrix(yte,pred).tolist()}
        trained[name]=pipe
        print(name,metrics[name])
    selected=max(metrics,key=lambda x:metrics[x]['roc_auc'])
    joblib.dump(trained[selected],MODEL_DIR/'credit_model.joblib')
    joblib.dump({'selected_model':selected,'models':metrics,'rows_used':len(df),'default_rate':float(y.mean())},MODEL_DIR/'metrics.joblib')
    print('\nSelected:',selected); print('Model saved to models/credit_model.joblib')
if __name__=='__main__': main()
