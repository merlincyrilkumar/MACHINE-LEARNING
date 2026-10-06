import pandas as pd
def clean_dataset(df):
 df=df.drop_duplicates().copy(); df['term_months']=df.term.astype(str).str.extract(r'(\d+)')[0].astype(float); df['int_rate']=df.int_rate.astype(str).str.replace('%','',regex=False).astype(float); df=df[df.loan_status.isin(['Fully Paid','Charged Off'])].copy(); df['default']=(df.loan_status=='Charged Off').astype(int); return df
