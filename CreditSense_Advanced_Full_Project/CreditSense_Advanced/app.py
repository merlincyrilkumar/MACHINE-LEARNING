from flask import Flask,render_template,request,redirect,url_for,session,flash,send_file
from werkzeug.security import generate_password_hash,check_password_hash
from pathlib import Path
from datetime import datetime
import sqlite3,joblib,pandas as pd,uuid
BASE=Path(__file__).resolve().parent; DB=BASE/'database'/'creditsense.db'; MODEL=BASE/'models'/'credit_model.joblib'; METRICS=BASE/'models'/'metrics.joblib'
app=Flask(__name__); app.secret_key='change-this-secret-key'
FEATURES=['loan_amnt','term_months','int_rate','installment','annual_inc','home_ownership','verification_status','purpose']
PURPOSES=['credit_card','car','small_business','other','wedding','debt_consolidation','home_improvement','major_purchase','medical','moving','vacation','house','renewable_energy']
def db():
 DB.parent.mkdir(exist_ok=True); c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init_db():
 c=db(); c.executescript('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,name TEXT NOT NULL,email TEXT UNIQUE NOT NULL,password TEXT NOT NULL,created_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,user_id INTEGER,created_at TEXT,loan_amnt REAL,annual_inc REAL,term_months INTEGER,int_rate REAL,installment REAL,home_ownership TEXT,verification_status TEXT,purpose TEXT,prediction INTEGER,probability REAL);'''); c.commit(); c.close()
def auth(): return 'user_id' in session
def form_data(f): return {'loan_amnt':float(f['loan_amnt']),'annual_inc':float(f['annual_inc']),'term_months':int(f['term_months']),'int_rate':float(f['int_rate']),'installment':float(f['installment']),'home_ownership':f['home_ownership'],'verification_status':f['verification_status'],'purpose':f['purpose']}
def validate(d):
 if d['loan_amnt']<=0 or d['annual_inc']<=0 or d['installment']<=0: raise ValueError('Loan amount, annual income and installment must be positive.')
 if d['term_months'] not in (36,60): raise ValueError('Term must be 36 or 60 months.')
 if not 0<d['int_rate']<=40: raise ValueError('Interest rate must be between 0 and 40.')
@app.route('/')
def home(): return render_template('index.html')
@app.route('/register',methods=['GET','POST'])
def register():
 if request.method=='POST':
  name=request.form['name'].strip(); email=request.form['email'].strip().lower(); pw=request.form['password']
  if len(pw)<6: flash('Password must contain at least 6 characters.','error'); return redirect(url_for('register'))
  c=db()
  try: c.execute('INSERT INTO users(name,email,password,created_at) VALUES(?,?,?,?)',(name,email,generate_password_hash(pw),datetime.now().isoformat())); c.commit(); flash('Registration successful.','success'); return redirect(url_for('login'))
  except sqlite3.IntegrityError: flash('Email already registered.','error')
  finally: c.close()
 return render_template('register.html')
@app.route('/login',methods=['GET','POST'])
def login():
 if request.method=='POST':
  email=request.form['email'].strip().lower(); pw=request.form['password']; c=db(); u=c.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone(); c.close()
  if u and check_password_hash(u['password'],pw): session['user_id']=u['id']; session['user_name']=u['name']; return redirect(url_for('dashboard'))
  flash('Invalid email or password.','error')
 return render_template('login.html')
@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('home'))
@app.route('/dashboard')
def dashboard():
 if not auth(): return redirect(url_for('login'))
 c=db(); total=c.execute('SELECT COUNT(*) n FROM predictions WHERE user_id=?',(session['user_id'],)).fetchone()['n']; high=c.execute('SELECT COUNT(*) n FROM predictions WHERE user_id=? AND prediction=1',(session['user_id'],)).fetchone()['n']; c.close(); m=joblib.load(METRICS) if METRICS.exists() else {}; return render_template('dashboard.html',total=total,high=high,low=total-high,metrics=m)
@app.route('/predict',methods=['GET','POST'])
def predict():
 if not auth(): return redirect(url_for('login'))
 result=None; error=None
 if request.method=='POST':
  try:
   if not MODEL.exists(): raise RuntimeError('Run python train.py first.')
   d=form_data(request.form); validate(d); model=joblib.load(MODEL); X=pd.DataFrame([d],columns=FEATURES); pred=int(model.predict(X)[0]); prob=float(model.predict_proba(X)[0][1]); c=db(); c.execute('INSERT INTO predictions(user_id,created_at,loan_amnt,annual_inc,term_months,int_rate,installment,home_ownership,verification_status,purpose,prediction,probability) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(session['user_id'],datetime.now().isoformat(),*d.values(),pred,prob)); c.commit(); pid=c.execute('SELECT last_insert_rowid() id').fetchone()['id']; c.close(); result={'id':pid,'risk':'High Default Risk' if pred else 'Low Default Risk','prediction':pred,'probability':round(prob*100,2)}
  except Exception as e: error=str(e)
 return render_template('predict.html',result=result,error=error,purposes=PURPOSES)
@app.route('/history')
def history():
 if not auth(): return redirect(url_for('login'))
 c=db(); rows=c.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); c.close(); return render_template('history.html',rows=rows)
@app.route('/analytics')
def analytics():
 if not auth(): return redirect(url_for('login'))
 return render_template('analytics.html',metrics=joblib.load(METRICS) if METRICS.exists() else {})
@app.route('/batch',methods=['GET','POST'])
def batch():
 if not auth(): return redirect(url_for('login'))
 table=None; msg=None
 if request.method=='POST':
  try:
   f=request.files.get('file');
   if not f or not f.filename.endswith('.csv'): raise ValueError('Upload a CSV file.')
   df=pd.read_csv(f); missing=[x for x in FEATURES if x not in df.columns]
   if missing: raise ValueError('Missing columns: '+', '.join(missing))
   model=joblib.load(MODEL); df['prediction']=model.predict(df[FEATURES]); df['risk']=df.prediction.map({0:'Low Default Risk',1:'High Default Risk'}); df['default_probability_percent']=(model.predict_proba(df[FEATURES])[:,1]*100).round(2); out=BASE/'reports'/f'batch_{uuid.uuid4().hex[:8]}.csv'; df.to_csv(out,index=False); session['batch']=str(out); msg=f'Processed {len(df)} applicants.'; table=df.head(20).to_html(classes='table',index=False)
  except Exception as e: flash(str(e),'error')
 return render_template('batch.html',table=table,message=msg)
@app.route('/download-batch')
def download_batch():
 p=Path(session.get('batch','')); return send_file(p,as_attachment=True,download_name=p.name) if p.exists() else redirect(url_for('batch'))
@app.route('/report/<int:pid>')
def report(pid):
 if not auth(): return redirect(url_for('login'))
 c=db(); r=c.execute('SELECT * FROM predictions WHERE id=? AND user_id=?',(pid,session['user_id'])).fetchone(); c.close(); return render_template('report.html',row=r) if r else redirect(url_for('history'))
if __name__=='__main__': init_db(); app.run(debug=True)
