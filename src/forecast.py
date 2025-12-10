# Forecasting module
# src/forecast.py
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta
import numpy as np
from .utils import build_features
import tensorflow as tf

DATA_PATH = Path("data/processed/sales.parquet")
MODEL_DIR = Path("models")
XGB_PATH = MODEL_DIR / "xgb_model.pkl"
PROPHET_PATH = MODEL_DIR / "prophet_model.pkl"
LSTM_PATH = MODEL_DIR / "lstm_model.h5"
BEST_PATH = MODEL_DIR / "best_model.txt"

def predict_with_xgb(model, df, weeks=4):
    features = ['store','lag_1','lag_2','lag_52','roll_4','roll_12',
                'holiday_flag','temperature','fuel_price','cpi','unemployment','week','month','year']
    outs=[]
    for store, g in df.groupby('store'):
        g = g.sort_values('date').copy()
        last = g.iloc[-1]
        cur = g.copy()
        for i in range(1, weeks+1):
            next_date = last['date'] + timedelta(weeks=i)
            inp = {
                'store': last['store'],
                'lag_1': cur['weekly_sales'].iloc[-1],
                'lag_2': cur['weekly_sales'].iloc[-2] if len(cur)>=2 else cur['weekly_sales'].iloc[-1],
                'lag_52': cur['weekly_sales'].iloc[-52] if len(cur)>=52 else cur['weekly_sales'].iloc[-1],
                'roll_4': cur['weekly_sales'].tail(4).mean(),
                'roll_12': cur['weekly_sales'].tail(12).mean(),
                'holiday_flag': 0,
                'temperature': last.get('temperature', np.nan),
                'fuel_price': last.get('fuel_price', np.nan),
                'cpi': last.get('cpi', np.nan),
                'unemployment': last.get('unemployment', np.nan),
                'week': next_date.isocalendar().week,
                'month': next_date.month,
                'year': next_date.year
            }
            Xpred = pd.DataFrame([inp])
            Xpred['store'] = Xpred['store'].astype('category').cat.codes
            yhat = model.predict(Xpred[features])[0]
            outs.append([store, next_date, yhat])
            # append predicted row so next step uses it
            cur = pd.concat([cur, pd.DataFrame({k:[v] for k,v in inp.items()})], ignore_index=True)
            cur['weekly_sales'].iloc[-1] = yhat
    return pd.DataFrame(outs, columns=['store','date','forecast'])

def predict_with_prophet(model, df, weeks=4):
    outs=[]
    # Prophet works on whole df using regressors — easiest is to produce future dataframe per store using last known regressors
    for store, g in df.groupby('store'):
        g = g.sort_values('date')
        last = g.iloc[-1]
        future_dates = pd.date_range(start=last['date']+pd.Timedelta(weeks=1), periods=weeks, freq='W')
        future = pd.DataFrame({'ds': future_dates})
        # copy regressors as constants (or replace with forecasted exogenous)
        for r in ['temperature','fuel_price','cpi','unemployment','holiday_flag']:
            if r in g.columns:
                future[r] = last[r]
        fc = model.predict(future)
        for d,yhat in zip(fc['ds'], fc['yhat']):
            outs.append([store, d, yhat])
    return pd.DataFrame(outs, columns=['store','date','forecast'])

def predict_with_lstm(model, df, weeks=4, seq_len=8):
    # simplified: use last seq per store (with same scaling used in train); here we assume training used simple min-max per store
    outs=[]
    exog_cols = [c for c in ['temperature','fuel_price','cpi','unemployment'] if c in df.columns]
    for store, g in df.groupby('store'):
        g = g.sort_values('date').copy()
        # scale same way as in training: min-max
        sales = g['weekly_sales'].values
        smin, smax = sales.min(), sales.max()
        sales_s = (sales - smin)/(smax - smin + 1e-9)
        if len(sales_s) < seq_len:
            continue
        
        # Scale exogenous features
        exog_scaled = []
        exog_mins, exog_maxs = [], []
        for c in exog_cols:
            vals = g[c].values
            vmin, vmax = vals.min(), vals.max()
            exog_mins.append(vmin)
            exog_maxs.append(vmax)
            vals_s = (vals - vmin)/(vmax - vmin + 1e-9)
            exog_scaled.append(vals_s[-seq_len:])
        
        # Build sequence with sales + exogenous features
        seq_sales = sales_s[-seq_len:].reshape(-1, 1)
        if exog_cols:
            exog_array = np.stack(exog_scaled, axis=1)
            seq = np.concatenate([seq_sales, exog_array], axis=1)
        else:
            seq = seq_sales
        seq = seq.reshape(1, seq_len, -1)
        
        for i in range(weeks):
            pred_s = model.predict(seq, verbose=0)[0,0]
            pred = pred_s*(smax - smin) + smin
            next_date = g['date'].iloc[-1] + pd.Timedelta(weeks=i+1)
            outs.append([store, next_date, float(pred)])
            # advance seq by appending predicted scaled value and last known exogenous values
            new_timestep = np.array([pred_s] + [exog_scaled[j][-1] for j in range(len(exog_cols))]).reshape(1, 1, -1)
            seq = np.concatenate([seq[:,1:,:], new_timestep], axis=1)
    return pd.DataFrame(outs, columns=['store','date','forecast'])

def run_forecast(weeks=4):
    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)
    with open(BEST_PATH,'r') as f:
        best = f.read().strip()
    if best == 'xgb':
        model = joblib.load(XGB_PATH)
        return predict_with_xgb(model, df, weeks=weeks)
    elif best == 'prophet':
        model = joblib.load(PROPHET_PATH)
        return predict_with_prophet(model, df, weeks=weeks)
    elif best == 'lstm':
        model = tf.keras.models.load_model(LSTM_PATH)
        return predict_with_lstm(model, df, weeks=weeks)
    else:
        raise ValueError("Unknown best model:", best)

if __name__ == "__main__":
    print(run_forecast(weeks=4).head())

