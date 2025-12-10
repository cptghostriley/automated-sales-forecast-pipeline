# Model training pipeline
# src/train.py
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from prophet import Prophet
import tensorflow as tf

from .utils import build_features

DATA_PATH = Path("data/processed/sales.parquet")
MODEL_DIR = Path("models")
XGB_PATH = MODEL_DIR / "xgb_model.pkl"
PROPHET_PATH = MODEL_DIR / "prophet_model.pkl"
LSTM_PATH = MODEL_DIR / "lstm_model.h5"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_xgboost(df):
    features = ['store','lag_1','lag_2','lag_52','roll_4','roll_12',
                'holiday_flag','temperature','fuel_price','cpi','unemployment',
                'week','month','year']
    X = df[features].copy()
    X['store'] = X['store'].astype('category').cat.codes
    y = df['weekly_sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    joblib.dump(model, XGB_PATH)
    print("XGBoost MAE:", mae)
    return mae

def train_prophet(df):
    # Prophet needs ds, y and added regressors
    df_p = df[['date','weekly_sales','temperature','fuel_price','cpi','unemployment','holiday_flag']].rename(columns={'date':'ds','weekly_sales':'y'})
    # Prophet expects no NaNs
    df_p = df_p.fillna(method='ffill').fillna(method='bfill')
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    for r in ['temperature','fuel_price','cpi','unemployment','holiday_flag']:
        if r in df_p.columns:
            m.add_regressor(r)
    m.fit(df_p)
    # Evaluate on last 10% by time
    split = int(len(df_p)*0.9)
    train = df_p.iloc[:split]
    val = df_p.iloc[split:]
    fc = m.predict(val[['ds'] + [c for c in ['temperature','fuel_price','cpi','unemployment','holiday_flag'] if c in val.columns]])
    mae = mean_absolute_error(val['y'], fc['yhat'])
    joblib.dump(m, PROPHET_PATH)
    print("Prophet MAE:", mae)
    return mae

def train_lstm(df, seq_len=8):
    # Global LSTM on sliding windows
    # use features: sales + exogenous numeric (temperature, fuel_price, cpi, unemployment)
    df2 = df.copy()
    df2['store_code'] = df2['store'].astype('category').cat.codes
    exog_cols = [c for c in ['temperature','fuel_price','cpi','unemployment'] if c in df2.columns]
    # Scale not included here — using simple scale (min-max) per series for speed. In prod use RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    scalers = {}
    df2['weekly_sales_scaled'] = df2.groupby('store')['weekly_sales'].transform(lambda x: (x - x.min())/(x.max()-x.min()+1e-9))
    for c in exog_cols:
        df2[c+'_s'] = df2.groupby('store')[c].transform(lambda x: (x - x.min())/(x.max()-x.min()+1e-9))
    # Build sequences
    Xs, ys = [], []
    grouped = df2.groupby('store_code')
    for store_code, g in grouped:
        g = g.sort_values('date')
        arr = g['weekly_sales_scaled'].values
        ex = np.stack([g[c+'_s'].values for c in exog_cols], axis=1) if exog_cols else None
        for i in range(len(g)-seq_len):
            seq_y = arr[i+seq_len]
            seq_x = arr[i:i+seq_len].reshape(-1,1)
            if ex is not None:
                ex_seq = ex[i:i+seq_len]
                seq_x = np.concatenate([seq_x, ex_seq], axis=1)
            Xs.append(seq_x)
            ys.append(seq_y)
    if len(Xs) < 20:
        print("Not enough sequences for LSTM. Skipping LSTM training.")
        return np.inf
    Xs = np.array(Xs)
    ys = np.array(ys)
    # train/test random split
    idx = np.arange(len(Xs))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.85*len(Xs))
    X_train, X_val = Xs[idx[:split]], Xs[idx[split:]]
    y_train, y_val = ys[idx[:split]], ys[idx[split:]]
    # build model
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_train.shape[1:]),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=20, batch_size=64, verbose=0)
    # eval
    preds = model.predict(X_val).flatten()
    mae = np.mean(np.abs(preds - y_val))
    model.save(LSTM_PATH)
    print("LSTM MAE (scaled):", mae)
    return mae

def train_all():
    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)
    # Drop rows without target or lags
    df = df.dropna(subset=['lag_1','lag_2'])
    # Train XGB
    xgb_mae = train_xgboost(df)
    # Train Prophet
    prophet_mae = train_prophet(df)
    # Train LSTM
    lstm_mae = train_lstm(df)
    # Pick best model = lowest MAE (note: Prophet MAE computed on time-split; XGB on randomized val)
    maes = {'xgb': xgb_mae, 'prophet': prophet_mae, 'lstm': lstm_mae}
    print("MAEs:", maes)
    best = min(maes, key=maes.get)
    print("Best model:", best)
    # (models were saved during each train function). Could store best name file for forecast step
    with open(MODEL_DIR/'best_model.txt','w') as f:
        f.write(best)

if __name__ == "__main__":
    train_all()
