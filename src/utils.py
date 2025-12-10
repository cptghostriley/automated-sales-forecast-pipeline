# Utility functions
# src/utils.py
import pandas as pd
import numpy as np

def build_features(df):
    df = df.sort_values(['store','date']).copy()
    # Weekly lag features
    df['lag_1'] = df.groupby('store')['weekly_sales'].shift(1)
    df['lag_2'] = df.groupby('store')['weekly_sales'].shift(2)
    df['lag_52'] = df.groupby('store')['weekly_sales'].shift(52)
    # Rolling
    df['roll_4'] = df.groupby('store')['weekly_sales'].rolling(4, min_periods=1).mean().reset_index(0,drop=True)
    df['roll_12'] = df.groupby('store')['weekly_sales'].rolling(12, min_periods=1).mean().reset_index(0,drop=True)
    # Calendar
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    # Fill small gaps with last known values for exogenous features
    for col in ['temperature','fuel_price','cpi','unemployment','holiday_flag']:
        if col in df.columns:
            df[col] = df.groupby('store')[col].ffill().bfill()
    return df

