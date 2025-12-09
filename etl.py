import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path

def load_data():
    #Loads the most recent CSV file from the current directory
    raw_path = Path(".")
    processed_path = Path("./processed.parquet")
    latest_path = max(raw_path.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    dff = pd.read_csv(latest_path)

    #feature engineering - date features
    dff['Date']= pd.to_datetime(dff['Date'], format='%d-%m-%Y')
    dff = dff.sort_values('Date').reset_index(drop=True)
    dff['week']=dff['Date'].dt.isocalendar().week
    dff['month']=dff['Date'].dt.month
    dff['year']=dff['Date'].dt.year

    #feature engineering - lag features
    dff['lag_1']=dff.groupby('Store')['Weekly_Sales'].shift(1)
    dff['lag_2']=dff.groupby('Store')['Weekly_Sales'].shift(2)
    dff['lag_52']=dff.groupby('Store')['Weekly_Sales'].shift(52)

    #drop rows with NaN values
    dff = dff.dropna().reset_index(drop=True)

    return dff


