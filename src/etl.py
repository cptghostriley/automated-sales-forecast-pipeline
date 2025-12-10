# ETL Pipeline for Sales Data
# src/etl.py
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/sales.parquet")

def latest_csv():
    files = list(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV in data/raw/")
    return max(files, key=lambda f: f.stat().st_mtime)

def run_etl():
    p = latest_csv()
    df = pd.read_csv(p)
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Expected columns: store, date, weekly_sales, holiday_flag, temperature, fuel_price, cpi, unemployment
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date','store','weekly_sales'])
    df = df.sort_values(['store','date']).reset_index(drop=True)
    # Keep only useful columns (preserve extras if present)
    keep = ['store','date','weekly_sales','holiday_flag','temperature','fuel_price','cpi','unemployment']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print("ETL done. Processed saved to", PROCESSED_PATH)

if __name__ == "__main__":
    run_etl()

