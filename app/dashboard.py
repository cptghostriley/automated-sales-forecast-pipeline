# Dashboard application
# app/dashboard.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import joblib
import duckdb
from src.utils import build_features
from src.forecast import run_forecast

DATA_PATH = Path("data/processed/sales.parquet")
MODEL_DIR = Path("models")

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)
    return df

def main():
    st.title("Retail Sales Forecasting (Walmart)")

    df = load_data()
    stores = sorted(df['store'].unique())
    store = st.selectbox("Store", stores)

    st.header("Historical weekly sales")
    hist = df[df['store']==store].sort_values('date')
    st.line_chart(hist.set_index('date')['weekly_sales'])

    if st.button("Generate Forecast (best model)"):
        fc = run_forecast(weeks=8)
        st.success("Forecast generated")
        # show store forecast
        sfc = fc[fc['store']==store].sort_values('date')
        if not sfc.empty:
            st.subheader("Next weeks forecast")
            st.dataframe(sfc.set_index('date'))
            st.line_chart(sfc.set_index('date')['forecast'])
        else:
            st.write("No forecast available for this store.")

    st.sidebar.header("Models")
    if (MODEL_DIR/'best_model.txt').exists():
        best = (MODEL_DIR/'best_model.txt').read_text().strip()
        st.sidebar.write("Best model:", best)
    else:
        st.sidebar.write("No trained model found. Run src/train.py")

if __name__ == "__main__":
    main()

