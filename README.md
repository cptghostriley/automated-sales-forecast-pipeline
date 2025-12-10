# 🛒 Automated Sales Forecast Pipeline

An end-to-end machine learning pipeline for predicting Walmart weekly sales using time-series forecasting. This system automates data ingestion, feature engineering, model training, and provides an interactive dashboard for visualizing forecasts.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Features

- **Automated ETL Pipeline** - Processes raw CSV sales data with economic indicators
- **Advanced Feature Engineering** - Creates lag features, rolling averages, and calendar variables
- **Multi-Model Training** - Trains and compares 3 forecasting models:
  - **XGBoost** - Gradient boosting regression
  - **Prophet** - Facebook's time-series forecasting
  - **LSTM** - Deep learning neural network
- **Model Selection** - Automatically selects the best performing model
- **Interactive Dashboard** - Streamlit app for visualizing historical sales and forecasts
- **One-Command Execution** - Run the entire pipeline with a single script

## 🏗️ Project Structure

```
automated-sales-forecast-pipeline/
├── data/
│   ├── raw/                    # Drop new CSV files here
│   │   └── Walmart_Sales.csv
│   └── processed/
│       └── sales.parquet       # Processed data
├── models/
│   ├── xgb_model.pkl          # XGBoost model
│   ├── prophet_model.pkl      # Prophet model
│   ├── lstm_model.h5          # LSTM model
│   └── best_model.txt         # Best model identifier
├── src/
│   ├── etl.py                 # Data extraction & transformation
│   ├── utils.py               # Feature engineering utilities
│   ├── train.py               # Model training pipeline
│   └── forecast.py            # Forecasting module
├── app/
│   └── dashboard.py           # Streamlit dashboard
├── pipeline.py                # Main pipeline orchestrator
├── requirements.txt           # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/cptghostriley/automated-sales-forecast-pipeline.git
cd automated-sales-forecast-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Place your data**
   - Add Walmart sales CSV files to `data/raw/`
   - CSV should contain: `store`, `date`, `weekly_sales`, `holiday_flag`, `temperature`, `fuel_price`, `cpi`, `unemployment`

### Usage

#### Run the Full Pipeline

Execute the entire ETL → Training → Forecasting pipeline:

```bash
python pipeline.py
```

This will:
1. Extract and clean data from `data/raw/`
2. Engineer time-series features
3. Train all three models (XGBoost, Prophet, LSTM)
4. Select the best model based on MAE
5. Save trained models to `models/`

#### Run Individual Components

**ETL Only:**
```bash
python src/etl.py
```

**Training Only:**
```bash
python src/train.py
```

**Forecasting Only:**
```bash
python src/forecast.py
```

#### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Open your browser to `http://localhost:8501` to:
- View historical sales by store
- Generate 8-week forecasts
- Visualize predictions
- Compare model performance

## 📊 Data Requirements

Your CSV file should include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `store` | int | Store identifier |
| `date` | date | Week ending date |
| `weekly_sales` | float | Sales for the week |
| `holiday_flag` | int | 1 if holiday week, 0 otherwise |
| `temperature` | float | Average temperature |
| `fuel_price` | float | Fuel cost in the region |
| `cpi` | float | Consumer Price Index |
| `unemployment` | float | Unemployment rate |

## 🧠 Models

### XGBoost
- Uses gradient boosting with 500 estimators
- Features: lags, rolling averages, economic indicators
- Best for: Short-term predictions with clear patterns

### Prophet
- Facebook's additive time-series model
- Handles weekly/yearly seasonality automatically
- Best for: Long-term trends with seasonality

### LSTM
- Recurrent neural network with 64 LSTM units
- Learns sequential dependencies in sales data
- Best for: Complex non-linear patterns

## 📈 Model Performance

The pipeline automatically selects the best model using Mean Absolute Error (MAE) on validation data. Typical performance:

- **XGBoost MAE**: ~15,000 - 25,000
- **Prophet MAE**: ~20,000 - 30,000
- **LSTM MAE**: ~18,000 - 28,000

*Results vary based on data quality and store patterns*

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **XGBoost** - Gradient boosting
- **Prophet** - Time-series forecasting
- **TensorFlow/Keras** - Deep learning
- **Scikit-learn** - ML utilities
- **Streamlit** - Dashboard UI
- **DuckDB** - SQL analytics
- **Matplotlib** - Visualizations

## 📦 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Set main file: `app/dashboard.py`
5. Deploy!

### Docker (Optional)

```bash
# Build image
docker build -t sales-forecast .

# Run container
docker run -p 8501:8501 sales-forecast
```

## 🔄 Future Enhancements

- [ ] Automated file watcher for new data
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model drift detection and alerts
- [ ] Multi-store batch predictions
- [ ] REST API for forecasts (FastAPI)
- [ ] Advanced hyperparameter tuning
- [ ] Database integration (PostgreSQL)
- [ ] What-if scenario analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**cptghostriley**
- GitHub: [@cptghostriley](https://github.com/cptghostriley)

## 🙏 Acknowledgments

- Walmart sales dataset for training and testing
- Facebook Prophet team for time-series forecasting
- Streamlit team for the amazing dashboard framework

---

⭐ Star this repo if you find it helpful!

