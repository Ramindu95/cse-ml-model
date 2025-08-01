import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib # For loading scikit-learn models
from datetime import date
from typing import List, Dict, Union, Optional

# Adjust import paths based on your directory structure
from data.data_loader import load_stock_data, load_financial_data, get_connection
from data.feature_engineering import FeatureEngineer

# --- Configuration ---
# Path to your trained model.
# Now pointing to a specific model within the 'models' directory.
MODEL_DIRECTORY = "models"
DEFAULT_MODEL_FILENAME = "random_forest_model.pkl" # Choose your default model here
MODEL_PATH = f"{MODEL_DIRECTORY}/{DEFAULT_MODEL_FILENAME}"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock movements (Buy, Hold, Sell) based on historical and financial data.",
    version="1.0.0"
)

# --- Load Model and Feature Engineer Globally ---
# This ensures the model and feature engineer are loaded once when the app starts
model = None
feature_engineer = None

@app.on_event("startup")
async def load_resources():
    """Load the trained model and initialize the FeatureEngineer on startup."""
    global model, feature_engineer
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        # Provide a more helpful error message including the expected path
        raise RuntimeError(f"❌ Model file not found at {MODEL_PATH}. "
                           "Please ensure it exists and train a model if necessary.")
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")

    feature_engineer = FeatureEngineer()
    print("✅ FeatureEngineer initialized.")


# --- Pydantic Models for Request and Response ---

class StockDataPoint(BaseModel):
    company_id: int
    trade_date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    turnover: Optional[float] = None
    market_cap: Optional[float] = None
    previous_close: Optional[float] = None
    percentage_change: Optional[float] = None
    change_amount: Optional[float] = None

class FinancialDataPoint(BaseModel):
    company_id: int
    statement_type: str
    period_type: str
    period_start_date: date
    period_end_date: date
    is_audited: Optional[int] = None
    revenue: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    earnings_per_share: Optional[float] = None
    interest_expense: Optional[float] = None
    tax_expense: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    current_assets: Optional[float] = None
    total_assets: Optional[float] = None
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    shares_outstanding: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    net_cash_change: Optional[float] = None
    dividends_paid: Optional[float] = None

class PredictionRequest(BaseModel):
    company_id: int
    recent_stock_data: List[StockDataPoint]
    recent_financial_data: Optional[List[FinancialDataPoint]] = None
    # Optional: Allow specifying which model to use, though default is set
    # model_name: Optional[str] = DEFAULT_MODEL_FILENAME # This would require loading dynamically per request or having a registry

class PredictionResponse(BaseModel):
    company_id: int
    trade_date: date
    predicted_action: str
    prediction_label: int

# --- Helper Functions ---

# Keep get_latest_data_for_company and get_latest_financial_data_for_company
# if you intend to fetch data from the DB within the API.
# If the API consumer always sends all data, these might not be strictly necessary for '/predict'
# but useful for other potential endpoints or internal logic.

def get_latest_data_for_company(company_id: int, num_days: int = 60) -> pd.DataFrame:
    """
    Fetches the most recent 'num_days' of stock data for a given company.
    """
    conn = None
    try:
        conn = get_connection()
        query = f"""
            SELECT
                trade_date, open_price, high_price, low_price, close_price, volume,
                turnover, market_cap, previous_close, percentage_change, change_amount
            FROM stock_prices
            WHERE company_id = %s
            ORDER BY trade_date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(company_id, num_days))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['company_id'] = company_id # Add company_id back
        return df.sort_values('trade_date')
    except Exception as e:
        print(f"Error fetching latest stock data for company {company_id}: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_latest_financial_data_for_company(company_id: int, num_statements: int = 4) -> pd.DataFrame:
    """
    Fetches the most recent financial statements for a given company.
    """
    conn = None
    try:
        conn = get_connection()
        query = f"""
            SELECT *
            FROM financial_statements
            WHERE company_id = %s
            ORDER BY period_end_date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(company_id, num_statements))
        for col in ['period_start_date', 'period_end_date']:
            df[col] = pd.to_datetime(df[col])
        return df.sort_values('period_end_date', ascending=False)
    except Exception as e:
        print(f"Error fetching latest financial data for company {company_id}: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock Prediction API!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_action(request: PredictionRequest):
    """
    Predicts the next day's stock action (Buy, Hold, Sell) for a given company.
    Requires recent historical stock data and optional financial data.
    """
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model or Feature Engineer not loaded. Server is starting up or encountered an error.")

    # Convert incoming Pydantic data to Pandas DataFrame
    stock_df = pd.DataFrame([s.model_dump() for s in request.recent_stock_data])
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])

    financial_df = None
    if request.recent_financial_data:
        financial_df = pd.DataFrame([f.model_dump() for f in request.recent_financial_data])
        financial_df['period_start_date'] = pd.to_datetime(financial_df['period_start_date'])
        financial_df['period_end_date'] = pd.to_datetime(financial_df['period_end_date'])

    # Ensure the company_id is consistent
    if stock_df['company_id'].nunique() > 1 or stock_df['company_id'].iloc[0] != request.company_id:
        raise HTTPException(status_code=400, detail="Provided stock data must be for a single company matching the request's company_id.")

    if financial_df is not None and (financial_df['company_id'].nunique() > 1 or financial_df['company_id'].iloc[0] != request.company_id):
        raise HTTPException(status_code=400, detail="Provided financial data must be for a single company matching the request's company_id.")

    # Sort data by date
    stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
    if financial_df is not None:
        financial_df = financial_df.sort_values('period_end_date', ascending=False).reset_index(drop=True)

    # --- Feature Engineering ---
    try:
        df_for_prediction = stock_df.copy()
        if financial_df is not None and not financial_df.empty:
            df_for_prediction = feature_engineer.merge_financial_with_stock_data(df_for_prediction, financial_df)
        
        df_for_prediction = df_for_prediction.sort_values(['company_id', 'trade_date'])
        
        df_for_prediction = feature_engineer.add_moving_averages(df_for_prediction)
        df_for_prediction = feature_engineer.add_exponential_moving_averages(df_for_prediction)
        df_for_prediction = feature_engineer.add_rsi(df_for_prediction)
        df_for_prediction = feature_engineer.add_macd(df_for_prediction)
        df_for_prediction = feature_engineer.add_bollinger_bands(df_for_prediction)
        df_for_prediction = feature_engineer.add_stochastic_oscillator(df_for_prediction)
        df_for_prediction = feature_engineer.add_volume_indicators(df_for_prediction)
        df_for_prediction = feature_engineer.add_volatility_indicators(df_for_prediction)
        df_for_prediction = feature_engineer.add_momentum_indicators(df_for_prediction)
        df_for_prediction = feature_engineer.add_price_action_features(df_for_prediction)

        if 'financial_revenue' in df_for_prediction.columns:
            df_for_prediction = feature_engineer.add_financial_ratios(df_for_prediction)
            df_for_prediction = feature_engineer.add_growth_metrics(df_for_prediction)
        
        df_for_prediction = feature_engineer.clean_and_validate_features(df_for_prediction)

        # Ensure that `feature_engineer.all_features` is populated after these calls
        feature_engineer.all_features = feature_engineer.technical_features + feature_engineer.fundamental_features
        
        if not feature_engineer.all_features:
            raise HTTPException(status_code=500, detail="No features generated. Check feature engineering logic.")

        # Select the relevant features from the last row
        features_df = df_for_prediction[feature_engineer.all_features].tail(1)

        if features_df.empty:
            raise HTTPException(status_code=400, detail="Insufficient data to generate features for prediction. Please provide more historical data.")

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise HTTPException(status_code=500, detail=f"Error during feature engineering: {e}")

    # --- Make Prediction ---
    try:
        prediction_label = model.predict(features_df)[0]
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    # Map numerical prediction to human-readable action
    if prediction_label == 1:
        predicted_action = "Buy"
    elif prediction_label == -1:
        predicted_action = "Sell"
    else:
        predicted_action = "Hold"

    return PredictionResponse(
        company_id=request.company_id,
        trade_date=stock_df['trade_date'].max().date(),
        predicted_action=predicted_action,
        prediction_label=int(prediction_label)
    )

# --- Endpoint to check model status ---
@app.get("/status")
async def get_status():
    return {
        "model_loaded": model is not None,
        "feature_engineer_initialized": feature_engineer is not None,
        "model_path": MODEL_PATH
    }