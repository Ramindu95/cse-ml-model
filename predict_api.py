import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib # For loading scikit-learn models
from datetime import date
from typing import List, Dict, Union, Optional, Any
import os
import logging
import json
from pathlib import Path
import sys


# Load .env file
load_dotenv()

# Get the minimum historical days, default to 15 if not set
MIN_HISTORICAL_DAYS = int(os.getenv("MIN_HISTORICAL_DAYS", 15))

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__) # Use logger for this module

# --- Adjust import paths for fs_parser.py ---
# This assumes predict_api.py is at the project root (e.g., cse-analysis-backend/predict_api.py)
# and fs_parser.py is in 'data/fs_data' relative to that root.
script_dir = Path(__file__).parent # This will be 'cse-analysis-backend/' if run from there

fs_parser_path = script_dir / "data" / "fs_data" # Corrected path: data/fs_data relative to script_dir
if str(fs_parser_path) not in sys.path:
    sys.path.insert(0, str(fs_parser_path))

try:
    # Import FinancialDataExtractor and related data models for feedback
    from fs_data_extraction.fs_parser import FinancialDataExtractor, ExtractedFinancialData, ExtractionFeedback
    logger.info("✅ Successfully imported fs_parser for feedback functionality.")
except ImportError as e:
    logger.error(f"❌ Failed to import fs_parser: {e}. Feedback API will not be available.")
    # Do not sys.exit(1) here, as the main prediction API should still function.
    FinancialDataExtractor = None # Set to None to prevent errors later

# --- Data Loader and Feature Engineering Imports (from your original predict_api.py) ---
# Assuming these are in a 'data' directory relative to the project root.
data_loader_path = script_dir / "data" # Corrected path: data relative to script_dir
if str(data_loader_path) not in sys.path:
    sys.path.insert(0, str(data_loader_path))

try:
    from data_loader import load_stock_data, load_financial_data, get_connection
    from feature_engineering import FeatureEngineer
    logger.info("✅ Successfully imported data_loader and feature_engineering.")
except ImportError as e:
    logger.error(f"❌ Failed to import data_loader or feature_engineering: {e}. Prediction API will not be fully functional.")
    load_stock_data = None
    load_financial_data = None
    get_connection = None
    FeatureEngineer = None


# --- Configuration ---
# Path to your trained model and training features
MODEL_DIRECTORY = "models" # Relative to where predict_api.py is run from
DEFAULT_MODEL_FILENAME = "random_forest_model.pkl"
TRAINING_FEATURES_FILENAME = "training_features.json"
MODEL_PATH = Path(MODEL_DIRECTORY) / DEFAULT_MODEL_FILENAME
TRAINING_FEATURES_PATH = Path(MODEL_DIRECTORY) / TRAINING_FEATURES_FILENAME

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stock Prediction & Financial Extraction Feedback API",
    description="API for predicting stock movements and receiving feedback on financial data extraction.",
    version="1.0.0"
)

# --- Global Resources ---
model = None
feature_engineer = None
training_features = None
financial_data_extractor: Optional[FinancialDataExtractor] = None # Global instance for feedback

@app.on_event("startup")
async def load_resources():
    """Load all resources: ML model, Feature Engineer, and FinancialDataExtractor."""
    global model, feature_engineer, training_features, financial_data_extractor
    
    # --- Load Stock Prediction Resources ---
    try:
        if not MODEL_PATH.exists():
            logger.error(f"❌ Model file not found at {MODEL_PATH}. Prediction API will be limited.")
        else:
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Stock Prediction Model loaded successfully from {MODEL_PATH}")
        
        if FeatureEngineer: # Check if FeatureEngineer was imported successfully
            feature_engineer = FeatureEngineer()
            logger.info("✅ FeatureEngineer initialized")
        else:
            logger.warning("⚠️ FeatureEngineer not initialized due to import error.")
        
        if TRAINING_FEATURES_PATH.exists():
            if feature_engineer:
                training_features = feature_engineer.load_training_features(TRAINING_FEATURES_PATH)
                if training_features:
                    logger.info(f"✅ Loaded {len(training_features)} training features")
                else:
                    logger.warning("⚠️ Training features file exists but no features loaded")
            else:
                logger.warning("⚠️ Cannot load training features: FeatureEngineer not initialized.")
        else:
            logger.warning(f"⚠️ Training features file not found at {TRAINING_FEATURES_PATH}")
            logger.warning("   This may cause feature alignment issues during prediction")
            
    except Exception as e:
        logger.error(f"❌ Error during stock prediction resource startup: {e}")
        # Allow the app to start, but prediction functionality might be impaired.

    # --- Initialize FinancialDataExtractor for Feedback ---
    if FinancialDataExtractor: # Check if FinancialDataExtractor was imported successfully
        try:
            # Adjust ml_model_dir and db_path for FinancialDataExtractor
            # Assuming 'ml_models' is at the project root and 'extraction_learning.db' is in 'data/fs_data'
            # relative to the project root.
            fp_ml_model_dir = script_dir / "ml_models" # Project root / ml_models
            fp_db_path = script_dir / "data" / "fs_data" / "extraction_learning.db" # Project root / data / fs_data / db
            
            financial_data_extractor = FinancialDataExtractor(ml_model_dir=str(fp_ml_model_dir), db_path=str(fp_db_path))
            logger.info("✅ FinancialDataExtractor initialized successfully for feedback API.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FinancialDataExtractor for feedback: {e}")
            financial_data_extractor = None # Ensure it's None if initialization fails
    else:
        logger.warning("⚠️ FinancialDataExtractor not initialized: Class not found due to import error.")


# --- Pydantic Models for Request and Response (Stock Prediction) ---

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

class PredictionResponse(BaseModel):
    company_id: int
    trade_date: date
    predicted_action: str
    prediction_label: int
    confidence: Optional[float] = None
    features_used: Optional[int] = None
    technical_features: Optional[int] = None
    fundamental_features: Optional[int] = None

# --- Pydantic Model for Feedback (Financial Extraction) ---
class FeedbackItem(BaseModel):
    document_hash: str
    field_name: str
    extracted_value: Any
    correct_value: Any
    confidence_score: Optional[float] = None
    user_id: str = "frontend_user" # Default user ID for feedback

# --- Helper Functions (Stock Prediction) ---

def get_latest_data_for_company(company_id: int, num_days: int = 60) -> pd.DataFrame:
    """
    Fetches the most recent 'num_days' of stock data for a given company.
    """
    if get_connection is None:
        logger.error("Database connection function not available.")
        return pd.DataFrame()
    conn = None
    try:
        conn = get_connection()
        query = """
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
        df['company_id'] = company_id
        return df.sort_values('trade_date')
    except Exception as e:
        logger.error(f"Error fetching latest stock data for company {company_id}: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_latest_financial_data_for_company(company_id: int, num_statements: int = 4) -> pd.DataFrame:
    """
    Fetches the most recent financial statements for a given company.
    """
    if get_connection is None:
        logger.error("Database connection function not available.")
        return pd.DataFrame()
    conn = None
    try:
        conn = get_connection()
        query = """
            SELECT *
            FROM financial_statements
            WHERE company_id = %s
            ORDER BY period_end_date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(company_id, num_statements))
        for col in ['period_start_date', 'period_end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df.sort_values('period_end_date', ascending=False)
    except Exception as e:
        logger.error(f"Error fetching latest financial data for company {company_id}: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()

def map_prediction_to_action(prediction_label: int) -> str:
    """
    Map numerical prediction labels to human-readable actions.
    Standard mapping: 0=Sell, 1=Hold, 2=Buy
    """
    action_map = {
        0: "Sell",
        1: "Hold", 
        2: "Buy"
    }
    return action_map.get(prediction_label, "Hold")

def validate_data_consistency(request: PredictionRequest) -> None:
    """Validate that all provided data is for the same company"""
    # Validate company_id consistency in stock data
    stock_company_ids = set(s.company_id for s in request.recent_stock_data)
    if len(stock_company_ids) > 1 or request.company_id not in stock_company_ids:
        raise HTTPException(
            status_code=400, 
            detail="All stock data must be for the same company as specified in company_id."
        )

    # Validate company_id consistency in financial data if provided
    if request.recent_financial_data:
        financial_company_ids = set(f.company_id for f in request.recent_financial_data)
        if len(financial_company_ids) > 1 or request.company_id not in financial_company_ids:
            raise HTTPException(
                status_code=400, 
                detail="All financial data must be for the same company as specified in company_id."
            )

def prepare_dataframes_from_request(request: PredictionRequest) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Convert Pydantic request data to pandas DataFrames"""
    try:
        # Convert stock data to DataFrame
        stock_df = pd.DataFrame([s.model_dump() for s in request.recent_stock_data])
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)

        # Convert financial data to DataFrame if provided
        financial_df = None
        if request.recent_financial_data:
            financial_df = pd.DataFrame([f.model_dump() for f in request.recent_financial_data])
            for col in ['period_start_date', 'period_end_date']:
                if col in financial_df.columns:
                    financial_df[col] = pd.to_datetime(financial_df[col])
            financial_df = financial_df.sort_values('period_end_date', ascending=False).reset_index(drop=True)

        return stock_df, financial_df

    except Exception as e:
        logger.error(f"Error converting request data to DataFrames: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Stock Prediction & Financial Extraction Feedback API!",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict - Make prediction with provided data",
            "predict_from_db": "/predict/company/{company_id} - Make prediction using database data",
            "feedback": "/feedback - Record financial extraction feedback",
            "status": "/status - Check API status",
            "health": "/health - Health check",
            "features": "/features/info - Get feature information"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_action(request: PredictionRequest):
    """
    Predicts the next day's stock action (Buy, Hold, Sell) for a given company.
    Uses the FeatureEngineer class for comprehensive feature generation.
    """
    if model is None or feature_engineer is None:
        raise HTTPException(
            status_code=503, 
            detail="Stock Prediction Model or Feature Engineer not loaded. Server is starting up or encountered an error."
        )
    if load_stock_data is None or load_financial_data is None or get_connection is None:
        raise HTTPException(
            status_code=503,
            detail="Database functions not loaded. Check data_loader import."
        )

    # Validate minimum data requirements
    if len(request.recent_stock_data) < MIN_HISTORICAL_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient historical data. Please provide at least {MIN_HISTORICAL_DAYS} days of stock data for accurate predictions."
        )


    # Validate data consistency
    validate_data_consistency(request)

    # Convert request data to DataFrames
    stock_df, financial_df = prepare_dataframes_from_request(request)

    logger.info(f"Processing prediction for company {request.company_id} with {len(stock_df)} stock records and {len(financial_df) if financial_df is not None else 0} financial records")

    # Feature Engineering using the FeatureEngineer class
    try:
        X, _, metadata = feature_engineer.create_features_and_labels(
            stock_df=stock_df,
            financial_df=financial_df,
            is_prediction=True,
            training_features=training_features
        )

        if X.empty:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate features from the provided data. Please check data quality and completeness."
            )

        # Use the latest data point for prediction
        features_for_prediction = X.tail(1)
        
        logger.info(f"Generated {len(X.columns)} features for prediction")
        logger.info(f"Using latest data point from {metadata['trade_date'].iloc[-1] if not metadata.empty and 'trade_date' in metadata.columns else 'unknown date'}")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise HTTPException(status_code=500, detail=f"Error during feature engineering: {e}")

    # Make Prediction
    try:
        # Get prediction
        prediction_label = model.predict(features_for_prediction)[0]
        
        # Get prediction probabilities if available (for confidence)
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features_for_prediction)[0]
                confidence = float(max(probabilities))
            except Exception as prob_error:
                logger.warning(f"Could not get prediction probabilities: {prob_error}")

        # Map prediction to action
        predicted_action = map_prediction_to_action(int(prediction_label))
        
        logger.info(f"Prediction completed: {predicted_action} (label: {prediction_label}, confidence: {confidence})")

    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    # Determine the prediction date (next trading day)
    latest_trade_date = stock_df['trade_date'].max().date()

    # Count features by type
    technical_count = len([f for f in feature_engineer.technical_features if f in X.columns])
    fundamental_count = len([f for f in feature_engineer.fundamental_features if f in X.columns])

    return PredictionResponse(
        company_id=request.company_id,
        trade_date=latest_trade_date,
        predicted_action=predicted_action,
        prediction_label=int(prediction_label),
        confidence=confidence,
        features_used=len(X.columns),
        technical_features=technical_count,
        fundamental_features=fundamental_count
    )

@app.get("/predict/company/{company_id}")
async def predict_from_database(company_id: int, days_lookback: int = 60):
    """
    Predict stock action using data directly from the database.
    Fetches recent stock and financial data automatically.
    """
    if model is None or feature_engineer is None:
        raise HTTPException(
            status_code=503, 
            detail="Stock Prediction Model or Feature Engineer not loaded."
        )
    if get_latest_data_for_company is None or get_latest_financial_data_for_company is None:
        raise HTTPException(
            status_code=503,
            detail="Database fetching functions not available. Check data_loader import."
        )

    try:
        # Fetch data from database
        stock_df = get_latest_data_for_company(company_id, days_lookback)
        financial_df = get_latest_financial_data_for_company(company_id, 4)

        if stock_df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No stock data found for company_id {company_id}"
            )

        logger.info(f"Fetched {len(stock_df)} stock records and {len(financial_df) if not financial_df.empty else 0} financial records from database")

        # Use the FeatureEngineer for feature generation
        X, _, metadata = feature_engineer.create_features_and_labels(
            stock_df=stock_df,
            financial_df=financial_df if not financial_df.empty else None,
            is_prediction=True,
            training_features=training_features
        )

        if X.empty:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate features from database data."
            )

        # Make prediction using latest data point
        features_for_prediction = X.tail(1)
        prediction_label = model.predict(features_for_prediction)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features_for_prediction)[0]
                confidence = float(max(probabilities))
            except Exception:
                pass

        predicted_action = map_prediction_to_action(int(prediction_label))
        latest_trade_date = stock_df['trade_date'].max().date()

        # Count features by type
        technical_count = len([f for f in feature_engineer.technical_features if f in X.columns])
        fundamental_count = len([f for f in feature_engineer.fundamental_features if f in X.columns])

        return PredictionResponse(
            company_id=company_id,
            trade_date=latest_trade_date,
            predicted_action=predicted_action,
            prediction_label=int(prediction_label),
            confidence=confidence,
            features_used=len(X.columns),
            technical_features=technical_count,
            fundamental_features=fundamental_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in database prediction for company {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

@app.post("/feedback")
async def record_feedback_api(feedback_item: FeedbackItem):
    """
    API endpoint to record user feedback for extracted financial data.
    This data will be used for retraining ML models.
    """
    if financial_data_extractor is None:
        raise HTTPException(status_code=500, detail="FinancialDataExtractor not initialized. Feedback functionality is unavailable.")

    try:
        financial_data_extractor.record_feedback(
            document_hash=feedback_item.document_hash,
            field_name=feedback_item.field_name,
            extracted_value=feedback_item.extracted_value,
            correct_value=feedback_item.correct_value,
            confidence_score=feedback_item.confidence_score,
            user_id=feedback_item.user_id
        )
        logger.info(f"Received and recorded feedback for document {feedback_item.document_hash}, field {feedback_item.field_name}")
        return {"message": "Feedback recorded successfully."}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")

@app.get("/status")
async def get_status():
    """Get the current status of the API and loaded resources."""
    feature_counts = {}
    if feature_engineer:
        feature_counts = {
            "technical_features": len(feature_engineer.technical_features),
            "fundamental_features": len(feature_engineer.fundamental_features),
            "all_features": len(feature_engineer.all_features)
        }
    
    return {
        "stock_model_loaded": model is not None,
        "feature_engineer_initialized": feature_engineer is not None,
        "training_features_loaded": training_features is not None and len(training_features) > 0,
        "training_features_count": len(training_features) if training_features else 0,
        "financial_data_extractor_initialized": financial_data_extractor is not None,
        "model_path": str(MODEL_PATH),
        "training_features_path": str(TRAINING_FEATURES_PATH),
        "feature_counts": feature_counts
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if model is None or feature_engineer is None or financial_data_extractor is None:
        raise HTTPException(status_code=503, detail="Service not fully ready. Check /status for details.")
    
    return {"status": "healthy", "message": "API is running and ready to serve predictions and feedback"}

@app.get("/features/info")
async def get_feature_info():
    """Get comprehensive information about the features used by the model."""
    if not training_features:
        raise HTTPException(status_code=404, detail="Training features not loaded")
    
    if not feature_engineer:
        raise HTTPException(status_code=503, detail="Feature engineer not initialized")
    
    # Get feature groups from the FeatureEngineer
    feature_groups = feature_engineer.get_feature_importance_groups()
    
    # Separate technical and fundamental features
    technical_features = [f for f in training_features if not f.startswith('financial_')]
    fundamental_features = [f for f in training_features if f.startswith('financial_')]
    
    return {
        "total_features": len(training_features),
        "technical_features_count": len(technical_features),
        "fundamental_features_count": len(fundamental_features),
        "feature_groups": {
            group: {
                "count": len(features),
                "features": features
            }
            for group, features in feature_groups.items()
        },
        "sample_technical_features": technical_features[:10],
        "sample_fundamental_features": fundamental_features[:10] if fundamental_features else [],
        "feature_engineer_status": {
            "technical_features_available": len(feature_engineer.technical_features),
            "fundamental_features_available": len(feature_engineer.fundamental_features),
            "all_features_available": len(feature_engineer.all_features)
        }
    }

@app.post("/features/save")
async def save_current_features():
    """Save current feature information to files."""
    if not feature_engineer:
        raise HTTPException(status_code=503, detail="Feature engineer not initialized")
    
    try:
        # Save feature info
        info_path = Path(MODEL_DIRECTORY) / "feature_info.txt"
        feature_engineer.save_feature_info(str(info_path))
        
        # Save training features
        features_path = Path(MODEL_DIRECTORY) / "training_features.json"
        feature_engineer.save_training_features(str(features_path))
        
        return {
            "message": "Feature information saved successfully",
            "files_created": [str(info_path), str(features_path)],
            "features_saved": len(feature_engineer.all_features)
        }
    except Exception as e:
        logger.error(f"Error saving feature information: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving features: {e}")

# --- Error Handlers ---

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions gracefully"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this from the project root if your MODEL_DIRECTORY is relative
    uvicorn.run(app, host="0.0.0.0", port=8000)
