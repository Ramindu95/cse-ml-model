# predict_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List
import logging
import os
from pathlib import Path
import sys

# Add the directory containing fs_parser.py to the Python path
# This assumes predict_api.py is in data/fs_data/ or a similar structure
# Adjust this path if fs_parser.py is located elsewhere relative to predict_api.py
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from data.fs_data.fs_parser import FinancialParser, ExtractedFinancialData, ExtractionFeedback
except ImportError as e:
    logging.error(f"Failed to import fs_parser: {e}. Ensure fs_parser.py is in the correct path.")
    sys.exit(1) # Exit if essential import fails

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize FinancialParser
# This ensures the database connection and ML models are ready
# Adjust ml_model_dir and db_path if they are different in your setup
ml_model_dir = str(script_dir.parent.parent / "ml_models") # Assuming ml_models is at project root
db_path = str(script_dir / "extraction_learning.db") # Assuming db is in data/fs_data/

# Global instance of FinancialParser
financial_parser: Optional[FinancialParser] = None

@app.on_event("startup")
async def startup_event():
    global financial_parser
    try:
        financial_parser = FinancialParser(ml_model_dir=ml_model_dir, db_path=db_path)
        logger.info("✅ FinancialParser initialized successfully for feedback API.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FinancialParser: {e}")
        # Depending on criticality, you might want to raise an exception or handle gracefully
        # For now, we'll just log and allow the app to start, but feedback won't work.

# Pydantic model for incoming feedback data
class FeedbackItem(BaseModel):
    document_hash: str
    field_name: str
    extracted_value: Any
    correct_value: Any
    confidence_score: Optional[float] = None
    user_id: str = "frontend_user" # Default user ID for feedback

@app.post("/feedback")
async def record_feedback_api(feedback_item: FeedbackItem):
    """
    API endpoint to record user feedback for extracted financial data.
    This data will be used for retraining ML models.
    """
    if financial_parser is None:
        raise HTTPException(status_code=500, detail="FinancialParser not initialized.")

    try:
        # The record_feedback method expects specific types, ensure conversion if needed
        financial_parser.record_feedback(
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

# You can keep your existing prediction endpoint here if you have one
# For example:
# @app.post("/predict")
# async def predict_financial_data(data: SomePredictionInputModel):
#     # Your existing prediction logic
#     pass

# To run this FastAPI app:
# uvicorn predict_api:app --reload --port 8000
