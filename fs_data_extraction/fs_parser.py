import re
import logging
import json # Added for json.dumps in example usage
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from dotenv import load_dotenv
import time
import random
import sqlite3 # For SQLite database operations
from pathlib import Path # For path manipulation
from pytesseract import Output # Import Output for pytesseract.image_to_data
import sqlite3 # For LearningDatabase

from transformers import AutoTokenizer, AutoModelForTokenClassification # Or other specific AutoModel types like AutoModelForQuestionAnswering
import torch # Required if your Hugging Face model uses PyTorch


# Configure logging for the parser
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # Use logger for this module

load_dotenv()  # Load environment variables from .env

# --- IMPORTANT: If you are training your own ML model, you will NOT use GEMINI_API_KEY. ---
# This variable is now commented out as it's not needed for a custom ML model.
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY not found in .env file in fs_parser.py. Please ensure it's set.")

# --- Placeholder for loading your custom ML model ---
# You would replace this with actual code to load your trained model.
# For example:
# from your_ml_library import load_model
# global_ml_model = None
# try:
#     global_ml_model = load_model("path/to/your/trained_financial_extractor_model.pkl") # or .pt, .h5, etc.
#     logger.info("Custom ML model loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load custom ML model: {e}. Ensure model path is correct.")

@dataclass
class FinancialMetrics:
    """Data class for financial statement line items"""
    revenue: Optional[float] = None
    profit_for_period: Optional[float] = None
    operating_profit: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    basic_eps: Optional[float] = None
    gross_profit: Optional[float] = None # Added
    ebitda: Optional[float] = None # Added
    net_cash_flow: Optional[float] = None # Added (from operating activities)

    # --- New fields added based on user's request ---
    profit_before_tax: Optional[float] = None
    income_tax_expense: Optional[float] = None
    other_income: Optional[float] = None
    administrative_expenses: Optional[float] = None
    distribution_expenses: Optional[float] = None
    total_comprehensive_income: Optional[float] = None
    cash_generated_from_operations: Optional[float] = None
    property_plant_equipment_assets: Optional[float] = None # Specific for balance sheet
    intangible_assets_balance: Optional[float] = None # Specific for balance sheet
    inventories_balance: Optional[float] = None # Specific for balance sheet
    trade_and_other_receivables_balance: Optional[float] = None # Specific for balance sheet
    cash_and_cash_equivalents_balance: Optional[float] = None # Specific for balance sheet
    trade_payables_balance: Optional[float] = None # Specific for balance sheet
    income_tax_payable_balance: Optional[float] = None # Specific for balance sheet
    net_assets_per_share: Optional[float] = None
    # --- End of new fields ---

@dataclass
class Shareholder:
    """Data class for shareholder information"""
    rank: int 
    name: str
    shares: Optional[int] = None
    percentage: Optional[float] = None

@dataclass
class Director:
    """Data class for director information"""
    name: str
    role: Optional[str] = None

@dataclass
class ExtractedFinancialData:
    """Main data structure for extracted financial information"""
    company_name: Optional[str] = None
    company_symbol: Optional[str] = None
    report_type: Optional[str] = None
    report_date: Optional[datetime] = None
    currency_unit: Optional[str] = None
    scale_factor: float = 1.0
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    shareholders: list[Shareholder] = field(default_factory=list)
    directors: list[Director] = field(default_factory=list)
    audit_status: Optional[str] = "Unknown"
    errata_notice: Optional[str] = None
    contingent_liabilities: Optional[str] = None
    events_after_reporting: Optional[str] = None
    extraction_confidence: float = 0.0
    document_hash: Optional[str] = None
    extraction_method: Optional[str] = "Rule-based"
    processing_time: Optional[float] = None
    # --- ADD THIS LINE HERE ---
    report_filename: Optional[str] = None # Add this field
    # --- END OF ADDED LINE ---
    
    def __post_init__(self):
        if self.financial_metrics is None:
            self.financial_metrics = FinancialMetrics()
        if self.shareholders is None:
            self.shareholders = []
        if self.directors is None:
            self.directors = []

@dataclass
class ExtractionFeedback:
    """Data class for user feedback on extracted data."""
    document_hash: str
    field_name: str
    extracted_value: Any
    correct_value: Any
    confidence_score: Optional[float] = None
    timestamp: datetime = datetime.now()
    user_id: str = "system"

class LearningDatabase:
    """
    Manages the SQLite database for storing extracted data and user feedback.
    This database will be used for retraining ML models.
    """
    def __init__(self, db_path: str = "extraction_learning.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_database()

    def _init_database(self):
        """Initializes the SQLite database schema if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Table for storing raw extractions (snapshot of ExtractedFinancialData)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    document_hash TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    extraction_data TEXT NOT NULL, -- Store full ExtractedFinancialData as JSON
                    confidence_score REAL,
                    extraction_method TEXT,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table for storing user feedback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    extracted_value TEXT, -- Store as TEXT to handle various types (numbers, strings, JSON)
                    correct_value TEXT NOT NULL, -- Store as TEXT
                    confidence_score REAL,
                    user_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_hash) REFERENCES extractions (document_hash)
                )
            """)
            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database: {e}")
        finally:
            if conn:
                conn.close()

    def get_connection(self):
        """Returns a new database connection."""
        return sqlite3.connect(self.db_path)

    def store_extraction(self, data: ExtractedFinancialData):
        """Stores an extracted financial data object into the database."""
        conn = self.get_connection()
        try:
            # Convert dataclass to dictionary and then to JSON string
            data_dict = asdict(data)
            # Ensure datetime objects are converted to string for JSON serialization
            data_dict['timestamp'] = data_dict.get('timestamp', datetime.now()).isoformat() 
            
            # Use JSON.dumps to store complex objects like FinancialMetrics, Shareholders, Directors
            # as part of the main extraction_data blob.
            # This makes it easier to retrieve the full context later.
            extraction_json = json.dumps(data_dict, default=str) # default=str handles datetime objects

            conn.execute("""
                INSERT OR REPLACE INTO extractions 
                (document_hash, company_name, extraction_data, confidence_score, extraction_method, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.document_hash,
                data.company_name,
                extraction_json,
                data.extraction_confidence,
                data.extraction_method,
                data.processing_time
            ))
            conn.commit()
            self.logger.info(f"Stored extraction for document: {data.company_name} ({data.document_hash})")
        except sqlite3.Error as e:
            self.logger.error(f"Error storing extraction for {data.document_hash}: {e}")
        finally:
            conn.close()

    def store_feedback(self, feedback: ExtractionFeedback):
        """Stores user feedback for a specific field."""
        conn = self.get_connection()
        try:
            conn.execute("""
                INSERT INTO feedback 
                (document_hash, field_name, extracted_value, correct_value, confidence_score, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.document_hash,
                feedback.field_name,
                json.dumps(feedback.extracted_value, default=str), # Store as JSON string
                json.dumps(feedback.correct_value, default=str), # Store as JSON string
                feedback.confidence_score,
                feedback.user_id,
                feedback.timestamp.isoformat()
            ))
            conn.commit()
            self.logger.info(f"Stored feedback for document {feedback.document_hash}, field {feedback.field_name}")
        except sqlite3.Error as e:
            self.logger.error(f"Error storing feedback for {feedback.document_hash} - {feedback.field_name}: {e}")
        finally:
            conn.close()

    def get_feedback_for_training(self, field_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves feedback entries for a specific field,
        useful for training ML models.
        """
        conn = self.get_connection()
        feedback_entries = []
        try:
            cursor = conn.execute("""
                SELECT document_hash, field_name, extracted_value, correct_value, confidence_score, user_id, timestamp
                FROM feedback
                WHERE field_name = ?
                ORDER BY timestamp DESC
            """, (field_name,))
            
            for row in cursor.fetchall():
                feedback_dict = {
                    'document_hash': row[0],
                    'field_name': row[1],
                    'extracted_value': json.loads(row[2]), # Parse back from JSON string
                    'correct_value': json.loads(row[3]), # Parse back from JSON string
                    'confidence_score': row[4],
                    'user_id': row[5],
                    'timestamp': datetime.fromisoformat(row[6])
                }
                feedback_entries.append(feedback_dict)
            self.logger.info(f"Retrieved {len(feedback_entries)} feedback entries for {field_name}.")
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving feedback for {field_name}: {e}")
        finally:
            conn.close()
        return feedback_entries

    def get_extraction_data(self, document_hash: str) -> Optional[ExtractedFinancialData]:
        """Retrieves a full ExtractedFinancialData object by its hash."""
        conn = self.get_connection()
        try:
            cursor = conn.execute("SELECT extraction_data FROM extractions WHERE document_hash = ?", (document_hash,))
            row = cursor.fetchone()
            if row:
                # Parse the JSON string back to a dictionary
                data_dict = json.loads(row[0])
                # Reconstruct ExtractedFinancialData dataclass
                # Handle nested dataclasses manually if json.loads doesn't do it automatically
                if 'financial_metrics' in data_dict and isinstance(data_dict['financial_metrics'], dict):
                    data_dict['financial_metrics'] = FinancialMetrics(**data_dict['financial_metrics'])
                if 'shareholders' in data_dict and isinstance(data_dict['shareholders'], list):
                    data_dict['shareholders'] = [Shareholder(**sh) for sh in data_dict['shareholders']]
                if 'directors' in data_dict and isinstance(data_dict['directors'], list):
                    data_dict['directors'] = [Director(**d) for d in data_dict['directors']]
                
                return ExtractedFinancialData(**data_dict)
            return None
        except (sqlite3.Error, json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Error retrieving or parsing extraction data for {document_hash}: {e}")
            return None
        finally:
            conn.close()


class FinancialDataExtractor:
    """
    Enhanced financial data extractor with better structure and error handling.
    Can also be initialized with a path to ML models and a database for feedback.
    """
    
    def __init__(self, ml_model_dir: Optional[str] = None, db_path: Optional[str] = None):
        """
        Initialize the extractor with optional custom ML model and a learning database.
        
        Args:
            ml_model_dir: Path to the directory containing trained ML models.
            db_path: Path to the SQLite database for storing extractions and feedback.
        """
        self.logger = logging.getLogger(self.__class__.__name__) # Initialize logger for the class
        self.model = None # Placeholder for a potential overall ML model for extraction
        self.tokenizer = None # Add tokenizer here
        self.model_loaded = False
        self.scale_factor = 1.0 # Default scale factor
        self.currency_unit = "LKR thousands" # Default currency unit
        
        self.db = None
        if db_path:
            try:
                self.db = LearningDatabase(db_path)
                self.logger.info(f"LearningDatabase initialized at {db_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LearningDatabase at {db_path}: {e}")
        
        # Placeholder for loading individual field-specific ML models if ml_model_dir is provided
        if ml_model_dir:
            # In a full implementation, you'd load your FinancialMLModelManager here
            # and use it for predictions within the extraction process.
            # from ml_models_manager import FinancialMLModelManager
            # self.ml_manager = FinancialMLModelManager(model_dir=ml_model_dir)
            # self.logger.info(f"MLModelManager initialized from {ml_model_dir}")
            # Call the method to load the custom ML model
            self.model_loaded = self._load_custom_model(ml_model_dir)
            if self.model_loaded:
                self.logger.info(f"ML model successfully initialized from {ml_model_dir}.")
            else:
                self.logger.warning(f"Failed to load ML model from {ml_model_dir}. Proceeding with rule-based extraction.")
            

    def _load_custom_model(self, model_path: str) -> bool:
        """
        Loads a custom ML model for financial data extraction.
        This example assumes a Hugging Face Transformers model fine-tuned for a task
        like token classification (e.g., for Named Entity Recognition or key-value extraction).

        Args:
            model_path: The path to the directory containing the saved model and tokenizer.
                        This directory should contain files like config.json, pytorch_model.bin, tokenizer.json.
        """
        try:
            # Ensure transformers and torch are installed if you plan to use this:
            # pip install transformers torch

            # Load tokenizer and model
            # The specific AutoModel class (e.g., AutoModelForTokenClassification)
            # depends on how your model was trained.
            # For NER or key-value extraction, TokenClassification or QuestionAnswering models are common.
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)

            # Set model to evaluation mode (important for inference)
            self.model.eval()

            # Optional: Move model to GPU if available for faster inference
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self.logger.info(f"Custom ML model and tokenizer loaded successfully from: {model_path} on device: {self.device}")
            self.model_loaded = True
            return True

        except ImportError:
            self.logger.error("Hugging Face Transformers or PyTorch not installed. Please install them: pip install transformers torch")
            self.model_loaded = False
            return False
        except Exception as e:
            self.logger.error(f"Failed to load custom ML model from {model_path}: {e}")
            self.model_loaded = False
            return False
    
    def _extract_company_info(self, text: str) -> tuple[str, str]:
        """Extract company name and symbol from text"""
        # Enhanced company detection logic
        company_patterns = [
            r"([A-Z][A-Za-z\s&]+(?:PLC|Limited|Ltd|Corporation|Corp|Inc))",
            r"Digital Mobility Solutions Lanka PLC",
            r"PickMe",
            r"CHEMANEX PLC",
            r'(?:Company|Corporation|Corp|Inc|Ltd|Limited|LLC|LLP|Partnership)[:\s]*([^.\n]+)',
            r'([A-Z][a-zA-Z\s&]+(?:Company|Corporation|Corp|Inc|Ltd|Limited|LLC|LLP))',
            r'(?:Entity|Business|Organization)[:\s]*([^.\n]+)',
            r'(?:Name|Company Name|Business Name)[:\s]*([^.\n]+)',
            r'^([A-Z][A-Za-z\s&,.-]+(?:Company|Corporation|Corp|Inc|Ltd|Limited|LLC|LLP))',
        ]
        
        symbol_patterns = [
            r"\b([A-Z]{3,5})(?:\.[NX]\d{4})?\b",  # 3-5 letter stock symbols, e.g., PKME or ABAN.N0000
            r"PKME",  # Specific symbol
            r"CHMX" # Added for specific company in the provided text
        ]
        
        company_name = "Unknown"
        company_symbol = "Unknown"
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_name = match.group(1).strip()
                break
        
        for pattern in symbol_patterns:
            match = re.search(pattern, text)
            if match:
                company_symbol = match.group(1)
                break
                
        return company_name, company_symbol
    
    def _parse_currency_unit_and_scale(self, text: str):
        """
        Parses the currency unit and scale factor from the document text.
        This should be called once at the beginning of the extraction process.
        """
        # Look for phrases like "(In Rs. 1000)", "(In Rs. Mn)", "Rs. '000", "RUPEES thousands"
        if re.search(r"\(In Rs\.?\s*1000\)|Rs\.?\s*'000|RUPEES\s*thousands", text, re.IGNORECASE): # ADDED 'RUPEES\s*thousands'
            self.scale_factor = 1000.0
            self.currency_unit = "Rs. 1000" # Or "RUPEES thousands" if you prefer
        elif re.search(r"\(In Rs\.\s*Mn\)", text, re.IGNORECASE):
            self.scale_factor = 1_000_000.0
            self.currency_unit = "Rs. Mn"
        else:
            self.scale_factor = 1.0 # Default to 1 if no specific scale is found
            self.currency_unit = "Rs." # Default currency unit

        self.logger.info(f"Detected currency unit: {self.currency_unit}, Scale factor: {self.scale_factor}")

    def _clean_financial_value(self, value_str: str) -> Optional[float]:
        """
        Clean and convert financial value strings to float, applying the detected scale factor.
        Handles parentheses for negatives and attempts to correct common OCR errors
        where a period might be used as a thousands separator.
        """
        if not value_str:
            return None

        original_value_str = value_str.strip()
        cleaned_str = original_value_str

        # 1. Handle parentheses for negative numbers FIRST
        is_negative = False
        if cleaned_str.startswith('(') and cleaned_str.endswith(')'):
            is_negative = True
            cleaned_str = cleaned_str[1:-1] # Remove parentheses

        # 2. Remove spaces
        cleaned_str = cleaned_str.replace(' ', '')

        # 3. Handle commas (thousands separators) - always remove them
        cleaned_str = cleaned_str.replace(',', '')

        # 4. Heuristic for period as thousands separator (e.g., "55.561" should be 55561)
        # If a number has multiple periods, or a single period followed by exactly 3 digits,
        # it's highly likely the periods are thousands separators and should be removed.
        if cleaned_str.count('.') > 1:
            # Multiple periods, assume all are thousands separators
            cleaned_str = cleaned_str.replace('.', '')
        elif cleaned_str.count('.') == 1:
            parts = cleaned_str.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                # Single period followed by exactly 3 digits (e.g., "55.561")
                # Assume it's a thousands separator and remove the period.
                cleaned_str = cleaned_str.replace('.', '')

        # Final sanitization: remove any remaining non-numeric characters except for a single decimal point
        # This regex allows digits and a single period.
        final_numeric_str = re.sub(r'[^\d.]', '', cleaned_str)

        # Ensure only one decimal point remains if multiple were somehow introduced (edge case)
        if final_numeric_str.count('.') > 1:
            parts = final_numeric_str.split('.')
            final_numeric_str = parts[0] + '.' + ''.join(parts[1:]) # Keep first part, then concatenate rest after first decimal

        if not final_numeric_str:
            self.logger.debug(f"Could not convert value after cleaning: '{original_value_str}' -> '{final_numeric_str}' (empty or invalid)")
            return None

        try:
            value = float(final_numeric_str)
            if is_negative:
                value *= -1

            # Apply the scale factor.
            return value * self.scale_factor

        except ValueError:
            self.logger.debug(f"Could not convert final cleaned value to float: '{final_numeric_str}' (from original: '{original_value_str}')")
            return None

    def _extract_financial_metrics(self, text: str) -> FinancialMetrics:
        """
        Extract financial metrics using improved regex patterns that are more precise
        in capturing only the current year's value.
        """
        metrics = FinancialMetrics()

        # Regex to capture a number, allowing for:
        # - Optional leading parenthesis for negatives
        # - Digits
        # - Optional thousands separators (comma or period, handled by _clean_financial_value)
        # - Optional decimal part
        # - Optional trailing parenthesis for negatives
        # This pattern is specifically designed to capture *one* number.
        num_pattern = r"([\(\-]?\d{1,3}(?:[,\.]\d{3})*(?:[\.]\d+)?[\)]?)"

        # Patterns for financial data extraction.
        # The key change here is to make the regex less greedy after the metric name,
        # and more specific about what comes between the label and the number.
        # We're looking for a common pattern like "Label [optional spaces/notes] Number"
        patterns = {
            # Assuming 'num_pattern' is defined as above

            'revenue': [
                # 1. Very specific: "Revenue" at the start of a line, followed by the number
                #    This is good if 'Revenue' is typically a line item by itself.
                r"^\s*Revenue\s*.*?" + num_pattern,
                r"^\s*Total Revenue\s*.*?" + num_pattern,
                r"^\s*Net Revenue\s*.*?" + num_pattern,
                r"^\s*Turnover\s*.*?" + num_pattern,

                # 2. Revenue followed directly by the number, potentially with a currency/period
                #    This is common in tables or summary lines.
                r"Revenue\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
                r"Total Revenue\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
                r"Net Revenue\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
                r"Turnover\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
                
                # 3. Revenue mentioned in a sentence, followed by 'of' or 'was'
                r"(?:Revenue|Turnover)\s*(?:of|was|stood at|reached|recorded)\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,

                # 4. Revenue growth mentioned (captures the original revenue figure if present nearby)
                #    Looks for "Revenue" then "grew by" or "increased by" and then *another* number (the growth percentage),
                #    and tries to capture the initial revenue number after that. This is more complex and might need re-evaluation
                #    depending on if you want the *growth percentage* or the *absolute revenue figure* when growth is discussed.
                #    For the absolute figure near growth discussion:
                r"(?:Revenue|Turnover).*?(?:grew by|increased by|up by|growth of)\s*\d+\.?\d*%\s*(?:to|reaching|recorded)\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,

                # 5. Patterns where "Revenue" might be abbreviated or part of a larger term
                r"Sales Revenue\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
                r"Group Revenue\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,

                # 6. Capturing numbers appearing *after* common indicators like 'in', 'for' following 'Revenue'
                r"Revenue\s*(?:in|for)\s*\d{4}\s*(?:of|was)?\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,

                # 7. Looking for common suffixes indicating it's a financial statement section
                r"(?:Statement of Comprehensive Income|Income Statement|Profit or Loss Statement)[\s\S]*?(?:Revenue|Turnover)\s*(?:Rs\.|LKR|USD|\$|€|£)?\s*" + num_pattern,
            ],
            'cost_of_revenue': [
                r"Cost of Revenue\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cost of Goods Sold\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cost of Sales\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cost of Goods Sold \(COGS\)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cost of Sales \(COS\)\s*(?:[^\d\n]*?)\s*" + num_pattern,
            ],
            'gross_profit': [
                r"Gross Profit\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Gross Profit/(Loss)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Gross Profit before Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
            ],
            'operating_expenses': [
                r"Operating Expenses\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Operating Expenses \(OPEX\)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Total Operating Expenses\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Total Operating Expenses \(OPEX\)\s*(?:[^\d\n]*?)\s*" + num_pattern,
            ],
            'interest_expense': [
                r"Interest Expense\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Interest Expense/(Income)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Finance Costs\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Finance Costs/(Income)\s*(?:[^\d\n]*?)\s*" + num_pattern,
            ],
            'net_income': [
                r"Net Income\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net Profit\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net Profit/(Loss)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit for the year\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit for the period\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'net_income_after_tax': [
                r"Net Income after Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net Profit after Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit after Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit after Tax from continuing operations\s*(?:[^\d\n]*?)\s*" + num_pattern,
            ],
            'net_income_before_tax': [
                r"Net Income before Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net Profit before Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit before Tax\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit before Tax from continuing operations\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'profit_for_period': [
                r"Profit for the period\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net Income\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Profit after Tax(?: from continuing operations)?\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'operating_profit': [
                r"Operating Profit(?:/Loss)?\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Results from operating activities\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"EBIT\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'basic_eps': [
                r"Basic(?:/Diluted)? earnings per share \(Rs\.\)\s*([\d\.]+)", # EPS is often just a float
                r"Basic\s*([\d\.]+)", # For simpler "Basic 0.73" lines
                r"Basic earnings per share \(Rs\.\)\s*([\d.]+)",
                r"Basic\s*([\d.]+)", # Fallback for simpler "Basic 0.73"
                r"Basic/Diluted earnings per share \(Rs\.\)\s*(\d+\.\d{2})\s*(\d+\.\d{2})"
            ],
            'eps_note_pattern'  : [
                r"Basic earnings per share \(Rs\.\)\s*[\d\.]+\s*Note\s*:\s*([\d\.]+)", # Note pattern for EPS
                r"Basic earnings per share \(Rs\.\)\s*[\d\.]+\s*Note\s*:\s*([\d\.]+)", # Note pattern for EPS
                r"Basic earnings per share \(Rs\.\)\s*[\d\.]+\s*Note\s*:\s*([\d\.]+)", # Note pattern for EPS
                r"Basic earnings per share \(Rs\.\)\s*[\d\.]+\s*Note\s*:\s*([\d\.]+)", # Note pattern for EPS
                r"Basic earnings per share \(Rs\.\)\s*(\d+\.\d{2})\s*(\d+\.\d{2})\s*(\d+\.\d{2})\s*(\d+\.\d{2})"
            ],
            'total_assets': [
                r"Total Assets\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'total_liabilities': [
                r"Total Liabilities\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'total_equity': [
                r"Total Equity\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'gross_profit': [
                r"Gross profit\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'ebitda': [
                r"EBITDA\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'net_cash_flow': [
                r"Net cash \(outflow\)/ inflow from operating activities\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Net cash flow from/\(used in\) operating activities\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'profit_before_tax': [
                r"Profit before tax\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'income_tax_expense': [
                r"Income tax expense\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Income tax expenses\s*(?:[^\d\n]*?)\s*" + num_pattern, # Added 'expenses' plural
                r"Tax \(expense\)\s*/\s*Reversal\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'other_income': [
                r"Other operating income\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Other income\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'administrative_expenses': [
                r"Administrative expenses\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'distribution_expenses': [
                r"Distribution expenses\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'total_comprehensive_income': [
                r"Total comprehensive income for the period\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Total comprehensive income, net of tax\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'cash_generated_from_operations': [
                r"Cash generated from operations\s*\(Note A\)\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cash generated from operations\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'property_plant_equipment_assets': [
                r"Property, plant & equipment\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'intangible_assets_balance': [
                r"Intangible assets\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'inventories_balance': [
                r"Inventories\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'trade_and_other_receivables_balance': [
                r"Trade and other receivables\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'cash_and_cash_equivalents_balance': [
                # Make this more robust, looking for "Cash & cash equivalents" followed by a number
                # and ensuring it's not picking up a small number from a note or header.
                r"Cash & cash equivalents\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cash and cash equivalents\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Cash & cash equivalents\s*([\(\-]?\d{1,3}(?:,\d{3})*(?:[\.]\d+)?[\)]?)",
                r"Cash and cash equivalents\s*([\(\-]?\d{1,3}(?:,\d{3})*(?:[\.]\d+)?[\)]?)"
            ],
            'trade_payables_balance': [
                r"Trade payable\s*(?:[^\d\n]*?)\s*" + num_pattern,
                r"Trade and other payables\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'income_tax_payable_balance': [
                r"Income tax payable\s*(?:[^\d\n]*?)\s*" + num_pattern
            ],
            'net_assets_per_share': [
                r"Net Assets per share \(Rs\.\)\s*([\d\.]+)",
                r"Net assets per share\s*([\d\.]+)"
            ]
        }

        # --- ADD THIS ML INTEGRATION BLOCK HERE ---
        if self.model_loaded and self.tokenizer:
            self.logger.info("ML model loaded. Attempting ML-driven extraction for financial metrics.")
            
            # Example for 'revenue' using a QA model
            question_for_revenue = "What is the revenue for the current period?"
            
            try:
                # Tokenize the input (document text and question)
                inputs = self.tokenizer(question_for_revenue, text, return_tensors="pt", truncation=True, max_length=512)
                
                # Move inputs to the same device as the model (CPU/GPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Perform inference
                with torch.no_grad(): # No need to calculate gradients for inference
                    outputs = self.model(**inputs)

                # Get the predicted start and end logits
                answer_start_index = outputs.start_logits.argmax()
                answer_end_index = outputs.end_logits.argmax()

                # Convert token indices back to actual text span
                # Ensure start is before end
                if answer_start_index <= answer_end_index:
                    answer_tokens = inputs.input_ids[0][answer_start_index:answer_end_index + 1]
                    answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    
                    if answer_text:
                        cleaned_value = self._clean_financial_value(answer_text)
                        if cleaned_value is not None:
                            metrics.revenue = cleaned_value
                            self.logger.info(f"ML Extracted revenue: {cleaned_value} (from raw: '{answer_text}')")
                        else:
                            self.logger.warning(f"ML extracted revenue but failed to clean/convert: '{answer_text}'")
                    else:
                        self.logger.debug("ML model extracted an empty answer for revenue.")
                else:
                    self.logger.debug("ML model predicted an invalid span for revenue (start > end).")

            except Exception as e:
                self.logger.error(f"Error during ML extraction for revenue: {e}")
                self.logger.warning("Falling back to regex for revenue due to ML error.")
                # If ML fails, fall back to regex for this specific metric
                for pattern in patterns['revenue']:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value_str = match.group(1).strip()
                        cleaned_value = self._clean_financial_value(value_str)
                        if cleaned_value is not None:
                            metrics.revenue = cleaned_value
                            self.logger.info(f"Regex Extracted revenue: {cleaned_value} (from raw: '{value_str}')")
                            break
        else:
            self.logger.info("ML model not loaded or available. Proceeding with regex-based extraction.")
        # --- END OF ML INTEGRATION BLOCK ---

        # Loop through all other metrics using regex (or replace with ML as needed)
        for metric, pattern_list in patterns.items():
            # Skip 'revenue' if it was already handled by ML and successfully extracted
            if metric == 'revenue' and metrics.revenue is not None:
                continue

            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value_str = match.group(1).strip()
                    self.logger.debug(f"Found match for {metric} with pattern '{pattern}': Raw value '{value_str}'")
                    cleaned_value = self._clean_financial_value(value_str)
                    if cleaned_value is not None:
                        setattr(metrics, metric, cleaned_value)
                        self.logger.info(f"Extracted {metric}: {cleaned_value}")
                        break # Move to next metric once a match is found
                    else:
                        self.logger.warning(f"Failed to clean/convert value for {metric}: '{value_str}'")
                else:
                    self.logger.debug(f"No match for {metric} with pattern '{pattern}'")

        return metrics
    
    def _clean_financial_value(self, value_str: str) -> Optional[float]:
        """
        Clean and convert financial value strings to float, applying the detected scale factor.
        """
        if not value_str:
            return None
            
        try:
            # Remove all non-numeric characters except decimal point and minus sign
            # This is more aggressive to handle OCR errors or unexpected characters
            cleaned = re.sub(r'[^\d\.\-]', '', value_str.replace(',', '').replace(' ', ''))
            
            # Handle parentheses as negative values AFTER initial cleaning
            if '(' in value_str and ')' in value_str:
                cleaned = '-' + cleaned.replace('(', '').replace(')', '')
            
            if cleaned and cleaned != '-':
                value = float(cleaned)
                # Apply the scale factor
                return value * self.scale_factor
                
        except ValueError:
            self.logger.debug(f"Could not convert value: {value_str}")
            
        return None

    def _extract_contingent_liabilities(self, text: str) -> Optional[str]:
        """
        Extracts the specific paragraph regarding contingent liabilities.
        Uses precise start and end markers.
        """
        # Define the start marker for the contingent liabilities paragraph.
        start_marker = r"There are no material contingent liabilities as at the reporting date, which require adjustment and/or disclosure in the Financial Statements\."

        # Define common end markers for such a note, like the start of accounting policies
        # or other distinct sections that follow. Prioritize specific section headers.
        end_markers = [
            r"As the float adjusted market capitalisation is below Rs\.2\.5 Bn,", # This is the start of the next sentence/paragraph
            r"Based on the fair valuation methodology,", # The start of the next paragraph
            r"The accounting policies followed in the preparation of the Interim Statement of Comprehensive Income",
            r"\n\n\s*(\d+\.\d+\s+)?Notes to the Financial Statements(?: \(Cont\.\))?", # Generic end for notes (e.g., "3.1 Notes to the Financial Statements")
            r"\n\n\s*[A-Z][A-Z\s]+(?:\n|$)" # Generic pattern for a new section header (ensure double newline)
        ]

        # Combine end markers into a single regex pattern using OR
        # Using a non-greedy match (.*?) followed by the non-capturing group (?:...)
        end_pattern = "|".join(end_markers)

        # Search for the pattern from the start marker until any of the end markers
        # Use re.DOTALL to allow '.' to match newlines
        # The key is to capture the content *between* the start and the *first* end marker.
        match = re.search(
            f"({start_marker}.*?)(?:{end_pattern})",
            text, re.IGNORECASE | re.DOTALL
        )

        if match:
            # Return the captured group, which is the content between start and end.
            return match.group(1).strip()
        else:
            self.logger.info("Contingent liabilities paragraph not found with precise markers. Trying fallback.")
            # Fallback to a less precise search if the exact paragraph isn't found
            fallback_match = re.search(
                r"There are no material contingent liabilities.*?(?:\n\n|\Z)",
                text, re.IGNORECASE | re.DOTALL
            )
            if fallback_match:
                self.logger.warning("Using fallback for contingent liabilities, may be less precise.")
                return fallback_match.group(0).strip() # group(0) is the whole match
        return None

    def _extract_shareholders(self, text: str) -> List[Shareholder]:
        """
        Extracts shareholder information from the text, including a rank.
        Assumes a table-like structure for the largest shareholders.
        This version uses a broader regex for section detection.
        """
        shareholders = []
        # Look for the section header "Twenty largest shareholders" or just "Shareholder"
        # and capture lines until a new section begins or end of document.
        # Adjusted end markers for the shareholder section
        shareholder_section_match = re.search(
            r"(Twenty largest shareholders|Shareholder.*?)(?:Notes to the Financial Statements|Public Holding|\n\s*\d+\.\s+[A-Z][a-zA-Z\s]+|\Z)", # Added 'Shareholder' here
            text, re.IGNORECASE | re.DOTALL
        )

        if shareholder_section_match:
            section_text = shareholder_section_match.group(1)
            self.logger.info("Found shareholders section. Attempting to parse lines.")

            # Find the actual table content, excluding headers.
            # The header "Name of the shareholder No of shares %" is on Page 11.
            table_start_match = re.search(r"Name of the shareholder\s+No of shares\s+%", section_text, re.IGNORECASE)
            if table_start_match:
                table_content = section_text[table_start_match.end():].strip()
            else:
                table_content = section_text # If header not found, take whole section

            lines = table_content.split('\n')

            # Pattern to capture Name, Shares, and Percentage
            # This pattern is more robust for names that might contain numbers or special chars.
            # It expects Name, then a number (shares), then a float (percentage).
            shareholder_pattern = re.compile(
                r"^\s*(.+?)\s+(\d{1,3}(?:,\d{3})*)\s+([\d.]+)\s*$", # Added ^ and $ for full line match
                re.IGNORECASE
            )

            rank = 1 # Initialize rank

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip lines that are just numbers (like totals) or headers
                if re.match(r"^\s*\d{1,3}(?:,\d{3})*\s+[\d.]+$", line) or \
                   re.match(r"^\s*Name of the shareholder", line, re.IGNORECASE) or \
                   re.match(r"^\s*Total", line, re.IGNORECASE): # Also skip "Total" lines
                    self.logger.debug(f"Skipping line (total/header): {line}")
                    continue

                match = shareholder_pattern.match(line) # Use match for start-of-string match
                if match:
                    name = match.group(1).strip()
                    shares_str = match.group(2).replace(',', '') # Remove commas for int conversion
                    percentage_str = match.group(3)

                    try:
                        shares = int(shares_str)
                        percentage = float(percentage_str)
                        shareholders.append(Shareholder(rank=rank, name=name, shares=shares, percentage=percentage))
                        rank += 1 # Increment rank for the next shareholder
                        self.logger.info(f"Parsed shareholder: {name}, Shares: {shares}, %: {percentage}")
                    except ValueError:
                        self.logger.warning(f"Failed to parse shareholder data from line: {line} (Value Error)")
                else:
                    self.logger.debug(f"No shareholder pattern match for line: {line}")
        else:
            self.logger.info("Shareholder section not found using refined markers.")

        return shareholders

    def _extract_directors(self, text: str) -> List[Director]:
        """
        Extracts director names and their roles from the text.
        """
        directors = []
        # Look for the "Directors" or "Board of Directors" section
        # and capture lines until a new section (e.g., Company Secretary, Auditors, Legal Advisers)
        director_section_match = re.search(
            r"(?:Directors|Board of Directors)\s*(.*?)(?:Company Secretary|Auditors|Legal Advisers|Subsidiary Companies|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )

        if director_section_match:
            section_text = director_section_match.group(1).strip()
            lines = section_text.split('\n')

            # Define common roles to look for
            common_roles = ["Chairman", "Director", "Chief Executive Officer", "CEO",
                            "Chief Financial Officer", "CFO", "Company Secretary"]

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Clean up common PDF/OCR artifacts like extra quotes or commas
                line = line.replace('"', '').strip()

                name = line
                role = None

                # Try to find a role within the line or immediately following the name
                for r in common_roles:
                    if r.lower() in line.lower():
                        role = r
                        # Remove the role from the name for cleaner name extraction
                        name = re.sub(r'\b' + re.escape(r) + r'\b', '', name, flags=re.IGNORECASE).strip()
                        break # Found a role, move on

                # If no specific role found but it's a name, assign "Director" as a default
                # This is a heuristic; might need refinement.
                if not role and len(name.split()) >= 2: # Simple check for likely a name
                     # Check if it's not just a single initial or very short string
                    if len(name) > 3 and not name.isupper(): # Avoid capturing just initials like "P R"
                         role = "Director"


                # Further clean name to remove extra spaces or leading/trailing non-alpha
                name = re.sub(r'[^a-zA-Z\s\.]', '', name).strip()
                # Remove any remaining "Mr." or "Mrs."
                name = re.sub(r'^(Mr|Mrs|Ms|Dr)\.\s*', '', name, flags=re.IGNORECASE).strip()

                if name: # Only add if name is not empty after cleaning
                    directors.append(Director(name=name, role=role))
        else:
            self.logger.info("Directors section header not found.")

        return directors

    def _extract_report_info(self, text: str) -> tuple[str, str]:
        """Extract report type and date"""
        report_type = "Unknown"
        report_date = "Unknown"
        
        # Detect report type
        if re.search(r"interim", text, re.IGNORECASE):
            report_type = "Interim"
        elif re.search(r"annual", text, re.IGNORECASE):
            report_type = "Annual"
        elif re.search(r"quarterly", text, re.IGNORECASE): # Added quarterly
            report_type = "Quarterly"
        
        # Extract date patterns (ordered by specificity/reliability)
        date_patterns = [
            # "For the three months ended 30th June 2025" or "As at 30th June 2025"
            # Added \s+ to allow for newlines/multiple spaces between date components
            r"(?:as at|for the (?:three|six|nine|twelve) months ended|for the year ended)\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})", 
            r"(?:as at|for the (?:three|six|nine|twelve) months ended|for the year ended)\s*(\w+\s+\d{1,2},?\s+\d{4})",
            # Direct "Day Month Year" or "Month Day, Year" - allowing flexible whitespace
            r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",  # "30th June 2025"
            r"(\w+\s+\d{1,2},?\s+\d{4})",  # "June 30, 2025"
            # YYYY-MM-DD or DD/MM/YYYY
            r"(\d{4}-\d{2}-\d{2})",  # "2025-06-30"
            r"(\d{1,2}/\d{1,2}/\d{4})",  # "30/06/2025"
            r"(\d{1,2}\s+\w+\s+\d{4})" # New pattern for "01st August 2025"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL) # Use DOTALL to match newlines with .
            if match:
                raw_date_str = match.group(1).strip()
                # Replace any internal newlines with a single space for parsing
                report_date = re.sub(r'\s+', ' ', raw_date_str) 
                
                # Basic validation: try to parse the date to ensure it's valid
                try:
                    # Attempt to parse common formats
                    if re.match(r'\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}', report_date, re.IGNORECASE):
                        # Remove 'st', 'nd', 'rd', 'th' for parsing
                        report_date = re.sub(r'(st|nd|rd|th)', '', report_date, flags=re.IGNORECASE).strip()
                        datetime.strptime(report_date, '%d %B %Y')
                    elif re.match(r'\w+\s+\d{1,2},?\s+\d{4}', report_date, re.IGNORECASE):
                        datetime.strptime(report_date.replace(',', ''), '%B %d %Y')
                    elif re.match(r'\d{4}-\d{2}-\d{2}', report_date):
                        datetime.strptime(report_date, '%Y-%m-%d')
                    elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', report_date):
                        datetime.strptime(report_date, '%d/%m/%Y') # Assuming DD/MM/YYYY
                    elif re.match(r'\d{1,2}\s+\w+\s+\d{4}', report_date, re.IGNORECASE):
                        # For "01 August 2025" format after removing 'st', 'nd', 'rd', 'th'
                        datetime.strptime(report_date, '%d %B %Y')
                    
                    self.logger.info(f"Detected report date: {report_date}")
                    return report_type, report_date
                except ValueError:
                    self.logger.warning(f"Extracted date '{report_date}' is not a valid date format. Trying next pattern.")
                    report_date = "Unknown" # Reset if parsing fails
                    continue # Try next pattern
        
        self.logger.warning("No valid report date detected.")
        return report_type, report_date
    
    def _extract_special_notices(self, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract errata notices, contingent liabilities, and events after reporting"""
        errata = None
        contingent = None
        events = None
        
        # Errata notice
        errata_match = re.search(
            r"ERRATA NOTICE(.*?)(?=\n\n|\Z|COMPANY STATEMENT|FINANCIAL STATEMENTS|NOTES TO THE FINANCIAL STATEMENTS)", # Added boundary conditions
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if errata_match:
            errata = errata_match.group(1).strip()
        
        # Contingent liabilities
        contingent_match = re.search(
            r"CONTINGENT LIABILITIES(.*?)(?=\n\n|\Z|EVENT OCCURRING|NOTES TO THE FINANCIAL STATEMENTS|Net Assets per Share)", # Added Net Assets per Share as a boundary
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if contingent_match:
            contingent = contingent_match.group(1).strip()
        
        # Events after reporting
        events_match = re.search(
            r"EVENT(?:S)? (?:OCCURRING )?AFTER (?:THE )?REPORTING DATE(.*?)(?=\n\n|\Z|SIGNATURES|APPENDIX|NOTES TO THE FINANCIAL STATEMENTS)", # Added boundary conditions
            text, 
            re.DOTALL | re.IGNORECASE
        )
        if events_match:
            events = events_match.group(1).strip()
        
        return errata, contingent, events

    def extract_financial_data(self, pdf_text: str, company_name: str, company_symbol: str, report_filename: str) -> ExtractedFinancialData:
        """
        Main method to orchestrate the extraction of all financial data.
        """
        extracted_data = ExtractedFinancialData(
            company_name=company_name,
            company_symbol=company_symbol,
            report_filename=report_filename,
            document_hash="placeholder_hash", # This should be calculated externally or passed in
            extraction_confidence=0.0 # Placeholder
        )

        # --- IMPORTANT: Call parse currency and scale first ---
        self._parse_currency_unit_and_scale(pdf_text)
        # --- ADD THESE TWO LINES HERE ---
        extracted_data.currency_unit = self.currency_unit
        extracted_data.scale_factor = self.scale_factor
        # --- End of important call and additions ---

        # 1. Extract Report Type and Date (Keep your existing logic for this)
        # This regex needs to be more robust for the report date from the first page
        # "FOR THE PERIOD ENDED 30TH JUNE 2025"
        report_date_match = re.search(r"FOR THE PERIOD ENDED (\d{1,2}(?:ST|ND|RD|TH)?\s+\w+\s+\d{4})", pdf_text, re.IGNORECASE)
        if report_date_match:
            try:
                # Remove ordinal suffixes (ST, ND, RD, TH) before parsing
                date_str = re.sub(r'(\d+)(ST|ND|RD|TH)', r'\1', report_date_match.group(1), flags=re.IGNORECASE)
                extracted_data.report_date = datetime.strptime(date_str, "%d %B %Y")
                extracted_data.report_type = "Interim" # Assuming based on "PERIOD ENDED"
            except ValueError:
                self.logger.warning(f"Could not parse report date: {report_date_match.group(1)}")
        else:
            self.logger.warning("Report date not found with initial pattern. Trying fallback.")
            # Fallback for date, e.g., from "As at 30 June 2025"
            as_at_date_match = re.search(r"As at (\d{1,2}\s+\w+\s+\d{4})", pdf_text, re.IGNORECASE)
            if as_at_date_match:
                try:
                    date_str = re.sub(r'(\d+)(ST|ND|RD|TH)', r'\1', as_at_date_match.group(1), flags=re.IGNORECASE)
                    extracted_data.report_date = datetime.strptime(date_str, "%d %B %Y")
                    extracted_data.report_type = "Interim" # Or determine based on context
                except ValueError:
                    self.logger.warning(f"Could not parse fallback report date: {as_at_date_match.group(1)}")


        # 2. Extract Financial Metrics (This will now use the improved _extract_financial_metrics)
        extracted_data.financial_metrics = self._extract_financial_metrics(pdf_text)

        # 3. Extract Shareholders
        extracted_data.shareholders = self._extract_shareholders(pdf_text)

        # 4. Extract Directors
        extracted_data.directors = self._extract_directors(pdf_text)

        # 5. Extract Contingent Liabilities
        extracted_data.contingent_liabilities = self._extract_contingent_liabilities(pdf_text)

        # Set a dummy confidence for now
        extracted_data.extraction_confidence = 0.85 # Example confidence

        return extracted_data
    
    def _calculate_confidence(self, data: ExtractedFinancialData) -> float:
        """Calculate extraction confidence score"""
        score = 0.0
        # Base points for company info, report info, shareholders, directors, audit status, notices
        total_possible = 9.0 
        
        # Company info (2 points)
        if data.company_name != "Unknown":
            score += 1.0
        if data.company_symbol != "Unknown":
            score += 1.0
        
        # Report info (2 points - Type, Date)
        if data.report_type != "Unknown":
            score += 1.0
        if data.report_date != "Unknown":
            score += 1.0
        
        # Additional data (3 points - Shareholders, Directors, Audit Status, Notices)
        if data.shareholders:
            score += 1.0
        if data.directors: 
            score += 1.0
        if data.audit_status != "Unknown":
            score += 1.0
        if data.errata_notice or data.contingent_liabilities or data.events_after_reporting: # At least one notice
            score += 1.0
        
        # Financial metrics (add points for each extracted metric)
        metrics = data.financial_metrics
        # Existing metrics
        if metrics.revenue is not None: score += 1.0
        if metrics.profit_for_period is not None: score += 1.0
        if metrics.operating_profit is not None: score += 1.0
        if metrics.basic_eps is not None: score += 1.0
        if metrics.total_assets is not None: score += 1.0
        if metrics.total_liabilities is not None: score += 1.0
        if metrics.total_equity is not None: score += 1.0
        if metrics.gross_profit is not None: score += 1.0
        if metrics.ebitda is not None: score += 1.0
        if metrics.net_cash_flow is not None: score += 1.0
        
        # New metrics
        if metrics.profit_before_tax is not None: score += 1.0
        if metrics.income_tax_expense is not None: score += 1.0
        if metrics.other_income is not None: score += 1.0
        if metrics.administrative_expenses is not None: score += 1.0
        if metrics.distribution_expenses is not None: score += 1.0
        if metrics.total_comprehensive_income is not None: score += 1.0
        if metrics.cash_generated_from_operations is not None: score += 1.0
        if metrics.property_plant_equipment_assets is not None: score += 1.0
        if metrics.intangible_assets_balance is not None: score += 1.0
        if metrics.inventories_balance is not None: score += 1.0
        if metrics.trade_and_other_receivables_balance is not None: score += 1.0
        if metrics.cash_and_cash_equivalents_balance is not None: score += 1.0
        if metrics.trade_payables_balance is not None: score += 1.0
        if metrics.income_tax_payable_balance is not None: score += 1.0
        if metrics.net_assets_per_share is not None: score += 1.0

        # Update total possible points to reflect all potential metrics
        total_possible += 10 # Existing financial metrics
        total_possible += 15 # New financial metrics
        
        return score / total_possible
    
    def record_feedback(self, feedback: ExtractionFeedback):
        """
        Records user feedback using the internal LearningDatabase instance.
        This method is called by the API to store corrections.
        """
        if self.db:
            self.db.store_feedback(feedback)
        else:
            self.logger.warning("LearningDatabase not initialized. Feedback cannot be recorded.")

    def to_legacy_format(self, data: ExtractedFinancialData) -> Dict[str, Any]:
        """Convert to the original dictionary format for backward compatibility"""
        return {
            "company_name": data.company_name, 
            "company_symbol": data.company_symbol, 
            "report_type": data.report_type,
            "report_date": data.report_date,
            "currency_unit": data.currency_unit,
            "scale_factor": data.scale_factor, # Include scale_factor in legacy format
            "financial_metrics": { 
                "revenue": data.financial_metrics.revenue,
                "profit_for_period": data.financial_metrics.profit_for_period,
                "operating_profit": data.financial_metrics.operating_profit,
                "basic_eps": data.financial_metrics.basic_eps,
                "total_assets": data.financial_metrics.total_assets,
                "total_liabilities": data.financial_metrics.total_liabilities,
                "total_equity": data.financial_metrics.total_equity,
                "gross_profit": data.financial_metrics.gross_profit,
                "ebitda": data.financial_metrics.ebitda,
                "net_cash_flow": data.financial_metrics.net_cash_flow,
                # --- New fields in legacy format ---
                "profit_before_tax": data.financial_metrics.profit_before_tax,
                "income_tax_expense": data.financial_metrics.income_tax_expense,
                "other_income": data.financial_metrics.other_income,
                "administrative_expenses": data.financial_metrics.administrative_expenses,
                "distribution_expenses": data.financial_metrics.distribution_expenses,
                "total_comprehensive_income": data.financial_metrics.total_comprehensive_income,
                "cash_generated_from_operations": data.financial_metrics.cash_generated_from_operations,
                "property_plant_equipment_assets": data.financial_metrics.property_plant_equipment_assets,
                "intangible_assets_balance": data.financial_metrics.intangible_assets_balance,
                "inventories_balance": data.financial_metrics.inventories_balance,
                "trade_and_other_receivables_balance": data.financial_metrics.trade_and_other_receivables_balance,
                "cash_and_cash_equivalents_balance": data.financial_metrics.cash_and_cash_equivalents_balance,
                "trade_payables_balance": data.financial_metrics.trade_payables_balance,
                "income_tax_payable_balance": data.financial_metrics.income_tax_payable_balance,
                "net_assets_per_share": data.financial_metrics.net_assets_per_share
                # --- End of new fields in legacy format ---
            },
            "shareholders": [
                asdict(sh) for sh in data.shareholders 
            ],
            "directors": [ 
                asdict(d) for d in data.directors
            ],
            "audit_status": data.audit_status, 
            "errata_notice": data.errata_notice, 
            "contingent_liabilities": data.contingent_liabilities, 
            "events_after_reporting": data.events_after_reporting, 
            "document_hash": data.document_hash, 
            "extraction_confidence": data.extraction_confidence,
            "extraction_method": data.extraction_method,
            "processing_time": data.processing_time
        }

    # Convenience function for backward compatibility (if needed for direct calls)
    def extract_financial_data_with_custom_ml(
        pdf_text: str, 
        company_name: str, 
        company_symbol: str, 
        report_filename: str,
        document_hash: Optional[str] = None,
        ml_model_dir: Optional[str] = None, # Added ml_model_dir
        db_path: Optional[str] = None # Added db_path
    ) -> Dict[str, Any]:
        """
        Legacy function wrapper for backward compatibility.
        Initializes FinancialDataExtractor and calls its method.
        """
        extractor = FinancialDataExtractor(ml_model_dir=ml_model_dir, db_path=db_path) # Pass arguments
        result = extractor.extract_financial_data(pdf_text, company_name, company_symbol, report_filename, document_hash) 
        return extractor.to_legacy_format(result)


# Example Usage and Testing
if __name__ == "__main__":
    # Ensure a dummy database file exists for testing
    dummy_db_path = "extraction_learning.db"
    if Path(dummy_db_path).exists():
        Path(dummy_db_path).unlink() # Delete existing for a clean test

    # Initialize LearningDatabase to ensure schema is created
    db_test_instance = LearningDatabase(dummy_db_path)
    
    sample_pdf_text = """

    PickMe

    07 February 2025

    ERRATA NOTICE

    Reference is made to the interim financial statements for Digital Mobility Solutions Lanka PLC/PickMe

    (the 'Company') for the three months ended 31st December 2024, filed with the Colombo Stock

    Exchange (CSE), and published on the Company's website on 28th January 2025.

    Page 4 (Formatting error) Operating Expenses, Administrative Expenses, and Selling and

    Distribution Expenses were inadvertently presented in LKR instead of LKR thousands.

    • Page 9 (Typological error) The 5th largest shareholder was incorrectly named "Ivenco

    Capital Private Limited". The correct name is "Invenco Capital Private Limited".



    COMPANY STATEMENT OF COMPREHENSIVE INCOME

    Quarter ended 31 December

                                                2024          2023

    Revenue/ Turnover                         1,524,047     1,083,276

    Other Income and Gains                      13,368        74,140

    Operating Expenses                        (464,931)     (384,247)

    Administrative Expenses                   (542,868)     (422,396)

    Selling and Distribution Expenses         (100,012)      (76,822)

    Operating Profit/Loss                       429,603       273,951

    Finance Cost                               (57,800)      (14,145)

    Finance Income                              28,816        26,580

    Net Finance Income                         (28,984)       12,436

    Profit before Tax                          400,619       286,387

    Income Tax (Expenses)/Reversal           (124,679)      (74,784)

    Profit for the period                      275,941       211,603

    Other Comprehensive income                       -             -

    Total Comprehensive Income/(Expense)       275,941       211,603

    Basic/Diluted Earning Per Share (EPS)         0.83          2.34



    Note: All values are in Rs. 1000s, unless otherwise stated

    Figures in brackets indicate deductions.

    The above figures are not audited



    --- PAGE BREAK ---



    Board of Directors

    Jiffry Zulfer - Director/CEO

    A.D. Gunewarderie - Executive Director

    R.HL Gunewardene, Independent Non-Executive Director

    K.NJ. Balendra, Chairman

    M.S. Riyaz (Company Secretary)

    No of Shares - This is a test entry.



    --- PAGE BREAK ---



    5.6. Twenty Largest Shareholders

    Twenty largest shareholders of the Company are as given below:

    As at 31.12.2024

    Number of Shares    %

    1   J.Z. Hassen                 119,300,000    35.79

    2   A.D. Gunewarderie            31,465,717     9.44

    3   LOLC Technology Services Limited 31,110,782     9.33

    4   International Finance Corporation 16,307,356     4.89

    5   Invenco Capital Private Limited  11,955,376     3.59

    6   K.NJ. Balendra                9,619,323     2.89

    7   R.HL Gunewardene              8,528,351     2.56

    8   Interblocks Holdings Pte Ltd  8,477,935     2.54

    9   H Capital (Private) Limited   7,225,159     2.17

    10  M.S. Riyaz                    6,600,000     1.98



    CONTINGENT LIABILITIES

    There has not been a significant change to the disclosure made on contingent liabilities in the Financial Statements for the year

    ended 31st March 2024



    EVENT OCCURRING AFTER THE REPORTING DATE

    No events have occurred since the statement of financial position date which would require adjustments to, or disclosure in the

    financial statements.

    """
    print("\n--- Enhanced Financial Data Extraction Test ---")
    
    # Test with new class-based approach
    extractor = FinancialDataExtractor(db_path=dummy_db_path) # Pass db_path for testing
    # Pass a dummy document_hash for testing purposes
    test_document_hash = "test_document_hash_123"
    result = extractor.extract_financial_data(sample_pdf_text, document_hash=test_document_hash)
    
    print(f"Company: {result.company_name} ({result.company_symbol})")
    print(f"Report Type: {result.report_type}, Date: {result.report_date}")
    print(f"Currency Unit: {result.currency_unit}, Scale Factor: {result.scale_factor}")
    print(f"Confidence Score: {result.extraction_confidence:.2f}")
    print(f"Revenue: {result.financial_metrics.revenue}")
    print(f"Profit: {result.financial_metrics.profit_for_period}")
    print(f"Operating Profit: {result.financial_metrics.operating_profit}")
    print(f"EPS: {result.financial_metrics.basic_eps}")
    print(f"Total Assets: {result.financial_metrics.total_assets}")
    print(f"Total Liabilities: {result.financial_metrics.total_liabilities}")
    print(f"Total Equity: {result.financial_metrics.total_equity}")
    print(f"Gross Profit: {result.financial_metrics.gross_profit}")
    print(f"EBITDA: {result.financial_metrics.ebitda}")
    print(f"Net Cash Flow: {result.financial_metrics.net_cash_flow}")
    print(f"Top Shareholders: {len(result.shareholders)}")
    for sh in result.shareholders:
        print(f"  - {sh.rank}. {sh.name}: {sh.shares} shares ({sh.percentage}%)")
    print(f"Directors Found: {len(result.directors)}")
    for d in result.directors:
        print(f"  - {d.name} (Role: {d.role or 'N/A'})")
    print(f"Audit Status: {result.audit_status}")
    print(f"Errata Notice: {result.errata_notice}")
    print(f"Contingent Liabilities: {result.contingent_liabilities}")
    print(f"Events After Reporting: {result.events_after_reporting}")
    print(f"Document Hash: {result.document_hash}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    
    # Test legacy format conversion
    legacy_format = extractor.to_legacy_format(result)
    print("\n--- Legacy Format Output ---")
    print(json.dumps(legacy_format, indent=2, ensure_ascii=False))

    # Simulate some feedback for testing the database
    feedback_item_test = ExtractionFeedback(
        document_hash=test_document_hash,
        field_name="revenue",
        extracted_value=result.financial_metrics.revenue,
        correct_value=result.financial_metrics.revenue * 1.05, # Simulate a correction
        confidence_score=0.9,
        user_id="test_user"
    )
    extractor.record_feedback(feedback_item_test)
    print(f"\nRecorded test feedback for revenue. Corrected value: {feedback_item_test.correct_value}")

    # Verify feedback retrieval
    retrieved_feedback = db_test_instance.get_feedback_for_training("revenue")
    print(f"Retrieved {len(retrieved_feedback)} feedback entries for 'revenue' from DB.")
    if retrieved_feedback:
        print(f"  Latest feedback: {retrieved_feedback[0]['correct_value']}")

    # Clean up dummy database file
    # Path(dummy_db_path).unlink()
    # print(f"\nCleaned up dummy database: {dummy_db_path}")
