import re
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Assuming ml_models_manager.py is in the same directory or accessible
from ml_models_manager import FinancialMLModelManager

# Configure logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Data Models (Data Classes) ---

@dataclass
class FinancialMetrics:
    """Represents various financial metric values."""
    revenue: Optional[float] = None
    profit_for_period: Optional[float] = None
    operating_profit: Optional[float] = None
    basic_eps: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    gross_profit: Optional[float] = None
    ebitda: Optional[float] = None
    net_cash_flow: Optional[float] = None

@dataclass
class Shareholder:
    """Represents a single shareholder entry."""
    name: Optional[str] = None
    shares: Optional[int] = None
    percentage: Optional[float] = None

@dataclass
class Director:
    """Represents a single director entry."""
    name: Optional[str] = None
    role: Optional[str] = None

@dataclass
class ExtractedFinancialData:
    """Comprehensive data structure for extracted financial information."""
    company_name: Optional[str] = None
    company_symbol: Optional[str] = None
    report_type: Optional[str] = None # e.g., "Annual", "Interim", "Quarterly"
    report_date: Optional[str] = None # YYYY-MM-DD format
    document_hash: str = "" # Unique identifier for the document
    
    # FIX: Use default_factory for mutable default arguments
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    shareholders: List[Shareholder] = field(default_factory=list) # Also good practice for lists
    directors: List[Director] = field(default_factory=list) # Also good practice for lists
    
    # Metadata about the extraction process
    extraction_confidence: Optional[float] = None # Overall confidence for the document
    extraction_method: Optional[str] = "hybrid" # "regex", "ml", "hybrid"
    processing_time: Optional[float] = None # Time taken to process this document
    full_text_context: Optional[str] = None # Stores the full text used for extraction


@dataclass
class ExtractionFeedback:
    """Data structure for user feedback on extracted values."""
    document_hash: str
    field_name: str
    extracted_value: Any
    correct_value: Any
    confidence_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now) # Use default_factory for mutable datetime
    user_id: Optional[str] = "system" # User who provided feedback, or 'system' for automated

# --- 2. Learning Database ---

class LearningDatabase:
    """Manages SQLite database for storing extracted data and feedback."""

    def __init__(self, db_path: str = "extraction_learning.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initializes the SQLite database tables."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Drop tables if they exist to ensure schema is always up-to-date
            # This is useful during development to avoid schema mismatch errors
            cursor.execute("DROP TABLE IF EXISTS feedback")
            cursor.execute("DROP TABLE IF EXISTS extractions")
            logger.info("Dropped existing 'feedback' and 'extractions' tables (if they existed).")


            # Create extractions table (stores parsed data for documents)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    document_hash TEXT PRIMARY KEY,
                    company_name TEXT,
                    report_date TEXT, -- Ensure this column exists
                    extraction_data TEXT, -- Stores JSON string of ExtractedFinancialData
                    confidence_score REAL,
                    extraction_method TEXT,
                    processing_time REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create feedback table (stores user corrections/feedback)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT,
                    field_name TEXT,
                    extracted_value TEXT,
                    correct_value TEXT,
                    confidence_score REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    FOREIGN KEY (document_hash) REFERENCES extractions (document_hash)
                )
            """)
            conn.commit()
            logger.info(f"Database initialized/checked at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
        finally:
            if conn:
                conn.close()

    def get_connection(self):
        """Returns a database connection."""
        return sqlite3.connect(self.db_path)

    def store_extraction_data(self, data: ExtractedFinancialData):
        """Stores extracted financial data for a document."""
        try:
            with self.get_connection() as conn:
                # Rely on dataclasses.asdict for recursive conversion of dataclasses to dicts
                # asdict handles nested dataclasses and lists/dicts of dataclasses automatically.
                data_dict = asdict(data)
                
                # json.dumps with default=str handles datetime objects and other non-standard types
                extraction_data_json = json.dumps(data_dict, default=str, ensure_ascii=False)

                conn.execute("""
                    INSERT OR REPLACE INTO extractions 
                    (document_hash, company_name, report_date, extraction_data, confidence_score, extraction_method, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.document_hash,
                    data.company_name,
                    data.report_date,
                    extraction_data_json,
                    data.extraction_confidence,
                    data.extraction_method,
                    data.processing_time
                ))
                conn.commit()
                logger.info(f"Stored extraction data for document: {data.document_hash}")
        except sqlite3.Error as e:
            logger.error(f"Error storing extraction data for {data.document_hash}: {e}")
        except Exception as e: # Catch general exceptions to get more info
            logger.error(f"Unhandled error in store_extraction_data for {data.document_hash}: {e}", exc_info=True)
            raise # Re-raise to show original traceback


    def store_feedback(self, feedback: ExtractionFeedback):
        """Stores user feedback for a specific field."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback 
                    (document_hash, field_name, extracted_value, correct_value, confidence_score, timestamp, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.document_hash,
                    feedback.field_name,
                    str(feedback.extracted_value), # Store as string for flexibility
                    str(feedback.correct_value),   # Store as string
                    feedback.confidence_score,
                    feedback.timestamp.isoformat(),
                    feedback.user_id
                ))
                conn.commit()
                logger.info(f"Stored feedback for {feedback.field_name} (Doc: {feedback.document_hash})")
        except sqlite3.Error as e:
            logger.error(f"Error storing feedback for {feedback.field_name} (Doc: {feedback.document_hash}): {e}")

    def get_feedback_for_training(self, field_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves feedback data for a specific field, suitable for ML training.
        Returns a list of dictionaries.
        """
        feedback_list = []
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT f.document_hash, f.field_name, f.extracted_value, f.correct_value, f.confidence_score, f.timestamp, f.user_id, e.extraction_data
                    FROM feedback f
                    JOIN extractions e ON f.document_hash = e.document_hash
                    WHERE f.field_name = ? AND f.correct_value IS NOT NULL AND f.correct_value != ''
                    ORDER BY f.timestamp DESC
                """, (field_name,))
                
                rows = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                
                for row in rows:
                    feedback_dict = dict(zip(column_names, row))
                    # Attempt to parse extraction_data to get full_text_context
                    try:
                        extraction_data = json.loads(feedback_dict['extraction_data'])
                        feedback_dict['full_text_context'] = extraction_data.get('full_text_context')
                    except (json.JSONDecodeError, KeyError):
                        feedback_dict['full_text_context'] = None # Or handle as needed
                    del feedback_dict['extraction_data'] # Remove raw JSON string
                    feedback_list.append(feedback_dict)
        except sqlite3.Error as e:
            logger.error(f"Error retrieving feedback for training for field {field_name}: {e}")
        return feedback_list

# --- 3. Financial Parser ---

class FinancialParser:
    """
    Parses financial documents, combining regex and ML models for extraction.
    Also manages a self-learning database for continuous improvement.
    """
    
    def __init__(self, ml_model_dir: str = "ml_models", db_path: str = "extraction_learning.db"):
        self.ml_manager = FinancialMLModelManager(model_dir=ml_model_dir)
        self.db = LearningDatabase(db_path=db_path)
        logger.info(f"FinancialParser initialized with ML models from '{ml_model_dir}' and DB '{db_path}'.")
        
        # Add a debug check to confirm method presence
        if hasattr(self, '_extract_using_ml'):
            logger.debug("'_extract_using_ml' method is present in FinancialParser.")
        else:
            logger.error("'_extract_using_ml' method is MISSING from FinancialParser!")


        # Define default confidence thresholds for ML
        self.ml_confidence_thresholds = {
            'numeric': 0.7,    # High confidence needed for numeric
            'categorical': 0.8 # Even higher for categorical to avoid misclassification
        }

    def _clean_and_convert_value(self, field_name: str, value: Any, target_type: str) -> Any:
        """
        Cleans and converts extracted string values to their target type (float, int, str).
        This is a shared utility for both regex and ML outputs.
        """
        if value is None:
            return None
        
        # Ensure value is string for cleaning
        str_value = str(value).strip()

        if target_type == 'numeric':
            if not str_value:
                return None
            
            # Handle common financial number formats
            cleaned = str_value.replace(',', '').replace(' ', '')
            
            # Handle parentheses for negative numbers (e.g., (123) -> -123)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            # Remove currency symbols or other non-numeric characters before final conversion
            cleaned = re.sub(r'[^\d\.\-]', '', cleaned)
            
            try:
                # Try converting to float first, then int if no decimal
                f_val = float(cleaned)
                # Check if it's an integer value
                if f_val.is_integer():
                    return int(f_val)
                return f_val
            except ValueError:
                return None # Conversion failed
        elif target_type == 'categorical':
            # For categorical, just return string, maybe normalize case
            return str_value.lower()
        else: # Default to string
            return str_value

    def _extract_using_regex(self, text: str, field_name: str, target_type: str = 'string') -> Tuple[Optional[Any], float]:
        """
        Extracts a field using predefined regex patterns.
        Returns the extracted value and a confidence score (e.g., 0.9 for regex match).
        """
        patterns = {
            'company_name': [
                r'(?i)(?:Company|Ltd|PLC|Inc|Corp|Corporation|Group|Holdings|Solutions|Investments)\s*([A-Z][a-zA-Z\s,&\.-]+(?:Ltd|PLC|Inc|Corp|Corporation|Group|Holdings|Solutions|Investments)\.?)\b',
                r'(?i)(?:For|Of)\s+([A-Z][a-zA-Z\s,&\.-]+(?:Ltd|PLC|Inc|Corp|Corporation|Group|Holdings|Solutions|Investments)\.?)\b',
                r'Digital Mobility Solutions Lanka PLC' # Specific example
            ],
            'company_symbol': [
                r'\b([A-Z]{2,5})\s+(?:Limited|PLC|Inc|Corp|Corporation)\b', # E.g., ABC Limited -> ABC
                r'\b([A-Z]{3,5})(?=\s+(?:Shares|Stock|CSE|Trading|Equity))\b', # E.g., XYZ Shares -> XYZ
                r'\b[A-Z]{3,5}\b' # General 3-5 letter uppercase word, might need context
            ],
            'report_type': [
                r'(?i)\b(Annual|Interim|Quarterly|Half-Yearly|Semi-Annual)\s+Financial\s+Statements?\b',
                r'(?i)\b(Annual|Interim|Quarterly|Half-Yearly|Semi-Annual)\s+Report\b',
                r'(?i)Full\s+Year\s+Results?\b'
            ],
            'report_date': [
                r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', # Added (?:st|nd|rd|th)?
                r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'(?:for\s+the\s+(?:year|period)\s+ended)\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})' # Added (?:st|nd|rd|th)?
            ],
            'revenue': [
                r'(?i)(?:Revenue|Sales|Turnover|Income from operations)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?', # Made LKR optional
                r'(?i)(?:Revenue|Sales|Turnover|Income from operations)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?', # Made LKR optional
                r'(?i)(?:Revenue|Sales|Turnover|Income from operations)\s*(?:of|is|was|amounted to|:)?\s*\$?\s*([0-9,.\(\)\-]+)(?:\s+million|\s+billion|\s+thousand)?'
            ],
            'profit_for_period': [
                r'(?i)(?:Profit for the period|Net Income|Profit after Tax|Earnings after Tax|Net Profit)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?', # Made LKR optional
                r'(?i)(?:Profit for the period|Net Income|Profit after Tax|Earnings after Tax|Net Profit)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?', # Made LKR optional
                r'(?i)(?:Profit for the period|Net Income|Profit after Tax|Earnings after Tax|Net Profit)\s*(?:of|is|was|amounted to|:)?\s*\$?\s*([0-9,.\(\)\-]+)(?:\s+million|\s+billion|\s+thousand)?'
            ],
            'operating_profit': [
                r'(?i)(?:Operating Profit|Operating Profit/\(Loss\)|EBIT|Operational Earnings)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?', # Made LKR optional, added /(Loss)
                r'(?i)(?:Operating Profit|Operating Profit/\(Loss\)|EBIT|Operational Earnings)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?', # Made LKR optional, added /(Loss)
            ],
            'basic_eps': [
                r'(?i)(?:Basic EPS|Diluted Earning Per Share|Earnings Per Share)\s*(?:of|is|was|:)?\s*([0-9.]+)' # Added Diluted Earning Per Share
            ],
            'total_assets': [
                r'(?i)(?:Total Assets)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)(?:Total Assets)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'total_liabilities': [
                r'(?i)(?:Total Liabilities)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)(?:Total Liabilities)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'total_equity': [
                r'(?i)(?:Total Equity|Shareholders\' Equity)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)(?:Total Equity|Shareholders\' Equity)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'gross_profit': [
                r'(?i)(?:Gross Profit|Gross Margin)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)(?:Gross Profit|Gross Margin)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'ebitda': [
                r'(?i)\bEBITDA\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)\bEBITDA\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'net_cash_flow': [
                r'(?i)(?:Net Cash Flow|Cash Flow from Operating Activities)\s*(?:of|is|was|amounted to|:)?\s*(?:LKR\s*)?([0-9,.\(\)\-]+)(?:\s+thousands?)?',
                r'(?i)(?:Net Cash Flow|Cash Flow from Operating Activities)\s*(?:of|is|was|amounted to|:)?\s*([0-9,.\(\)\-]+)\s*(?:LKR)?(?:\s+thousands?)?',
            ],
            'audit_status': [
                r'(?i)\b(audited|unaudited)\s+financial\s+statements?\b'
            ]
        }

        for pattern in patterns.get(field_name, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Apply specific post-processing for certain fields
                if field_name == 'report_date':
                    try: # Try to standardize date format to YYYY-MM-DD
                        # Handle "31st December 2024"
                        if re.search(r'\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', value, re.IGNORECASE):
                            # Remove the suffix before parsing
                            value = re.sub(r'(st|nd|rd|th)\b', '', value, flags=re.IGNORECASE)
                            dt = datetime.strptime(value.strip(), '%d %B %Y')
                        elif '/' in value:
                            dt = datetime.strptime(value, '%d/%m/%Y')
                        elif '-' in value:
                            dt = datetime.strptime(value, '%Y-%m-%d')
                        else:
                            dt = None
                        if dt:
                            value = dt.strftime('%Y-%m-%d')
                        else:
                            logger.warning(f"Could not standardize date format for '{value}' in '{field_name}'.")
                    except ValueError:
                        logger.warning(f"Failed to parse date '{value}' for '{field_name}' with known formats.")

                # Clean and convert the extracted value
                cleaned_value = self._clean_and_convert_value(field_name, value, target_type)
                if cleaned_value is not None:
                    return cleaned_value, 0.95 # High confidence for regex match
        
        return None, 0.0 # No match

    def _extract_using_ml(self, text: str, field_name: str) -> Tuple[Optional[Any], float]:
        """Calls the ML model manager for prediction."""
        try:
            prediction, confidence = self.ml_manager.predict(field_name, text)
            return prediction, confidence
        except Exception as e:
            logger.error(f"Error during ML prediction for {field_name}: {e}")
            return None, 0.0 # Return None and 0 confidence on error

    def _hybrid_extract_field(self, document_text: str, field_name: str) -> Tuple[Optional[Any], Optional[float], str]:
        """
        Attempts to extract a field using ML, falls back to regex if ML confidence is low.
        Returns extracted value, confidence, and method used ('ml', 'regex', 'none').
        """
        # Get field type from ML manager's config
        field_type = self.ml_manager.field_configs.get(field_name, {}).get('type', 'string')
        ml_confidence_threshold = self.ml_confidence_thresholds.get(field_type, 0.7) # Default to 0.7

        ml_value, ml_confidence = self._extract_using_ml(document_text, field_name)
        ml_method = 'ml' if ml_value is not None else 'none'

        if ml_value is not None and ml_confidence is not None and ml_confidence >= ml_confidence_threshold:
            logger.debug(f"ML extraction successful for {field_name}: {ml_value} (Confidence: {ml_confidence:.2f})")
            return self._clean_and_convert_value(field_name, ml_value, field_type), ml_confidence, ml_method
        else:
            logger.debug(f"ML extraction for {field_name} (Value: {ml_value}, Confidence: {ml_confidence:.2f}) below threshold or not found. Falling back to regex.")
            regex_value, regex_confidence = self._extract_using_regex(document_text, field_name, field_type)
            regex_method = 'regex' if regex_value is not None else 'none'
            
            if regex_value is not None:
                logger.debug(f"Regex extraction successful for {field_name}: {regex_value} (Confidence: {regex_confidence:.2f})")
                return regex_value, regex_confidence, regex_method
            
            logger.debug(f"No extraction found for {field_name} via ML or Regex.")
            return None, None, 'none'

    def parse_document(self, document_text: str, document_hash: str) -> ExtractedFinancialData:
        """
        Parses the full text of a financial document to extract structured data.
        
        Args:
            document_text: The full text extracted from the PDF.
            document_hash: A unique identifier for the document.
            
        Returns:
            An ExtractedFinancialData object containing all parsed information.
        """
        start_time = datetime.now()
        extracted_data = ExtractedFinancialData(document_hash=document_hash, full_text_context=document_text)
        overall_confidence_sum = 0
        extracted_field_count = 0
        extraction_methods_used = set()

        # Iterate through fields defined in ml_models_manager to ensure consistency
        for field_name, config in self.ml_manager.field_configs.items():
            value, confidence, method = self._hybrid_extract_field(document_text, field_name)
            
            if value is not None:
                extraction_methods_used.add(method)
                overall_confidence_sum += confidence if confidence is not None else 0
                extracted_field_count += 1

                # Assign extracted value to the correct place in ExtractedFinancialData
                # Check if it's a financial metric
                if field_name in asdict(FinancialMetrics()): # Check against a default FinancialMetrics instance
                    setattr(extracted_data.financial_metrics, field_name, value)
                elif hasattr(extracted_data, field_name):
                    setattr(extracted_data, field_name, value)
                else:
                    logger.warning(f"Extracted field '{field_name}' not found in ExtractedFinancialData structure. Value: {value}")

        # Special handling for shareholders and directors (currently regex-only in this parser)
        # These would typically be handled by more advanced ML models (e.g., NER, table extraction)
        # but for now, they use dedicated regex functions if not covered by ml_manager.field_configs
        extracted_data.shareholders = self._extract_shareholders(document_text)
        extracted_data.directors = self._extract_directors(document_text)


        # Calculate overall confidence
        if extracted_field_count > 0:
            extracted_data.extraction_confidence = overall_confidence_sum / extracted_field_count
        else:
            extracted_data.extraction_confidence = 0.0

        # Determine overall extraction method
        if 'ml' in extraction_methods_used and 'regex' in extraction_methods_used:
            extracted_data.extraction_method = 'hybrid'
        elif 'ml' in extraction_methods_used:
            extracted_data.extraction_method = 'ml'
        elif 'regex' in extraction_methods_used:
            extracted_data.extraction_method = 'regex'
        else:
            extracted_data.extraction_method = 'none'

        end_time = datetime.now()
        extracted_data.processing_time = (end_time - start_time).total_seconds()

        # Store the initial extraction data (before any potential user feedback)
        self.db.store_extraction_data(extracted_data)
        
        return extracted_data

    def record_feedback(self, document_hash: str, field_name: str, extracted_value: Any, correct_value: Any, confidence_score: Optional[float] = None, user_id: str = "manual"):
        """
        Records user feedback for a specific field in a document.
        This data will be used for future retraining.
        """
        feedback = ExtractionFeedback(
            document_hash=document_hash,
            field_name=field_name,
            extracted_value=extracted_value,
            correct_value=correct_value,
            confidence_score=confidence_score,
            user_id=user_id
        )
        self.db.store_feedback(feedback)
        logger.info(f"Feedback recorded for {field_name} in {document_hash}")

    # --- Specific Regex-based Extractors (can be replaced by ML models later) ---
    def _extract_shareholders(self, text: str) -> List[Shareholder]:
        """
        Extract shareholder information from text using a more robust regex.
        Assumes the shareholder list is typically in a table-like format.
        """
        shareholders = []
        
        # Broader section capture for "Twenty Largest Shareholders"
        # Look for the header and capture until another major section or end of document
        shareholder_section_match = re.search(
            r"(?:Twenty Largest Shareholders|Shareholders' List|Major Shareholders)(.*?)(?=\n\n\n|\Z|FINANCIAL STATEMENTS|DIRECTORS|AUDITORS|NOTES TO THE FINANCIAL STATEMENTS)", 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if shareholder_section_match:
            section_text = shareholder_section_match.group(1) # Capture the content after the header
            logger.debug(f"Shareholder section found. Attempting to parse lines.")
            
            # Pattern to capture rank, name, shares, percentage
            # This pattern is more flexible for names and numbers, and expects a percentage at the end.
            # It tries to avoid capturing lines that are clearly not shareholder entries.
            # Rank (optional, but usually present for largest shareholders), Name, Shares, Percentage
            # Rank: (\d+)\s*
            # Name: ([A-Z][a-zA-Z\s\.,&\-'\(\)]+?) - flexible for names, special chars, non-greedy
            # Shares: ([\d,]+\.?\d*) - allows for thousands separators and optional decimals
            # Percentage: ([\d\.]+) - allows for decimals
            
            # Refined pattern:
            # 1. Start of line or whitespace before rank
            # 2. Optional rank number
            # 3. Name: Capture anything that looks like a name, avoiding numbers/percentages too early
            # 4. Shares: A number with commas/dots, possibly in thousands/millions context
            # 5. Percentage: A number with a dot, followed by optional %
            
            # This pattern is designed to pick up lines like:
            # "7\",\"R.HL Gunewardene\",\"8.528.351\",\"2.56\""
            # "10\",\"M.S. Riyaz\",\"6,600,000\",\"1.98\""
            
            # Let's try to be more specific for the typical CSV-like or spaced format
            # Pattern for lines that look like: RANK, NAME, SHARES, PERCENTAGE
            # It tries to be robust to missing quotes or extra spaces.
            shareholder_line_pattern = re.compile(
                r'^\s*(\d+)\s*[,"]?\s*([^,"]+?)\s*[,"]?\s*([\d,]+\.?\d*)\s*[,"]?\s*([\d\.]+)\s*%', 
                re.MULTILINE | re.IGNORECASE
            )
            # A more generic pattern if the above fails, looking for Name, Shares, Percentage in a line
            # This is less structured but might catch more if OCR is messy.
            # Name: [A-Z][a-zA-Z\s\.,&\-'\(\)]+
            # Shares: \d[\d,]*\d
            # Percentage: \d+\.\d+
            
            # Let's use the provided snippet's format: "7\n\",\"R.HL Gunewardene\n\",\"8.528.351\n\",\"2.56\n\"
            # This is a challenging format due to newlines *within* fields and CSV-like structure.
            # The previous pattern was close, but the `\n` within fields is problematic for `.`
            # Let's process the section line by line and then apply a pattern.

            lines = section_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that are clearly not shareholder data (e.g., headers, notes, other sections)
                if re.search(r'(dividends paid|contingent liabilities|event occurring|note|total|sub-total|summary|auditors)', line, re.IGNORECASE):
                    logger.debug(f"Skipping line (likely non-shareholder): {line}")
                    continue
                if len(line.split()) < 3 and not re.search(r'\d', line): # Skip very short lines without numbers
                    continue

                # Try to match the specific format seen in the user's initial PDF snippet
                # Example: "7\n\",\"R.HL Gunewardene\n\",\"8.528.351\n\",\"2.56\n\"
                # This seems to be a CSV-like line, possibly with newlines inside fields due to OCR.
                match = re.search(
                    r'(\d+)\s*[,"]?\s*([^,"]+?)\s*[,"]?\s*([\d,\.]+)\s*[,"]?\s*([\d\.]+)', 
                    line, 
                    re.IGNORECASE | re.DOTALL # DOTALL to allow . to match newline
                )
                
                if match:
                    try:
                        # rank = int(match.group(1).strip()) # Rank is not stored in Shareholder dataclass
                        name = match.group(2).strip().replace('\n', ' ').replace('\"', '') # Remove quotes and newlines from name
                        shares_str = match.group(3).strip().replace(',', '').replace('\"', '') # Clean shares
                        percentage_str = match.group(4).strip().replace('\"', '') # Clean percentage

                        shares = None
                        if shares_str:
                            try:
                                shares = int(float(shares_str)) # Convert to float first to handle "8.528.351" (OCR error for 8,528,351)
                            except ValueError:
                                logger.warning(f"Could not convert shares '{shares_str}' to int for shareholder '{name}'.")

                        percentage = None
                        if percentage_str:
                            try:
                                percentage = float(percentage_str)
                                if percentage > 1000: # Heuristic: if percentage is too high, it's likely a misread number
                                    logger.warning(f"Unusual percentage value for '{name}': {percentage}. Setting to None.")
                                    percentage = None
                            except ValueError:
                                logger.warning(f"Could not convert percentage '{percentage_str}' to float for shareholder '{name}'.")
                        
                        # Only add if name and shares are reasonably extracted
                        if name and shares is not None and shares > 0:
                            shareholders.append(Shareholder(name=name, shares=shares, percentage=percentage))
                            logger.debug(f"  Found shareholder: {name}, Shares: {shares}, Pct: {percentage}")
                        else:
                            logger.debug(f"  Skipping potential shareholder line due to invalid data: {line}")
                            
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse shareholder entry from line: '{line}', error: {e}")
                        continue
                else:
                    logger.debug(f"  No shareholder match in line: '{line}'")
        else:
            logger.info("Shareholders section header not found.")
        
        return shareholders

    def _extract_directors(self, text: str) -> List[Director]:
        """
        Extracts director names and roles from sections like "Board of Directors" (regex fallback).
        """
        directors = []
        
        director_section_match = re.search(
            r"(Board of Directors|Directors' Report|Directors)(.*?)(?=\n\n\n|\Z|FINANCIAL STATEMENTS|SHAREHOLDERS|AUDITORS|NOTES TO THE FINANCIAL STATEMENTS)", 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if director_section_match:
            section_text = director_section_match.group(2) 
            logger.info(f"Found directors section: {director_section_match.group(1)}")
            
            # Pattern to capture name and optional role
            name_role_pattern = re.compile(
                r"([A-Z][A-Za-z\s\.\-']+(?:[A-Z][A-Za-z\s\.\-']+)?)" # Name (can be multi-part)
                r"(?:\s*[,|\-|\(]?\s*([A-Za-z\s&,.'\/()]+?)\s*[\)]?)?" # Optional role in brackets or after comma
                r"(?=\n|\Z)", # Ends at newline or end of string
                re.MULTILINE
            )
            
            lines = section_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that look like numbers, short fragments, or page/note references
                if re.match(r'^[\d\s\.,\(\)-]+$', line) or len(line.split()) < 2:
                    continue
                if re.search(r'(page|note|table|figure|statement)', line, re.IGNORECASE):
                    continue

                match = name_role_pattern.search(line)
                if match:
                    name = match.group(1).strip()
                    role = match.group(2).strip() if match.group(2) else None
                    
                    # Basic filtering to ensure it's likely a person's name and not just a title
                    if len(name.split()) > 1 and not re.search(r'(chief|head|manager|officer|secretary)', name, re.IGNORECASE):
                        directors.append(Director(name=name, role=role))
                        logger.debug(f"  Found director: {name} (Role: {role})")
                else:
                    logger.debug(f"  No director match in line: '{line}'")
        else:
            logger.info("Directors section header not found.")
        
        return directors
