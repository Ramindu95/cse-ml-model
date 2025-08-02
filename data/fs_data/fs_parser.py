import re
import logging
import json # Added for json.dumps in example usage
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
import time
import random

# Configure logging for the parser
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

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
    net_cash_flow: Optional[float] = None # Added

@dataclass
class Shareholder:
    """Data class for shareholder information"""
    rank: int
    name: str
    shares: int
    percentage: float

@dataclass
class Director:
    """Data class for director information"""
    name: str
    role: Optional[str] = None

@dataclass
class ExtractedFinancialData:
    """Main data structure for extracted financial information"""
    company_name: str
    company_symbol: str
    report_type: str = "Unknown"
    report_date: str = "Unknown"
    currency_unit: str = "LKR thousands"
    scale_factor: float = 1.0 # ADDED THIS LINE to store the scaling factor
    financial_metrics: FinancialMetrics = None
    shareholders: List[Shareholder] = None
    directors: List[Director] = None
    audit_status: str = "Unknown"
    errata_notice: Optional[str] = None
    contingent_liabilities: Optional[str] = None
    events_after_reporting: Optional[str] = None
    extraction_confidence: float = 0.0
    document_hash: Optional[str] = None 
    
    def __post_init__(self):
        if self.financial_metrics is None:
            self.financial_metrics = FinancialMetrics()
        if self.shareholders is None:
            self.shareholders = []
        if self.directors is None:
            self.directors = []

class FinancialDataExtractor:
    """
    Enhanced financial data extractor with better structure and error handling
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the extractor with optional custom ML model
        
        Args:
            model_path: Path to the trained ML model file
        """
        self.logger = logging.getLogger(self.__class__.__name__) # Initialize logger for the class
        self.model = None
        self.model_loaded = False
        self.scale_factor = 1.0 # Default scale factor
        self.currency_unit = "LKR thousands" # Default currency unit
        
        if model_path:
            self._load_custom_model(model_path)
    
    def _load_custom_model(self, model_path: str) -> bool:
        """
        Load custom ML model for financial data extraction
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Placeholder for actual model loading
            # Replace with your actual ML framework (scikit-learn, torch, tensorflow, etc.)
            # Example:
            # import joblib  # for sklearn models
            # self.model = joblib.load(model_path)
            # 
            # import torch  # for PyTorch models
            # self.model = torch.load(model_path)
            # 
            # import tensorflow as tf  # for TensorFlow models
            # self.model = tf.keras.models.load_model(model_path)
            
            self.logger.info(f"Custom ML model would be loaded from: {model_path}")
            self.model_loaded = True
            return True
            
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
            r"PickMe"
        ]
        
        symbol_patterns = [
            r"\b([A-Z]{3,5})(?:\.[NX]\d{4})?\b",  # 3-5 letter stock symbols, e.g., PKME or ABAN.N0000
            r"PKME"  # Specific symbol
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
        Parses the currency unit and scale factor from the text.
        Updates self.currency_unit and self.scale_factor.
        """
        # Look for phrases like "in Rupees '000s", "in millions", "in thousands"
        scale_patterns = [
            r"values are in Rupees '000s", # Specific to your example
            r"values are in Rs\. '000s",
            r"in (?:Rs\.|LKR)\s*('?000s|thousands)",
            r"in (?:Rs\.|LKR)\s*(millions)",
            r"in (?:Rs\.|LKR)\s*(billions)",
            r"in (?:USD|US\$)\s*(thousands)",
            r"in (?:USD|US\$)\s*(millions)",
            r"in (?:USD|US\$)\s*(billions)"
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                unit_phrase = match.group(0).lower()
                if "'000s" in unit_phrase or "thousands" in unit_phrase:
                    self.scale_factor = 1000.0
                    self.currency_unit = re.search(r"(rupees|rs\.|lkr|usd|us\$)", unit_phrase).group(0).upper() + " thousands"
                elif "millions" in unit_phrase:
                    self.scale_factor = 1_000_000.0
                    self.currency_unit = re.search(r"(rupees|rs\.|lkr|usd|us\$)", unit_phrase).group(0).upper() + " millions"
                elif "billions" in unit_phrase:
                    self.scale_factor = 1_000_000_000.0
                    self.currency_unit = re.search(r"(rupees|rs\.|lkr|usd|us\$)", unit_phrase).group(0).upper() + " billions"
                
                self.logger.info(f"Detected currency unit: {self.currency_unit}, scale factor: {self.scale_factor}")
                return # Found and set, exit
        
        self.logger.info("No specific currency unit or scale factor found. Defaulting to LKR thousands and scale 1.0.")
        self.scale_factor = 1.0 # Default if no specific unit found (assuming base unit)
        self.currency_unit = "LKR" # Default currency unit if not specified

    def _extract_financial_metrics(self, text: str) -> FinancialMetrics:
        """Extract financial metrics using improved regex patterns"""
        metrics = FinancialMetrics()
        
        # Enhanced patterns for financial data extraction
        patterns = {
            'revenue': [
                r"Revenue(?: from contracts with customers)?\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Total Revenue\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Turnover\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Net Revenue\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'profit_for_period': [
                r"Profit for the period\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Net Income\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Profit after Tax\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'operating_profit': [
                r"Operating Profit(?:/Loss)?\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Results from operating activities\s*[^\d\n]*?([\d,\.\(\)\-\s]+)", 
                r"EBIT\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'basic_eps': [
                r"Basic(?:/Diluted)?\s+Earning Per Share\s*\(EPS\)\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"EPS\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Basic\s*([\d\.]+)" # Added for simpler EPS lines like "Basic 0.73"
            ],
            'total_assets': [
                r"Total Assets\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'total_liabilities': [
                r"Total Liabilities\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Total liabilities\s*[^\d\n]*?([\d,\.\(\)\-\s]+)" 
            ],
            'total_equity': [
                r"Total Equity\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'gross_profit': [
                r"Gross Profit\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'ebitda': [
                r"EBITDA\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ],
            'net_cash_flow': [
                r"Net cash flow from/\(used in\) operating activities\s*[^\d\n]*?([\d,\.\(\)\-\s]+)",
                r"Net Cash Flow(?: from Operating Activities)?\s*[^\d\n]*?([\d,\.\(\)\-\s]+)"
            ]
        }
        
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        value_str = match.group(1).strip()
                        # Clean and convert the value, applying scale factor
                        cleaned_value = self._clean_financial_value(value_str)
                        if cleaned_value is not None:
                            setattr(metrics, metric, cleaned_value)
                            break
                    except (ValueError, AttributeError) as e:
                        self.logger.debug(f"Failed to parse {metric}: {e}")
                        continue
        
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
    
    def _extract_shareholders(self, text: str) -> List[Shareholder]:
        """Extract shareholder information from text using a table-like pattern."""
        shareholders = []
        
        # Look for shareholder tables with more flexible boundaries
        shareholder_section_match = re.search(
            r"(?:Twenty Largest Shareholders|Public Share Holdings|Share Information|Shareholders)(.*?)(?=\n\n\n|\Z|NOTES TO THE INTERIM CONDENSED FINANCIAL STATEMENTS|CONTINGENCIES|EVENTS AFTER THE REPORTING PERIOD|Directors' and CEO's share holdings)", 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if shareholder_section_match:
            section_text = shareholder_section_match.group(1)
            self.logger.info(f"Found shareholders section.")
            
            # New pattern to match table-like entries (Rank, Name, Shares, Percentage)
            # This pattern is more robust for space-separated columns
            pattern = re.compile(
                r"^\s*(\d+)\s+" # Rank (Group 1)
                r"([A-Z][A-Za-z\s\.\-']+(?:[A-Z][A-Za-z\s\.\-']+)?(?: PLC| Limited| Ltd| Corp| Inc)?)\s+" # Name (Group 2) - improved to handle multiple words and common suffixes
                r"([\d,]+)\s+" # Shares (Group 3) - allows commas
                r"([\d\.]+)(?:%|)\s*$", # Percentage (Group 4) - allows decimal, optional % sign, ends line
                re.MULTILINE
            )
            
            lines = section_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip header lines, common irrelevant lines, or director shareholdings
                if re.search(r'^(?:Number of shares|%|As at|Name|Shares|Percentage|Directors\' and CEO\'s share holdings|K\. N\. J\. Balendra|J\. G\. A\. Cooray|Nil)$', line, re.IGNORECASE):
                    continue
                # Skip lines that look like a director name followed by Nil or shares (already handled by director extraction)
                if re.match(r'^([A-Z][A-Za-z\s\.\-'']+)(?:\s+Nil|\s+\d+)\s*(?:%|)\s*$', line):
                    continue

                match = pattern.search(line)
                if match:
                    try:
                        rank = int(match.group(1).strip())
                        name = match.group(2).strip()
                        shares = int(match.group(3).replace(',', ''))
                        percentage = float(match.group(4))
                        
                        shareholders.append(Shareholder(rank=rank, name=name, shares=shares, percentage=percentage))
                        self.logger.debug(f"  Found shareholder: {name} (Shares: {shares}, Pct: {percentage})")
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Failed to parse shareholder entry with main pattern: '{line}', error: {e}")
                        continue
        else:
            self.logger.info("Shareholders section header not found.")
        
        return shareholders

    def _extract_directors(self, text: str) -> List[Director]:
        """
        Extracts director names and roles from sections like "Board of Directors".
        Handles multi-line name-role pairs.
        """
        directors = []
        
        # Look for sections related to directors with more flexible boundaries
        director_section_match = re.search(
            r"(Board of Directors|Directors' Report|Directors|Management Team)(.*?)(?=\n\n\n|\Z|FINANCIAL STATEMENTS|SHAREHOLDERS|AUDITORS|NOTES TO THE FINANCIAL STATEMENTS|8\.4 Twenty largest shareholders|Share Information)", # Added Share Information as a clear boundary
            text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if director_section_match:
            section_text = director_section_match.group(2) # Capture content after the heading
            self.logger.info(f"Found directors section: {director_section_match.group(1)}")
            
            # Pattern to find names (starts with capital, allows multiple words, dots, hyphens, etc.)
            name_pattern = re.compile(r"([A-Z][A-Za-z\s\.\-']+(?:[A-Z][A-Za-z\s\.\-']+)?(?: PLC| Limited| Ltd| Corp| Inc)?)(?=\s*[,|\-|\(]?|\n|\Z)")
            
            # Pattern for common roles, potentially on the next line or same line
            role_pattern = re.compile(r"(?:Chief Financial Officer|Chairperson|Chairman|Director|CEO|Managing Director|Company Secretary|Non-Executive(?: Non-Independent)? Director|Independent(?: Non-Executive)? Director|Executive Director|Senior Non-Executive Independent Director|President|Vice President|Secretary|Treasurer|Head of\s+[A-Za-z\s]+)", re.IGNORECASE)

            lines = section_text.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                
                # --- Improved Filtering Logic ---
                # Skip lines that are too short to be a name or look like generic text
                if len(line) < 5 or re.match(r'^(?:No of Shares|Total|Amount|Balance|As at|Number of Shares|Percentage|Position|Note|I certify that|The Board of Directors is responsible|30th July \d{4})$', line, re.IGNORECASE):
                    self.logger.debug(f"  Skipping generic/short line: '{line}'")
                    i += 1
                    continue
                # Skip lines that are primarily numbers or special characters
                if re.match(r'^[\d\s\.,\(\)-]+$', line):
                    self.logger.debug(f"  Skipping numeric line: '{line}'")
                    i += 1
                    continue
                # Skip lines that are entirely uppercase and very short (might be section titles)
                if line.isupper() and len(line.split()) < 4:
                    self.logger.debug(f"  Skipping short uppercase line: '{line}'")
                    i += 1
                    continue
                # Skip lines that start with numbers (e.g., "1. Introduction")
                if re.match(r'^\d+\.', line):
                    self.logger.debug(f"  Skipping numbered line: '{line}'")
                    i += 1
                    continue
                
                name_match = name_pattern.search(line)
                if name_match:
                    name = name_match.group(1).strip()
                    role = None
                    
                    # Check for role on the same line
                    remaining_line = line[name_match.end():].strip()
                    role_on_same_line_match = role_pattern.search(remaining_line)
                    if role_on_same_line_match:
                        role = role_on_same_line_match.group(0).strip()
                    
                    # If no role found on same line, check the next line
                    if role is None and (i + 1) < len(lines):
                        next_line = lines[i+1].strip()
                        role_on_next_line_match = role_pattern.search(next_line)
                        if role_on_next_line_match:
                            role = role_on_next_line_match.group(0).strip()
                            # Advance index if role was found on next line
                            i += 1 
                    
                    # Further refine: if the 'name' is actually a common role, skip it unless it's part of a longer name
                    if re.match(r'^(?:Chief Financial Officer|Chairperson|Director|Company Secretary|Chairman|CEO|Managing Director)$', name, re.IGNORECASE) and len(name.split()) <= 3:
                        self.logger.debug(f"  Skipping role-as-name: '{name}'")
                    else:
                        directors.append(Director(name=name, role=role))
                        self.logger.debug(f"  Found director: {name} (Role: {role})")
                else:
                    self.logger.debug(f"  No director name match in line: '{line}'")
                
                i += 1
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
            r"(\d{1,2}/\d{1,2}/\d{4})"  # "30/06/2025"
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
    
    def extract_financial_data(
        self, 
        pdf_text: str, 
        company_name: Optional[str] = None, 
        company_symbol: Optional[str] = None, 
        report_filename: Optional[str] = None,
        document_hash: Optional[str] = None 
    ) -> ExtractedFinancialData:
        """
        Extract comprehensive financial data from PDF text
        
        Args:
            pdf_text: The full text content of the PDF
            company_name: Optional pre-detected company name
            company_symbol: Optional pre-detected company symbol
            report_filename: Optional PDF filename
            document_hash: Optional hash of the document content
            
        Returns:
            ExtractedFinancialData: Structured financial data
        """
        self.logger.info(f"Starting financial data extraction for {company_name or 'unknown company'}")
        
        try:
            # Extract company info if not provided
            if not company_name or not company_symbol:
                detected_name, detected_symbol = self._extract_company_info(pdf_text)
                company_name = company_name or detected_name
                company_symbol = company_symbol or detected_symbol
            
            # Parse currency unit and scale factor first
            self._parse_currency_unit_and_scale(pdf_text)

            # Create the main data structure
            extracted_data = ExtractedFinancialData(
                company_name=company_name,
                company_symbol=company_symbol,
                document_hash=document_hash,
                currency_unit=self.currency_unit, # Set the detected currency unit
                scale_factor=self.scale_factor # Set the detected scale factor
            )
            
            # Extract various components
            extracted_data.financial_metrics = self._extract_financial_metrics(pdf_text)
            extracted_data.shareholders = self._extract_shareholders(pdf_text)
            extracted_data.directors = self._extract_directors(pdf_text) 
            extracted_data.report_type, extracted_data.report_date = self._extract_report_info(pdf_text)
            
            # Extract special notices
            errata, contingent, events = self._extract_special_notices(pdf_text)
            extracted_data.errata_notice = errata
            extracted_data.contingent_liabilities = contingent
            extracted_data.events_after_reporting = events
            
            # Determine audit status
            if re.search(r"not audited", pdf_text, re.IGNORECASE):
                extracted_data.audit_status = "Unaudited"
            elif re.search(r"audited", pdf_text, re.IGNORECASE):
                extracted_data.audit_status = "Audited"
            
            # Calculate extraction confidence (simple heuristic)
            confidence_score = self._calculate_confidence(extracted_data)
            extracted_data.extraction_confidence = confidence_score
            
            self.logger.info(f"Financial data extraction completed with {confidence_score:.2f} confidence")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error during financial data extraction: {e}")
            # Return minimal data structure
            return ExtractedFinancialData(
                company_name=company_name or "Unknown",
                company_symbol=company_symbol or "Unknown",
                document_hash=document_hash,
                currency_unit=self.currency_unit, # Ensure these are set even on error
                scale_factor=self.scale_factor
            )
    
    def _calculate_confidence(self, data: ExtractedFinancialData) -> float:
        """Calculate extraction confidence score"""
        score = 0.0
        total_possible = 12.0 # Updated total possible points
        
        # Company info (2 points)
        if data.company_name != "Unknown":
            score += 1.0
        if data.company_symbol != "Unknown":
            score += 1.0
        
        # Financial metrics (4 points - Revenue, Profit, Operating Profit, EPS)
        metrics = data.financial_metrics
        if metrics.revenue is not None:
            score += 1.0
        if metrics.profit_for_period is not None:
            score += 1.0
        if metrics.operating_profit is not None:
            score += 1.0
        if metrics.basic_eps is not None:
            score += 1.0
        
        # Report info (2 points - Type, Date)
        if data.report_type != "Unknown":
            score += 1.0
        if data.report_date != "Unknown":
            score += 1.0
        
        # Additional data (4 points - Shareholders, Directors, Audit Status, Notices)
        if data.shareholders:
            score += 1.0
        if data.directors: # New: Directors
            score += 1.0
        if data.audit_status != "Unknown":
            score += 1.0
        if data.errata_notice or data.contingent_liabilities or data.events_after_reporting: # At least one notice
            score += 1.0
        
        return score / total_possible
    
    def to_legacy_format(self, data: ExtractedFinancialData) -> Dict[str, Any]:
        """Convert to the original dictionary format for backward compatibility"""
        return {
            "company_name": data.company_name, 
            "company_symbol": data.company_symbol, 
            "report_type": data.report_type,
            "report_date": data.report_date,
            "currency_unit": data.currency_unit,
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
                "net_cash_flow": data.financial_metrics.net_cash_flow
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
            "extraction_confidence": data.extraction_confidence 
        }


# Convenience function for backward compatibility (if needed for direct calls)
def extract_financial_data_with_custom_ml(
    pdf_text: str, 
    company_name: str, 
    company_symbol: str, 
    report_filename: str,
    document_hash: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Legacy function wrapper for backward compatibility.
    Initializes FinancialDataExtractor and calls its method.
    """
    extractor = FinancialDataExtractor()
    result = extractor.extract_financial_data(pdf_text, company_name, company_symbol, report_filename, document_hash) 
    return extractor.to_legacy_format(result)


# Example Usage and Testing
if __name__ == "__main__":
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
    extractor = FinancialDataExtractor()
    result = extractor.extract_financial_data(sample_pdf_text)
    
    print(f"Company: {result.company_name} ({result.company_symbol})")
    print(f"Report Type: {result.report_type}, Date: {result.report_date}")
    print(f"Confidence Score: {result.extraction_confidence:.2f}")
    print(f"Revenue: {result.financial_metrics.revenue}")
    print(f"Profit: {result.financial_metrics.profit_for_period}")
    print(f"EPS: {result.financial_metrics.basic_eps}")
    print(f"Total Assets: {result.financial_metrics.total_assets}")
    print(f"Total Liabilities: {result.financial_metrics.total_liabilities}")
    print(f"Total Equity: {result.financial_metrics.total_equity}")
    print(f"Gross Profit: {result.financial_metrics.gross_profit}")
    print(f"EBITDA: {result.financial_metrics.ebitda}")
    print(f"Net Cash Flow: {result.financial_metrics.net_cash_flow}")
    print(f"Top Shareholders: {len(result.shareholders)}")
    print(f"Directors Found: {len(result.directors)}")
    
    # Test legacy format conversion
    legacy_format = extractor.to_legacy_format(result)
    print("\n--- Legacy Format Output ---")
    print(json.dumps(legacy_format, indent=2, ensure_ascii=False))
