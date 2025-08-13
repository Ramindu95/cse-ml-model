import os
import fitz  # PyMuPDF
import json
import mysql.connector
from dotenv import load_dotenv
# Corrected: Import FinancialDataExtractor instead of FinancialParser
from fs_parser import FinancialDataExtractor, ExtractedFinancialData, FinancialMetrics, Shareholder, Director
import logging
import pytesseract  # For OCR
from PIL import Image  # For image processing in OCR
from fuzzywuzzy import fuzz  # For fuzzy string matching
import re
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
import hashlib # For document hashing
from dataclasses import asdict # Import asdict for dataclass serialization

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Constants
RAW_FS_DIR = 'raw_fs'     # Folder containing manually downloaded PDFs
OUTPUT_DIR = 'fs_data'    # Folder where structured fs_data.json will be stored


class PDFFinancialProcessor:
    """
    Enhanced PDF financial statement processor using the FinancialDataExtractor
    """

    def __init__(self, ml_model_dir: Optional[str] = "ml_models", db_path: Optional[str] = "extraction_learning.db"):
        """
        Initialize the processor

        Args:
            ml_model_dir: Directory where ML models are stored/loaded from.
            db_path: Path to the SQLite database for learning.
        """
        # Initialize the FinancialDataExtractor
        # Removed 'model_path' argument as per the TypeError.
        # If FinancialDataExtractor needs this path, it must be handled internally
        # or passed via a different method/property after initialization.
        self.extractor = FinancialDataExtractor()
        self.companies = []

    def load_companies(self) -> List[Dict[str, Any]]:
        """Loads company details from the MySQL database."""
        try:
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                port=int(os.getenv("DB_PORT", 3306))
            )
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, name, symbol FROM companies WHERE deleted = 0 OR deleted IS NULL")
            companies = cursor.fetchall()
            conn.close()
            logger.info(f"Loaded {len(companies)} companies from database.")
            self.companies = companies
            return companies
        except mysql.connector.Error as err:
            logger.error(f"Error connecting to or querying database: {err}")
            return []

    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """
        Attempts to extract a company symbol (e.g., 'ABAN', 'CDB') from the PDF filename.
        Assumes symbols are typically 3-5 uppercase letters.
        """
        # Example filename: 642_1739527719477.12.2024.pdf -> try to find ABAN
        # Also handles cases like ABC.N0000.pdf
        match = re.search(r'\b([A-Z]{3,5})(?:\.[NX]\d{4})?\b', filename.upper())
        if match:
            symbol = match.group(1)
            # Add common words that might accidentally match regex but aren't symbols
            if symbol not in ['PDF', 'FILE', 'REPORT', 'DATA', 'ANNUAL', 'INTERIM']:
                return symbol
        return None

    def detect_company(self, text: str, fuzzy_threshold: int = 80) -> Optional[Dict[str, Any]]:
        """
        Detects a company from the provided text based on the loaded companies list.
        Prioritizes exact name matches, then uses fuzzy matching.
        Also attempts to match by symbol if found in text.
        """
        if not self.companies:
            logger.warning("No companies loaded for detection")
            return None

        text_lower = text.lower()

        # 1. Try exact name match
        for company in self.companies:
            if company['name'].lower() in text_lower:
                logger.info(f"Detected company '{company['name']}' by exact name match.")
                return company

        # 2. Try symbol match from text
        symbol_matches = re.findall(r'\b([A-Z]{3,5})\b', text)
        for symbol_candidate in symbol_matches:
            for company in self.companies:
                if company['symbol'].upper() == symbol_candidate.upper():
                    logger.info(f"Detected company '{company['name']}' by symbol '{symbol_candidate}' from text.")
                    return company

        # 3. Try fuzzy name match using partial_ratio for better substring matching
        logger.info(f"Attempting fuzzy matching for company name (threshold: {fuzzy_threshold})...")
        best_match_company = None
        best_score = -1

        # Iterate through all company names and calculate fuzzy partial ratio
        for company in self.companies:
            # Use fuzz.partial_ratio for finding substring matches
            score = fuzz.partial_ratio(company['name'].lower(), text_lower)
            if score > best_score and score >= fuzzy_threshold:
                best_score = score
                best_match_company = company

        if best_match_company:
            logger.info(f"Detected company '{best_match_company['name']}' by fuzzy partial name match (score: {best_score}).")
            return best_match_company

        logger.warning("No company detected from text via exact, symbol, or fuzzy matching.")
        return None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extracts all text from a PDF file using PyMuPDF (fitz).
        If initial text extraction yields little content, it falls back to OCR.
        """
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()

                    # Check if direct text extraction yielded significant content
                    # A threshold of 50 characters is arbitrary; adjust as needed
                    if len(page_text.strip()) > 50:
                        text += page_text
                        logger.debug(f"Page {page_num + 1}: Direct text extraction successful.")
                    else:
                        logger.info(f"Page {page_num + 1}: Little text extracted directly. Attempting OCR...")
                        # Render page to a high-resolution image
                        # Use a higher DPI for better OCR accuracy, but increases memory/time
                        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Render at 3x resolution
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        # Apply OCR
                        try:
                            # lang='eng' for English. Add other languages if needed
                            ocr_text = pytesseract.image_to_string(img, lang='eng')
                            text += ocr_text
                            logger.info(f"Page {page_num + 1}: OCR successful.")
                        except pytesseract.TesseractNotFoundError:
                            logger.error("Tesseract is not installed or not in PATH. OCR will not work.")
                            text += page_text  # Fallback to direct text if OCR fails
                        except Exception as ocr_e:
                            logger.error(f"Error during OCR on page {page_num + 1}: {ocr_e}")
                            text += page_text  # Fallback to direct text if OCR fails

                    if page_num < len(doc) - 1:  # Add a separator between pages
                        text += "\n--- PAGE BREAK ---\n"
            logger.info(f"Successfully extracted text from {pdf_path} (with potential OCR fallback).")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return None

    def process_single_pdf(self, filename: str, pdf_path: str) -> bool:
        """
        Process a single PDF file and extract financial data

        Args:
            filename: Name of the PDF file
            pdf_path: Full path to the PDF file

        Returns:
            bool: True if processing was successful
        """
        logger.info(f"\nProcessing PDF: {filename}")

        matched_company = None
        full_text = None

        # --- Attempt 1: Detect company by symbol in filename (most reliable if present) ---
        symbol_from_filename = self._extract_symbol_from_filename(filename)
        if symbol_from_filename:
            # Look up company by symbol in DB
            for company in self.companies:
                if company['symbol'].upper() == symbol_from_filename:
                    matched_company = company
                    logger.info(f"Detected company '{matched_company['name']}' by symbol in filename.")
                    break

        if not matched_company:
            # --- Attempt 2: Extract text from PDF for detection (first page then full text) ---
            try:
                with fitz.open(pdf_path) as doc:
                    if len(doc) > 0:
                        first_page_text = doc[0].get_text()
                        # Try to detect company from first page text (exact/fuzzy)
                        matched_company = self.detect_company(first_page_text)

                        if not matched_company and len(doc) > 1:
                            # If not found on first page, extract full text with OCR fallback
                            logger.info(f"Company not detected on first page. Extracting full text for {filename}...")
                            full_text = self.extract_text(pdf_path)  # This will use OCR if needed
                            if full_text:
                                matched_company = self.detect_company(full_text)
                    else:
                        logger.warning(f"PDF {filename} is empty. Skipping.")
                        return False
            except Exception as e:
                logger.error(f"❌ Failed to read PDF {filename} for company detection: {e}")
                return False

        if not matched_company:
            logger.warning(f"❌ Cannot detect company from: {filename}. Skipping.")
            return False

        # If full_text wasn't extracted during detection, extract it now for parsing
        if not full_text:
            full_text = self.extract_text(pdf_path)

        if not full_text:
            logger.error(f"No text extracted from {filename} even after OCR. Skipping financial statement parsing.")
            return False

        # Calculate document hash
        document_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()

        # Use the FinancialDataExtractor
        try:
            # Call extract_financial_data method of the FinancialDataExtractor
            extracted_data = self.extractor.extract_financial_data(
                pdf_text=full_text,
                company_name=matched_company['name'],
                company_symbol=matched_company['symbol'],
                report_filename=filename
            )

            # Update company name/symbol in extracted_data if detection was more accurate
            # Ensure extracted_data.company_name is not None before comparing
            if matched_company and (extracted_data.company_name is None or \
               (extracted_data.company_name and fuzz.ratio(matched_company['name'].lower(), extracted_data.company_name.lower()) < 90)):
                extracted_data.company_name = matched_company['name']

            # Ensure extracted_data.company_symbol is not None before comparing
            if matched_company and (extracted_data.company_symbol is None or \
               (extracted_data.company_symbol and fuzz.ratio(matched_company['symbol'].lower(), extracted_data.company_symbol.lower()) < 90)):
                extracted_data.company_symbol = matched_company['symbol']

            # Prepare folder name (safe from slashes, spaces)
            # Use extracted_data's company name/symbol as they might have been updated
            company_name_for_folder = extracted_data.company_name if extracted_data.company_name else "UnknownCompany"
            company_symbol_for_folder = extracted_data.company_symbol if extracted_data.company_symbol else "UNK"
            company_id_for_folder = matched_company.get('id', 'unknown') # Still use matched_company ID if available

            folder_name = f"{company_name_for_folder.replace('/', '_').replace(' ', '_')}_{company_symbol_for_folder}_{company_id_for_folder}"
            folder_path = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Output folder for '{filename}': {folder_path}")

            # Save JSON data
            output_path = os.path.join(folder_path, "fs_data.json")
            with open(output_path, "w", encoding="utf-8") as f:
                # Use asdict to convert dataclass to dict for JSON serialization
                # The ExtractedFinancialData dataclass now contains all necessary fields directly
                json.dump(asdict(extracted_data), f, indent=2, default=str, ensure_ascii=False)

            # Also save a summary report
            summary_path = os.path.join(folder_path, "extraction_summary.txt")
            self._save_extraction_summary(extracted_data, summary_path)

            logger.info(f"✅ Processed: {filename} → {folder_name}/fs_data.json (Confidence: {extracted_data.extraction_confidence:.2f})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to extract financial data from {filename}: {e}")
            return False

    def _save_extraction_summary(self, data: ExtractedFinancialData, summary_path: str):
        """Save a human-readable summary of the extraction"""
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"Financial Data Extraction Summary\n")
                f.write(f"==================================\n\n")
                f.write(f"Company: {data.company_name} ({data.company_symbol})\n")
                f.write(f"Report Type: {data.report_type}\n")
                f.write(f"Report Date: {data.report_date}\n")
                # Removed currency_unit and Audit Status as they are not top-level fields in ExtractedFinancialData
                f.write(f"Extraction Confidence: {data.extraction_confidence:.2f}\n")
                # The ExtractedFinancialData dataclass doesn't have extraction_method or processing_time directly
                # If these are desired, they should be added to ExtractedFinancialData and populated
                # f.write(f"Extraction Method: {data.extraction_method}\n")
                # f.write(f"Processing Time: {data.processing_time:.2f} seconds\n")
                f.write(f"Document Hash: {data.document_hash}\n\n")

                f.write(f"Financial Metrics:\n")
                # Iterate through financial_metrics attributes
                for field_name, value in asdict(data.financial_metrics).items():
                    f.write(f"- {field_name.replace('_', ' ').title()}: {value}\n")
                f.write("\n")

                if data.shareholders:
                    f.write(f"Shareholders ({len(data.shareholders)}):\n")
                    for sh in data.shareholders: # No rank in Shareholder dataclass, iterate directly
                        f.write(f"  - {sh.name} - {sh.shares} shares ({sh.percentage}%)\n")
                    f.write("\n")

                if data.directors:
                    f.write(f"Directors ({len(data.directors)}):\n")
                    for d in data.directors:
                        f.write(f"  - {d.name} ({d.role or 'N/A'})\n")
                    f.write("\n")

                # Removed errata_notice, contingent_liabilities, events_after_reporting
                # as they are not top-level fields in ExtractedFinancialData

        except Exception as e:
            logger.error(f"Failed to save extraction summary: {e}")

    def process_pdfs(self) -> Dict[str, int]:
        """
        Main function to process all PDFs in the raw_fs directory

        Returns:
            Dict with processing statistics
        """
        # Ensure folders exist
        os.makedirs(RAW_FS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Ensured '{RAW_FS_DIR}' and '{OUTPUT_DIR}' directories exist.")

        # Load company reference list
        companies = self.load_companies()
        if not companies:
            logger.error("No companies loaded. Exiting.")
            return {"total": 0, "successful": 0, "failed": 0}

        # Iterate through all PDFs in raw_fs/
        pdf_files = [f for f in os.listdir(RAW_FS_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No PDF files found in '{RAW_FS_DIR}'.")
            return {"total": 0, "successful": 0, "failed": 0}

        stats = {"total": len(pdf_files), "successful": 0, "failed": 0}

        for filename in pdf_files:
            pdf_path = os.path.join(RAW_FS_DIR, filename)

            if self.process_single_pdf(filename, pdf_path):
                stats["successful"] += 1
            else:
                stats["failed"] += 1

        # Log final statistics
        logger.info(f"\n📊 Processing Complete!")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")

        if stats['total'] > 0:
            logger.info(f"Success rate: {(stats['successful']/stats['total']*100):.1f}%")
        else:
            logger.info("No files to process, success rate N/A.")

        return stats


# --- Entry point ---
if __name__ == "__main__":
    logger.info("Starting Enhanced PDF Financial Statement Processor (Self-Learning Capable).")
    logger.info(f"PDFs will be read from: {RAW_FS_DIR}")
    logger.info(f"JSON output will be stored in: {OUTPUT_DIR}")

    # Initialize processor with optional ML model directory and DB path
    # You can set these via environment variables if preferred
    ml_model_dir = os.getenv("ML_MODEL_DIR", "ml_models")
    db_path = os.getenv("DB_PATH", "extraction_learning.db")

    processor = PDFFinancialProcessor(ml_model_dir=ml_model_dir, db_path=db_path)

    # Process all PDFs
    stats = processor.process_pdfs()

    logger.info("Enhanced PDF Financial Statement Processing completed.")

    # Exit with appropriate code
    if stats["failed"] > 0:
        exit(1 if stats["successful"] == 0 else 0)  # Exit 1 if all failed, 0 if some succeeded
