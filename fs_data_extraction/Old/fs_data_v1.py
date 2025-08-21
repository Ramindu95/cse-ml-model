import os
import fitz  # PyMuPDF
import json
import mysql.connector
from dotenv import load_dotenv
from fs_parser import extract_income_statement, extract_balance_sheet, extract_shareholders # Import updated functions
import logging # Added for logging
import pytesseract # For OCR
from PIL import Image # For image processing in OCR
from fuzzywuzzy import fuzz # For fuzzy string matching
import re # Ensure 're' is explicitly imported at the top

# Configure logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env
load_dotenv()

# Constants
RAW_FS_DIR = 'raw_fs'     # Folder containing manually downloaded PDFs
OUTPUT_DIR = 'fs_data'    # Folder where structured fs_data.json will be stored

# --- MySQL: Load company list ---
def load_companies():
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
        logging.info(f"Loaded {len(companies)} companies from database.")
        return companies
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to or querying database: {err}")
        return []

def _extract_symbol_from_filename(filename):
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

# --- Try to detect matching company by name or symbol ---
def detect_company(text, companies, fuzzy_threshold=80):
    """
    Detects a company from the provided text based on a list of companies.
    Prioritizes exact name matches, then uses fuzzy matching.
    Also attempts to match by symbol if found in text.
    """
    text_lower = text.lower()
    
    # 1. Try exact name match
    for company in companies:
        if company['name'].lower() in text_lower:
            logging.info(f"Detected company '{company['name']}' by exact name match.")
            return company
    
    # 2. Try symbol match from text
    # This regex is a simplified version, assuming symbols are 3-5 uppercase letters
    # For more robust symbol extraction from text, you might need a dedicated function
    symbol_match_in_text = re.search(r'\b([A-Z]{3,5})\b', text_lower)
    if symbol_match_in_text:
        extracted_symbol = symbol_match_in_text.group(1).upper()
        for company in companies:
            if company['symbol'].upper() == extracted_symbol:
                logging.info(f"Detected company '{company['name']}' by symbol '{extracted_symbol}' from text.")
                return company

    # 3. Try fuzzy name match using partial_ratio for better substring matching
    logging.info(f"Attempting fuzzy matching for company name (threshold: {fuzzy_threshold})...")
    best_match_company = None
    best_score = -1
    
    # Iterate through all company names and calculate fuzzy partial ratio
    for company in companies:
        # Use fuzz.partial_ratio for finding substring matches
        score = fuzz.partial_ratio(company['name'].lower(), text_lower) 
        if score > best_score and score >= fuzzy_threshold:
            best_score = score
            best_match_company = company
            
    if best_match_company:
        logging.info(f"Detected company '{best_match_company['name']}' by fuzzy partial name match (score: {best_score}).")
        return best_match_company
        
    logging.warning("No company detected from text via exact, symbol, or fuzzy matching.")
    return None

# --- Extract all text from a PDF file, with OCR fallback ---
def extract_text(pdf_path):
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
                    logging.debug(f"Page {page_num + 1}: Direct text extraction successful.")
                else:
                    logging.info(f"Page {page_num + 1}: Little text extracted directly. Attempting OCR...")
                    # Render page to a high-resolution image
                    # Use a higher DPI for better OCR accuracy, but increases memory/time
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) # Render at 3x resolution (e.g., 300 DPI)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Apply OCR
                    try:
                        # lang='eng' for English. Add other languages if needed (e.g., 'eng+sin' for Sinhala)
                        ocr_text = pytesseract.image_to_string(img, lang='eng') 
                        text += ocr_text
                        logging.info(f"Page {page_num + 1}: OCR successful.")
                    except pytesseract.TesseractNotFoundError:
                        logging.error("Tesseract is not installed or not in PATH. OCR will not work.")
                        text += page_text # Fallback to direct text if OCR fails
                    except Exception as ocr_e:
                        logging.error(f"Error during OCR on page {page_num + 1}: {ocr_e}")
                        text += page_text # Fallback to direct text if OCR fails
                
                if page_num < len(doc) - 1: # Add a separator between pages
                    text += "\n--- PAGE BREAK ---\n" 
        logging.info(f"Successfully extracted text from {pdf_path} (with potential OCR fallback).")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return None

# --- Main processor ---
def process_pdfs():
    """Main function to process PDFs, extract financial data, and save as JSON."""
    # Ensure folders exist
    os.makedirs(RAW_FS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Ensured '{RAW_FS_DIR}' and '{OUTPUT_DIR}' directories exist.")

    # Load company reference list
    companies = load_companies()
    if not companies:
        logging.error("No companies loaded. Exiting.")
        return

    # Iterate through all PDFs in raw_fs/
    pdf_files = [f for f in os.listdir(RAW_FS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(f"No PDF files found in '{RAW_FS_DIR}'.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(RAW_FS_DIR, filename)
        logging.info(f"\nProcessing PDF: {filename}")

        matched_company = None
        full_text = None # Initialize full_text

        # --- Attempt 1: Detect company by symbol in filename (most reliable if present) ---
        symbol_from_filename = _extract_symbol_from_filename(filename)
        if symbol_from_filename:
            # Look up company by symbol in DB
            for company in companies:
                if company['symbol'].upper() == symbol_from_filename:
                    matched_company = company
                    logging.info(f"Detected company '{matched_company['name']}' by symbol in filename.")
                    break
        
        if not matched_company:
            # --- Attempt 2: Extract text from PDF for detection (first page then full text) ---
            try:
                with fitz.open(pdf_path) as doc:
                    if len(doc) > 0:
                        first_page_text = doc[0].get_text()
                        # Try to detect company from first page text (exact/fuzzy)
                        matched_company = detect_company(first_page_text, companies)
                        
                        if not matched_company and len(doc) > 1:
                            # If not found on first page, extract full text with OCR fallback
                            logging.info(f"Company not detected on first page. Extracting full text for {filename}...")
                            full_text = extract_text(pdf_path) # This will use OCR if needed
                            if full_text:
                                matched_company = detect_company(full_text, companies)
                    else:
                        logging.warning(f"PDF {filename} is empty. Skipping.")
                        continue
            except Exception as e:
                logging.error(f"❌ Failed to read PDF {filename} for company detection: {e}")
                continue

        if not matched_company:
            logging.warning(f"❌ Cannot detect company from: {filename}. Skipping.")
            continue

        # If full_text wasn't extracted during detection, extract it now for parsing
        if not full_text:
            full_text = extract_text(pdf_path)
        
        if not full_text:
            logging.error(f"No text extracted from {filename} even after OCR. Skipping financial statement parsing.")
            continue

        # Prepare folder name (safe from slashes, spaces)
        folder_name = f"{matched_company['name'].replace('/', '_').replace(' ', '_')}_{matched_company['symbol']}_{matched_company['id']}"
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"Output folder for '{filename}': {folder_path}")

        # Extract financial statements using the expanded functions
        income_statement = extract_income_statement(full_text)
        balance_sheet = extract_balance_sheet(full_text)
        shareholders = extract_shareholders(full_text) # New: Extract shareholders

        # Save JSON data
        output_path = os.path.join(folder_path, "fs_data.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "company_id": matched_company["id"],
                    "symbol": matched_company["symbol"],
                    "name": matched_company["name"],
                    "source_pdf": filename,
                    # "full_text": full_text, # Uncomment if you want to store the full text in JSON
                    "income_statement": income_statement,
                    "balance_sheet": balance_sheet,
                    "shareholders": shareholders # New: Add shareholders data
                }, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ Processed: {filename} → {folder_name}/fs_data.json")
        except Exception as e:
            logging.error(f"❌ Failed to save JSON for {filename}: {e}")

# --- Entry point ---
if __name__ == "__main__":
    logging.info("Starting PDF Financial Statement Processor.")
    logging.info(f"PDFs will be read from: {RAW_FS_DIR}")
    logging.info(f"JSON output will be stored in: {OUTPUT_DIR}")
    process_pdfs()
    logging.info("PDF Financial Statement Processing completed.")
