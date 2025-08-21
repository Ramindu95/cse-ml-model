import json
from pathlib import Path
from ml_models_manager import FinancialMLModelManager

# You can use any PDF parser. Here is a simple example using pdfplumber for text extraction.
import pdfplumber

def extract_pdf_text(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    return all_text

def main(pdf_path: str, model_dir: str = "ml_models", output_json: str = "extracted_financial_data.json"):
    # Step 1: Extract text from PDF
    pdf_text = extract_pdf_text(pdf_path)
    
    # Step 2: Initialize the ML model manager
    manager = FinancialMLModelManager(model_dir=model_dir)
    
    # Step 3: Predict all financial fields from the PDF text
    results = manager.predict_all(pdf_text)
    
    # Step 4: Prepare results for JSON output
    json_ready = {field: value for field, (value, conf) in results.items()}
    
    # Step 5: Save to JSON file
    with open(output_json, "w") as f:
        json.dump(json_ready, f, indent=2)
    
    print(f"Extraction complete. Results saved to {output_json}")
    print(json.dumps(json_ready, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_fs_data.py <path_to_pdf> [model_dir] [output_json]")
    else:
        pdf_path = sys.argv[1]
        model_dir = sys.argv[2] if len(sys.argv) > 2 else "ml_models"
        output_json = sys.argv[3] if len(sys.argv) > 3 else "extracted_financial_data.json"
        main(pdf_path, model_dir, output_json)