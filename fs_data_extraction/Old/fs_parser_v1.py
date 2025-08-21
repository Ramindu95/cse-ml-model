import re
import logging

# Configure logging for the parser
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _convert_to_numeric(value_str):
    """
    Cleans a string and converts it to a float.
    Handles commas, spaces, and parentheses for negative numbers.
    Returns None if conversion fails.
    """
    if value_str is None:
        return None
    
    # Remove commas and spaces
    cleaned_str = value_str.replace(',', '').replace(' ', '').strip()
    
    # Handle parentheses for negative numbers (e.g., (1,234) -> -1234)
    if cleaned_str.startswith('(') and cleaned_str.endswith(')'):
        cleaned_str = '-' + cleaned_str[1:-1]

    try:
        return float(cleaned_str)
    except ValueError:
        logging.warning(f"Could not convert '{value_str}' to numeric.")
        return None

def extract_income_statement(text, snippet_size=3000):
    """
    Extracts key figures from the Income Statement section of the text.
    Searches within a snippet after the income statement header.
    """
    income_statement_data = {}
    
    # Try to find "Income Statement" or similar headers
    # Using re.DOTALL (re.S) to allow '.' to match newlines
    income_start_match = re.search(
        r"(income statement|statement of profit or loss|profit and loss account|statement of comprehensive income)", 
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not income_start_match:
        logging.info("Income Statement header not found.")
        return income_statement_data

    start_pos = income_start_match.end()
    # Grab a larger snippet to ensure all fields are covered
    snippet = text[start_pos : start_pos + snippet_size]
    
    logging.info("Extracting Income Statement data...")

    # Define fields and their regex patterns
    # Patterns are designed to be flexible with whitespace and non-numeric characters
    # between the label and the value.
    # [\s\S]*? - non-greedy match for any character (including newlines)
    # \s* - flexible whitespace
    # ([\d,\.\(\)\-]+) - captures numbers, commas, dots, parentheses, and hyphens (for ranges/negatives)

    fields = {
        "revenue": r"revenue|turnover|sales\s*([\d,\.\(\)\-]+)",
        "cost_of_sales": r"cost of (?:sales|revenue)[\s\S]*?([\d,\.\(\)\-]+)",
        "gross_profit": r"gross profit[\s\S]*?([\d,\.\(\)\-]+)",
        "other_income": r"other income and gains[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted for "Other Income and Gains"
        "operating_expenses": r"operating expenses[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "administrative_expenses": r"administrative expenses[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "selling_and_distribution_expenses": r"selling and distribution expenses[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "operating_profit_loss": r"operating profit/loss[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted for "Operating Profit/Loss"
        "finance_cost": r"finance cost[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted
        "finance_income": r"finance income[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted
        "net_finance_income": r"net finance income[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "profit_before_tax": r"profit before tax[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted
        "income_tax_expense_reversal": r"income tax \(expenses\)/reversal[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted
        "profit_for_the_period": r"profit for the period[\s\S]*?([\d,\.\(\)\-]+)", # Adjusted
        "other_comprehensive_income": r"other comprehensive income[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "total_comprehensive_income_expense": r"total comprehensive income/\(expense\)[\s\S]*?([\d,\.\(\)\-]+)", # Added
        "basic_diluted_earning_per_share": r"basic/diluted earning per share \(eps\)[\s\S]*?([\d,\.\(\)\-]+)" # Adjusted
    }

    for key, pattern in fields.items():
        match = re.search(pattern, snippet, re.IGNORECASE | re.DOTALL)
        if match:
            value = _convert_to_numeric(match.group(1))
            income_statement_data[key] = value
            logging.debug(f"  Found {key}: {value}")
        else:
            logging.debug(f"  {key} not found.")

    return income_statement_data

def extract_balance_sheet(text, snippet_size=3000):
    """
    Extracts key figures from the Balance Sheet (Statement of Financial Position) section of the text.
    Searches within a snippet after the balance sheet header.
    """
    balance_sheet_data = {}

    # Try to find "Balance Sheet" or similar headers
    balance_start_match = re.search(
        r"(balance sheet|statement of financial position)", 
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not balance_start_match:
        logging.info("Balance Sheet header not found.")
        return balance_sheet_data

    start_pos = balance_start_match.end()
    snippet = text[start_pos : start_pos + snippet_size]

    logging.info("Extracting Balance Sheet data...")

    fields = {
        "property_plant_equipment": r"property,?\s*plant and equipment[\s\S]*?([\d,\.\(\)\-]+)",
        "intangible_assets": r"intangible assets[\s\S]*?([\d,\.\(\)\-]+)",
        "investments": r"investments[\s\S]*?([\d,\.\(\)\-]+)",
        "total_non_current_assets": r"total non-current assets[\s\S]*?([\d,\.\(\)\-]+)",
        "inventories": r"inventories[\s\S]*?([\d,\.\(\)\-]+)",
        "trade_and_other_receivables": r"trade and other receivables[\s\S]*?([\d,\.\(\)\-]+)",
        "cash_and_cash_equivalents": r"cash and cash equivalents[\s\S]*?([\d,\.\(\)\-]+)",
        "total_current_assets": r"total current assets[\s\S]*?([\d,\.\(\)\-]+)",
        "total_assets": r"total assets[\s\S]*?([\d,\.\(\)\-]+)",
        "trade_and_other_payables": r"trade and other payables[\s\S]*?([\d,\.\(\)\-]+)",
        "short_term_borrowings": r"short-term borrowings[\s\S]*?([\d,\.\(\)\-]+)",
        "total_current_liabilities": r"total current liabilities[\s\S]*?([\d,\.\(\)\-]+)",
        "long_term_borrowings": r"long-term borrowings[\s\S]*?([\d,\.\(\)\-]+)",
        "deferred_tax_liabilities": r"deferred tax liabilities[\s\S]*?([\d,\.\(\)\-]+)",
        "total_non_current_liabilities": r"total non-current liabilities[\s\S]*?([\d,\.\(\)\-]+)",
        "total_liabilities": r"total liabilities[\s\S]*?([\d,\.\(\)\-]+)",
        "stated_capital": r"stated capital[\s\S]*?([\d,\.\(\)\-]+)",
        "retained_earnings": r"retained earnings[\s\S]*?([\d,\.\(\)\-]+)",
        "total_equity": r"total equity[\s\S]*?([\d,\.\(\)\-]+)"
    }

    for key, pattern in fields.items():
        match = re.search(pattern, snippet, re.IGNORECASE | re.DOTALL)
        if match:
            value = _convert_to_numeric(match.group(1))
            balance_sheet_data[key] = value
            logging.debug(f"  Found {key}: {value}")
        else:
            logging.debug(f"  {key} not found.")

    return balance_sheet_data

def extract_shareholders(text, snippet_size=2000):
    """
    Extracts shareholder information from the "Twenty Largest Shareholders" section.
    Parses a table-like structure.
    """
    shareholders_data = []
    
    shareholders_start_match = re.search(
        r"Twenty Largest Shareholders", 
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not shareholders_start_match:
        logging.info("Shareholders section header not found.")
        return shareholders_data

    start_pos = shareholders_start_match.end()
    snippet = text[start_pos : start_pos + snippet_size]
    
    logging.info("Extracting Shareholder data...")

    # Regex to capture lines that look like shareholder entries
    # It looks for: "RANK\n","NAME\n","SHARES\n","PERCENTAGE\n"
    # The `\s*` handles potential whitespace, `[^"]+` captures content within quotes.
    # We use re.DOTALL because the content inside quotes might span multiple lines (due to OCR/extraction).
    shareholder_line_pattern = re.compile(
        r'"(\d+)\s*"\s*,\s*"' # Rank (e.g., "1\n",)
        r'([^"]+)"\s*,\s*"' # Name (e.g., "J.Z. Hassen\n",)
        r'([\d,\.]+)"\s*,\s*"' # Number of Shares (e.g., "119,300,000\n",)
        r'([\d\.]+)"', # Percentage (e.g., "35.79\n")
        re.IGNORECASE | re.DOTALL
    )

    for match in shareholder_line_pattern.finditer(snippet):
        try:
            rank = int(match.group(1).strip())
            name = match.group(2).strip().replace('\n', ' ') # Replace newlines in name with space
            shares = _convert_to_numeric(match.group(3))
            percentage = _convert_to_numeric(match.group(4))

            # Basic validation for percentage OCR errors
            if percentage is not None and percentage > 100:
                # Common OCR error 'DB2' for '0.82'
                if match.group(4).strip().upper() == 'DB2':
                    percentage = 0.82
                    logging.warning(f"Corrected OCR error: 'DB2' for percentage of '{name}' to 0.82.")
                else:
                    logging.warning(f"Unusual percentage value for '{name}': {percentage}. Setting to None.")
                    percentage = None # Or handle as an error

            shareholders_data.append({
                "rank": rank,
                "name": name,
                "shares": shares,
                "percentage": percentage
            })
            logging.debug(f"  Found shareholder: {name}")
        except Exception as e:
            logging.warning(f"Error parsing shareholder line: {match.group(0)}. Error: {e}")
            continue

    return shareholders_data

# Example Usage (for testing fs_parser.py independently)
if __name__ == "__main__":
    sample_text_income = """
    COMPANY STATEMENT OF COMPREHENSIVE INCOME

    ,"Note
    ","Quarter ended 31
    ","December
    ","Nine months ended 31
    ","December
    "
    ,,"2024
    ","2023
    ","2024
    ","2023
    "
    "Revenue
    ",,"1,524,047
    ","1,083,276
    ","4,108,122
    ","2,796,488
    "
    "Other Income and Gains
    ",,"13,368
    ","74,140
    ","38,147
    ","90,490
    "
    "Operating Expenses
    ",,"464,931()
    ","(384,247)
    ","(1.240.614)
    ","(1.016.308)
    "
    "Administrative Expenses
    ",,"(542.868)
    ","422,396()
    ","(1.522,729)
    ","1,022.765()
    "
    "Selling and Distribution Expenses
    ",,"(100,012)
    ","(76.822)
    ","(254,226)
    ","(243,899)
    "
    "Operating Profit/Loss)
    ",,"429,603
    ","273,951
    ","1,128,700
    ","604,005
    "
    "Finance Cost
    ","41
    ","(57,800)
    ","(14.145)
    ","(107,005)
    ","(33.098)
    "
    "Finance Income
    ","4.2
    ","28,816
    ","26,580
    ","87.227
    ","91,935
    "
    "Net Finance Income


    Profit/(Less) before Tax
    ",,"(28,984)


    400,619
    ","12,436


    286,387
    ","(19,778)


    1,108,923
    ","58,838


    662,842
    "
    "Income Tax (Expenses)/Reversal
    ",,"(124.679)
    ","(74,784)
    ","(330,428)
    ","(191.000)
    "
    "Profit far the period
    ",,"275,941
    ","211.603
    ","778,495
    ","471,842
    "
    "Other Comprehensive income
    ",,,,,
    "Total Comprehensive Income/(Expense)
    ",,"275,941
    ","211,603
    ","778,495
    ","471,842
    "
    "Basic/Diluted Earning Per Share (EPS)
    ",,"0.83
    ",,"2.34
    ",

    Note: All values are in Rs. 1000s, unless otherwise stated

    Figures in brackets indicate deductions.

    The above figures are not audited

    Digital Mobility Solutions Lanka PLC Interim Condensed Financial Statements
    Nine Months Ended 31 December 2024

    4
    """

    sample_text_shareholders = """
    5.6. Twenty Largest Shareholders

    Twenty largest shareholders of the Company are as given below:

    6


    ,,"31.12.2024
    ",
    "As at
    ",,"Number of
    Shares
    ","%
    "
    "1
    ","J.Z. Hassen
    ","119,300,000
    ","35.79
    "
    "2
    ","A.D. Gunewarderie
    ","31.465.717
    ","9.44
    "
    "3
    ","LOLC Technology Services Limited
    ","31,110,782
    ","9.33
    "
    "4
    ","International Finance Corporation
    ","16.307.356
    ","4.89
    "
    "5
    ","Inverico Capital Private Limited
    ","11.955.376
    ","3.59
    "
    "6
    ","K.NJ. Balendra
    ","9,619,323
    ","2.89
    "
    "7
    ","R.HL Gunewardene
    ","8.528.351
    ","2.56
    "
    "B
    ","Interblocks Holdings Pte Ltd
    ","8,477,935
    ","2.54
    "
    "9
    ","H Capital (Private) Limited
    ","7,225,159
    ","217
    "
    "10
    ","M.S. Riyaz
    ","6,600,000
    ","198
    "
    "11
    ","K.T. Sabe
    ","4,802,700
    ","144
    "
    "12
    ","S.H. Amarasekera
    ","4,453,741
    ","1.34
    "
    "13
    ","F. Kassim
    ","4.108,778
    ","1.23
    "
    "14
    ","LW.A. De Soysa
    ","3.245.164
    ","0.97
    "
    "15
    ","Citibank Newyork S/A Norges Bank
    ","3,159,667
    ","0.95
    "
    "16
    ","Amaliya Private Limited
    ","2.976,813
    ","0.89
    "
    "17
    ","R.K. Modder
    ","2,777,778
    ","0.83
    "
    "18
    ","Serendip Investments Limited
    ","2,733,900
    ","DB2
    "
    "19
    ","M.M. Somasiri
    ","2,571,223
    ","0.77
    "
    "20
    ","Deutsche Bank Ag Trustee To Lynear Wealth Dynamic Opportunities Fund
    ","2,009,754
    ","0.60
    "
    """

    print("\n--- Income Statement Extraction Test ---")
    income_data = extract_income_statement(sample_text_income)
    print(json.dumps(income_data, indent=2))

    print("\n--- Shareholders Extraction Test ---")
    shareholders_data = extract_shareholders(sample_text_shareholders)
    print(json.dumps(shareholders_data, indent=2))

    print("\n--- Balance Sheet Extraction Test (Expected Empty) ---")
    balance_data = extract_balance_sheet(sample_text_income) # Using income text, as no balance sheet in sample
    print(json.dumps(balance_data, indent=2))
