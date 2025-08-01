import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

def get_connection():
    """Create database connection"""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def load_stock_data():
    """
    Loads historical stock price data from MySQL.
    Returns raw OHLCV data without any feature engineering.
    """
    conn = get_connection()
    query = """
        SELECT 
            c.symbol,
            c.name,
            sp.company_id,
            sp.trade_date,
            sp.open_price,
            sp.high_price,
            sp.low_price,
            sp.close_price,
            sp.volume,
            sp.turnover,
            sp.market_cap,
            sp.previous_close,
            sp.percentage_change,
            sp.change_amount
        FROM stock_prices sp
        JOIN companies c ON c.id = sp.company_id
        WHERE sp.close_price IS NOT NULL 
        AND sp.volume IS NOT NULL
        ORDER BY sp.company_id, sp.trade_date ASC
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Basic data cleaning
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['company_id', 'trade_date'])
        
        # Remove any duplicate records
        df = df.drop_duplicates(subset=['company_id', 'trade_date'], keep='last')
        
        # Basic validation
        df = df[df['close_price'] > 0]  # Remove invalid prices
        df = df[df['volume'] >= 0]      # Remove negative volumes
        
        print(f"✅ Loaded stock data: {len(df)} records for {df['company_id'].nunique()} companies")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error loading stock data: {e}")
        return pd.DataFrame()
    finally:
        if conn.is_connected():
            conn.close()

def load_financial_data():
    """
    Loads financial statements data from MySQL.
    Returns raw financial data without any calculations.
    """
    conn = get_connection()
    query = """
        SELECT 
            company_id,
            statement_type,
            period_type,
            period_start_date,
            period_end_date,
            is_audited,
            revenue,
            cost_of_goods_sold,
            gross_profit,
            operating_expenses,
            operating_income,
            net_income,
            earnings_per_share,
            interest_expense,
            tax_expense,
            cash_and_equivalents,
            accounts_receivable,
            inventory,
            current_assets,
            total_assets,
            accounts_payable,
            short_term_debt,
            current_liabilities,
            long_term_debt,
            total_liabilities,
            total_equity,
            shares_outstanding,
            operating_cash_flow,
            investing_cash_flow,
            financing_cash_flow,
            net_cash_change,
            dividends_paid
        FROM financial_statements
        WHERE period_end_date IS NOT NULL
        ORDER BY company_id, period_end_date DESC
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Basic data cleaning
        df['period_start_date'] = pd.to_datetime(df['period_start_date'])
        df['period_end_date'] = pd.to_datetime(df['period_end_date'])
        
        # Remove duplicates - keep most recent for same company/period
        df = df.drop_duplicates(subset=['company_id', 'period_end_date', 'statement_type'], keep='first')
        
        print(f"✅ Loaded financial data: {len(df)} records for {df['company_id'].nunique()} companies")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error loading financial data: {e}")
        return pd.DataFrame()
    finally:
        if conn.is_connected():
            conn.close()

def load_companies_info():
    """
    Load company information (sectors, industries, etc.)
    """
    conn = get_connection()
    query = """
        SELECT 
            id as company_id,
            symbol,
            name,
            sector,
            industry,
            market_cap_category,
            listing_date,
            is_active
        FROM companies
        WHERE is_active = 1
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        
        df['listing_date'] = pd.to_datetime(df['listing_date'])
        
        print(f"✅ Loaded company info: {len(df)} companies")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error loading company info: {e}")
        return pd.DataFrame()
    finally:
        if conn.is_connected():
            conn.close()

def validate_data_quality(stock_df, financial_df=None):
    """
    Perform basic data quality checks
    """
    issues = []
    
    # Stock data validation
    if stock_df.empty:
        issues.append("Stock data is empty")
        return issues
    
    # Check for missing critical columns
    required_stock_cols = ['company_id', 'trade_date', 'close_price', 'volume']
    missing_cols = [col for col in required_stock_cols if col not in stock_df.columns]
    if missing_cols:
        issues.append(f"Missing required stock columns: {missing_cols}")
    
    # Check for companies with insufficient data
    company_counts = stock_df.groupby('company_id').size()
    insufficient_data = company_counts[company_counts < 30]  # Less than 30 days
    if not insufficient_data.empty:
        issues.append(f"{len(insufficient_data)} companies have less than 30 days of data")
    
    # Check for date gaps
    for company_id in stock_df['company_id'].unique()[:5]:  # Check first 5 companies
        company_data = stock_df[stock_df['company_id'] == company_id].sort_values('trade_date')
        date_diffs = company_data['trade_date'].diff().dt.days
        large_gaps = date_diffs[date_diffs > 7]  # More than 7 days gap
        if not large_gaps.empty:
            issues.append(f"Company {company_id} has {len(large_gaps)} date gaps > 7 days")
    
    # Financial data validation
    if financial_df is not None and not financial_df.empty:
        # Check for companies with no financial data
        stock_companies = set(stock_df['company_id'].unique())
        financial_companies = set(financial_df['company_id'].unique())
        no_financial = stock_companies - financial_companies
        if no_financial:
            issues.append(f"{len(no_financial)} companies have no financial data")
    
    # Print summary
    if issues:
        print("⚠️ Data Quality Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data quality checks passed")
    
    return issues

def get_data_summary(stock_df, financial_df=None):
    """
    Generate a summary of loaded data
    """
    summary = {
        'stock_data': {
            'total_records': len(stock_df),
            'companies': stock_df['company_id'].nunique() if not stock_df.empty else 0,
            'date_range': {
                'start': stock_df['trade_date'].min() if not stock_df.empty else None,
                'end': stock_df['trade_date'].max() if not stock_df.empty else None
            },
            'avg_records_per_company': len(stock_df) // stock_df['company_id'].nunique() if not stock_df.empty else 0
        }
    }
    
    if financial_df is not None and not financial_df.empty:
        summary['financial_data'] = {
            'total_records': len(financial_df),
            'companies': financial_df['company_id'].nunique(),
            'statement_types': financial_df['statement_type'].unique().tolist(),
            'period_types': financial_df['period_type'].unique().tolist(),
            'date_range': {
                'start': financial_df['period_end_date'].min(),
                'end': financial_df['period_end_date'].max()
            }
        }
    
    return summary

def prepare_features_labels(stock_df, financial_df=None, include_financials=True):
    """
    Wrapper function to maintain compatibility with existing training scripts.
    This imports and uses the feature engineering module.
    """
    try:
        from data.feature_engineering import FeatureEngineer
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Create features and labels
        X, y, metadata = engineer.create_features_and_labels(
            stock_df, 
            financial_df if include_financials else None
        )
        
        # Return only X and y for compatibility
        return X, y
        
    except ImportError:
        print("❌ Feature engineering module not found. Please ensure feature_engineering.py exists.")
        return pd.DataFrame(), pd.Series()
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return pd.DataFrame(), pd.Series()

def main():
    """
    Main function for testing data loading
    """
    print("Loading data...")
    
    # Load all data
    stock_data = load_stock_data()
    financial_data = load_financial_data()
    company_info = load_companies_info()
    
    # Validate data quality
    validate_data_quality(stock_data, financial_data)
    
    # Print summary
    summary = get_data_summary(stock_data, financial_data)
    print(f"\n📊 Data Summary:")
    print(f"Stock Data: {summary['stock_data']}")
    if 'financial_data' in summary:
        print(f"Financial Data: {summary['financial_data']}")
    
    return stock_data, financial_data, company_info

if __name__ == "__main__":
    stock_df, financial_df, company_df = main()