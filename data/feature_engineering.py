import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple, List
import warnings
import logging # Added for better internal logging
warnings.filterwarnings('ignore')

# Configure logging for better visibility within FeatureEngineer
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class FeatureEngineer:
    """
    Comprehensive feature engineering class for stock market data.
    Handles both technical and fundamental analysis features.
    """
    
    def __init__(self):
        self.technical_features = []
        self.fundamental_features = []
        self.all_features = [] # This will store the final list of features used

    def merge_financial_with_stock_data(self, stock_df: pd.DataFrame, financial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge financial data with stock data using nearest date matching.
        """
        if financial_df is None or financial_df.empty:
            logging.info("No financial data provided for merging or financial_df is empty.")
            return stock_df.copy()
            
        merged_data = []
        
        # Ensure 'period_end_date' is datetime in financial_df for proper comparison
        financial_df['period_end_date'] = pd.to_datetime(financial_df['period_end_date'])
        
        # Ensure financial_df columns are strings to prevent issues if column names are integers etc.
        financial_df.columns = financial_df.columns.astype(str)
        
        # Ensure company_id in financial_df is consistent string type
        financial_df['company_id'] = financial_df['company_id'].astype(str)

        # Sort financial data by company and date, so .iloc[0] gets the latest before a trade_date
        financial_df_sorted = financial_df.sort_values(by=['company_id', 'period_end_date'], ascending=[True, False])
        
        # Process each company separately
        for company_id_val in stock_df['company_id'].unique():
            # Ensure the company_id from stock_df is also a string for consistent comparison
            company_id_val_str = str(company_id_val) 
            
            company_stock = stock_df[stock_df['company_id'] == company_id_val].copy()
            # Use the string-converted company_id for filtering
            company_financial = financial_df_sorted[financial_df_sorted['company_id'] == company_id_val_str]
            
            if company_financial.empty:
                merged_data.append(company_stock)
                continue
            
            # Process each stock row individually instead of using apply
            for idx, stock_row in company_stock.iterrows():
                stock_trade_date = stock_row['trade_date']
                
                # Find the most recent financial data before or on the trade date
                relevant_financials = company_financial[
                    company_financial['period_end_date'] <= stock_trade_date
                ]
                
                if not relevant_financials.empty:
                    latest_financial_row = relevant_financials.iloc[0]
                    
                    # Add financial data to the stock row and record as fundamental features
                    for col_name in financial_df.columns:
                        if col_name not in ['company_id', 'period_end_date', 'statement_type']: # Exclude statement_type too
                            merged_col_name = f'financial_{col_name}'
                            if col_name in latest_financial_row.index:
                                company_stock.at[idx, merged_col_name] = latest_financial_row[col_name]
                                if merged_col_name not in self.fundamental_features:
                                    self.fundamental_features.append(merged_col_name)
                            else:
                                company_stock.at[idx, merged_col_name] = np.nan
                else:
                    # No financial data available for this date, fill with NaN for all financial_ columns
                    for col_name in financial_df.columns:
                        if col_name not in ['company_id', 'period_end_date', 'statement_type']:
                            merged_col_name = f'financial_{col_name}'
                            company_stock.at[idx, merged_col_name] = np.nan
                            if merged_col_name not in self.fundamental_features:
                                self.fundamental_features.append(merged_col_name)
            
            merged_data.append(company_stock)
        
        result = pd.concat(merged_data, ignore_index=True) if merged_data else stock_df.copy()
        logging.info(f"✅ Merged data shape: {result.shape}")
        return result

    # Technical Analysis Features (Leave these as is, they extend self.technical_features)
    def add_moving_averages(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Add multiple moving averages"""
        for w in windows:
            col_name = f'MA_{w}'
            df[col_name] = df.groupby('company_id')['close_price'].transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )
            if col_name not in self.technical_features:
                self.technical_features.append(col_name)
        return df

    def add_exponential_moving_averages(self, df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """Add exponential moving averages"""
        for w in windows:
            col_name = f'EMA_{w}'
            df[col_name] = df.groupby('company_id')['close_price'].transform(
                lambda x: x.ewm(span=w, adjust=False, min_periods=1).mean() # Added adjust=False for classic EMA
            )
            if col_name not in self.technical_features:
                self.technical_features.append(col_name)
        return df

    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        def safe_rsi(series):
            # Ensure proper handling for small series
            if len(series) < window:
                return pd.Series([np.nan] * len(series), index=series.index)
            try:
                return ta.momentum.rsi(series, window=window, fillna=False)
            except Exception as e:
                logging.warning(f"RSI calculation failed for a series. Error: {e}")
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss.replace(0, np.nan) 
                rsi = 100 - (100 / (1 + rs))
                return rsi
        
        col_name = 'RSI'
        df[col_name] = df.groupby('company_id')['close_price'].transform(safe_rsi)
        if col_name not in self.technical_features:
            self.technical_features.append(col_name)
        return df

    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicators"""
        # Ensure 'close_price' is always present
        if 'close_price' not in df.columns:
            logging.warning("Missing 'close_price' for MACD calculation. Skipping.")
            return df

        df['MACD'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.trend.macd(x, window_fast=fast, window_slow=slow, fillna=False)
        )
        df['MACD_Signal'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.trend.macd_signal(x, window_fast=fast, window_slow=slow, window_sign=signal, fillna=False)
        )
        df['MACD_Histogram'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.trend.macd_diff(x, window_fast=fast, window_slow=slow, window_sign=signal, fillna=False)
        )
        
        macd_features = ['MACD', 'MACD_Signal', 'MACD_Histogram']
        for feat in macd_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        if 'close_price' not in df.columns:
            logging.warning("Missing 'close_price' for Bollinger Bands calculation. Skipping.")
            return df

        df['BB_Upper'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.volatility.bollinger_hband(x, window=window, window_dev=std_dev, fillna=False)
        )
        df['BB_Middle'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.volatility.bollinger_mavg(x, window=window, fillna=False)
        )
        df['BB_Lower'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.volatility.bollinger_lband(x, window=window, window_dev=std_dev, fillna=False)
        )
        df['BB_Width'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.volatility.bollinger_wband(x, window=window, window_dev=std_dev, fillna=False)
        )
        df['BB_Percent'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.volatility.bollinger_pband(x, window=window, window_dev=std_dev, fillna=False)
        )
        
        bb_features = ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent']
        for feat in bb_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        
        return df

    def add_stochastic_oscillator(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        if not all(col in df.columns for col in ['high_price', 'low_price', 'close_price']):
            logging.warning("Missing high_price, low_price or close_price for Stochastic Oscillator. Skipping.")
            df['Stoch_K'] = np.nan
            df['Stoch_D'] = np.nan
            return df 

        stoch_k_results = []
        stoch_d_results = []

        for company_id, group_df in df.groupby('company_id'):
            k_series = ta.momentum.stoch(
                high=group_df['high_price'],
                low=group_df['low_price'],
                close=group_df['close_price'],
                window=k_window,
                fillna=False
            )
            d_series = ta.momentum.stoch_signal(
                high=group_df['high_price'],
                low=group_df['low_price'],
                close=group_df['close_price'],
                window=k_window,
                smooth_window=d_window,
                fillna=False
            )
            stoch_k_results.append(k_series)
            stoch_d_results.append(d_series)

        df['Stoch_K'] = pd.concat(stoch_k_results).reindex(df.index)
        df['Stoch_D'] = pd.concat(stoch_d_results).reindex(df.index)
        
        stoch_features = ['Stoch_K', 'Stoch_D']
        for feat in stoch_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        
        return df

    def _calculate_pvt(self, close_prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Manually calculate Price-Volume Trend (PVT) for a single series, matching ta.volume.pvt(fillna=False) behavior."""
        close_prices_numeric = pd.to_numeric(close_prices, errors='coerce')
        volumes_numeric = pd.to_numeric(volumes, errors='coerce')

        temp_df = pd.DataFrame({'close': close_prices_numeric, 'volume': volumes_numeric}, index=close_prices.index)
        temp_df = temp_df.dropna()

        if len(temp_df) < 2:
            return pd.Series(np.nan, index=close_prices.index)

        pct_change = temp_df['close'].pct_change()
        pvt_contribution = pct_change * temp_df['volume']
        pvt_series_calculated = pvt_contribution.cumsum()

        final_pvt = pd.Series(np.nan, index=close_prices.index, dtype=float)
        final_pvt.loc[pvt_series_calculated.index] = pvt_series_calculated

        return final_pvt

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        # Ensure necessary columns are present for volume indicators
        if not all(col in df.columns for col in ['close_price', 'volume', 'high_price', 'low_price']):
            logging.warning("Missing required price/volume columns for some volume indicators. Skipping affected indicators.")
            # Ensure columns exist, even if NaN
            for col in ['volume_change_pct', 'volume_MA_20', 'volume_ratio', 'PVT', 'OBV', 'VWAP']:
                if col not in df.columns:
                    df[col] = np.nan
            return df

        df['volume_change_pct'] = df.groupby('company_id')['volume'].pct_change().fillna(0)
        df['volume_MA_20'] = df.groupby('company_id')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        ).fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume_MA_20']).replace([np.inf, -np.inf], 1).fillna(1)

        pvt_results = []
        for company_id, group_df in df.groupby('company_id'):
            try:
                pvt_series = ta.volume.pvt(
                    close=group_df['close_price'],
                    volume=group_df['volume'],
                    fillna=False
                )
            except Exception as e:
                logging.warning(f"Error calculating PVT for {company_id}: {e}. Falling back to manual calculation.")
                pvt_series = self._calculate_pvt(
                    close_prices=group_df['close_price'],
                    volumes=group_df['volume']
                )
            pvt_results.append(pvt_series)
        df['PVT'] = pd.concat(pvt_results).reindex(df.index)

        # On-Balance Volume (OBV)
        df['OBV'] = df.groupby('company_id').apply(
            lambda x: ta.volume.on_balance_volume(x['close_price'], x['volume'], fillna=False)
        ).droplevel(0) # droplevel because apply adds an extra level
        
        # Volume Weighted Average Price (VWAP) - requires group apply as it needs high, low, close, volume for each date
        # It's an aggregate function typically, but can be done rolling. ta.volume.vwap requires a 'period'
        # Let's add it as a simple daily VWAP, or if a rolling VWAP is desired, that needs more logic.
        # For simplicity, if your intent was a single-day VWAP, ensure all columns are there.
        # If your intent was a rolling VWAP from `ta.volume`, it's not a direct 'transform' per company.
        # I'll add a simple daily VWAP here as a feature, assuming the ta.volume.vwap function is intended for rolling.
        # As your previous code called `ta.volume.volume_weighted_average_price`, let's stick to that if it implies a daily calculation without rolling.
        # If ta.volume.volume_weighted_average_price means daily, then:
        df['VWAP'] = (df['volume'] * (df['high_price'] + df['low_price'] + df['close_price']) / 3).fillna(0) / df['volume'].fillna(1)
        # This is a common simplified daily VWAP. If you meant a different VWAP, clarify.

        volume_features = ['volume_change_pct', 'volume_MA_20', 'volume_ratio', 'PVT', 'OBV', 'VWAP']
        for feat in volume_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # Price volatility (Rolling Standard Deviation of returns)
        df['price_volatility'] = df.groupby('company_id')['close_price'].transform(
            lambda x: x.pct_change().rolling(window=20, min_periods=1).std()
        ).fillna(0)
        
        # Average True Range
        if all(col in df.columns for col in ['high_price', 'low_price', 'close_price']):
            atr_results = []
            atr_window = 14
            for company_id, group_df in df.groupby('company_id'):
                if len(group_df) >= atr_window:
                    atr_series = ta.volatility.average_true_range(
                        high=group_df['high_price'],
                        low=group_df['low_price'],
                        close=group_df['close_price'],
                        window=atr_window,
                        fillna=False
                    )
                    atr_results.append(atr_series)
                else:
                    logging.warning(f"Not enough data for ATR calculation for company_id {company_id}. Expected at least {atr_window} rows, got {len(group_df)}. Filling with NaN.")
                    atr_results.append(pd.Series(np.nan, index=group_df.index))
            
            df['ATR'] = pd.concat(atr_results).reindex(df.index)
        else:
            df['ATR'] = np.nan
            logging.warning("Missing high_price, low_price or close_price for ATR calculation. Skipping.")
        
        volatility_features = ['price_volatility', 'ATR']
        for feat in volatility_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        if 'close_price' not in df.columns:
            logging.warning("Missing 'close_price' for momentum indicators. Skipping.")
            for col in ['ROC_10', 'momentum_5', 'price_acceleration']:
                if col not in df.columns:
                    df[col] = np.nan
            return df

        df['ROC_10'] = df.groupby('company_id')['close_price'].transform(
            lambda x: ta.momentum.roc(x, window=10, fillna=False)
        )
        
        df['momentum_5'] = df.groupby('company_id')['close_price'].transform(
            lambda x: x.diff(periods=5)
        )
        
        df['price_acceleration'] = df.groupby('company_id')['close_price'].transform(
            lambda x: x.pct_change().diff()
        )
        
        momentum_features = ['ROC_10', 'momentum_5', 'price_acceleration']
        for feat in momentum_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features"""
        if 'close_price' not in df.columns:
            logging.warning("Missing 'close_price' for price action features. Skipping.")
            return df # Cannot compute most features without close_price

        df['daily_return'] = df.groupby('company_id')['close_price'].pct_change()
        
        df['price_vs_20day_high'] = (df['close_price'] / 
            df.groupby('company_id')['close_price'].transform(
                lambda x: x.rolling(window=20, min_periods=1).max()
            ))
        
        df['price_vs_20day_low'] = (df['close_price'] / 
            df.groupby('company_id')['close_price'].transform(
                lambda x: x.rolling(window=20, min_periods=1).min()
            ))
        
        df['volatility_rank'] = df.groupby('company_id')['daily_return'].transform(
            lambda x: x.abs().rolling(window=20, min_periods=1).rank(pct=True)
        )
        
        price_action_features = ['daily_return', 'price_vs_20day_high', 'price_vs_20day_low', 'volatility_rank']
        
        if all(col in df.columns for col in ['high_price', 'low_price', 'open_price', 'close_price']):
            df['hl_spread'] = ((df['high_price'] - df['low_price']) / df['close_price'])
            df['intraday_momentum'] = ((df['close_price'] - df['open_price']) / df['open_price'])
            price_action_features.extend(['hl_spread', 'intraday_momentum'])
            
            # For opening_gap, ensure 'previous_close' is available from shifted 'close_price'
            df['previous_close'] = df.groupby('company_id')['close_price'].shift(1)
            df['opening_gap'] = ((df['open_price'] - df['previous_close']) / df['previous_close'])
            price_action_features.append('opening_gap')
        
        for feat in price_action_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    # Fundamental Analysis Features
    def add_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental financial ratios"""
        # Ensure required financial columns exist (prefixed with 'financial_')
        # If they don't, the ratio columns will be created as NaN and filled later.
        required_base_financial_cols = [
            'market_cap', 'net_income', 'total_equity', 'long_term_debt',
            'short_term_debt', 'total_assets', 'current_assets',
            'current_liabilities', 'cash_and_equivalents', 'inventory',
            'gross_profit', 'revenue', 'operating_income',
            'cost_of_goods_sold', 'operating_cash_flow', 
            'investing_cash_flow' 
        ]
        
        # Ensure these columns exist, possibly as NaN if not merged
        for col in required_base_financial_cols:
            if f'financial_{col}' not in df.columns:
                df[f'financial_{col}'] = np.nan 

        # Valuation ratios
        # Use .fillna(0) for denominator to avoid division by zero and inf results, before replace(0,np.nan) for ratio.
        # Correcting logic: PE_ratio should use market_cap and financial_net_income
        df['PE_ratio'] = df['market_cap'] / (df['financial_net_income'].replace(0, np.nan)) # Removed *4 as it's typically annual
        df['PB_ratio'] = df['market_cap'] / df['financial_total_equity'].replace(0, np.nan)
        
        # Debt ratios
        total_debt = (df['financial_long_term_debt'].fillna(0) + df['financial_short_term_debt'].fillna(0))
        df['debt_to_equity'] = total_debt / df['financial_total_equity'].replace(0, np.nan)
        df['debt_to_assets'] = total_debt / df['financial_total_assets'].replace(0, np.nan)
        
        # Liquidity ratios
        df['current_ratio'] = df['financial_current_assets'] / df['financial_current_liabilities'].replace(0, np.nan)
        df['cash_ratio'] = df['financial_cash_and_equivalents'] / df['financial_current_liabilities'].replace(0, np.nan)
        df['quick_ratio'] = (df['financial_current_assets'] - df['financial_inventory'].fillna(0)) / df['financial_current_liabilities'].replace(0, np.nan)
        
        # Profitability ratios (assuming these are annual figures from financial_df)
        df['ROA'] = df['financial_net_income'] / df['financial_total_assets'].replace(0, np.nan) 
        df['ROE'] = df['financial_net_income'] / df['financial_total_equity'].replace(0, np.nan) 
        df['gross_margin'] = df['financial_gross_profit'] / df['financial_revenue'].replace(0, np.nan)
        df['operating_margin'] = df['financial_operating_income'] / df['financial_revenue'].replace(0, np.nan)
        df['net_margin'] = df['financial_net_income'] / df['financial_revenue'].replace(0, np.nan)
        
        # Efficiency ratios (assuming these are annual figures from financial_df)
        df['asset_turnover'] = df['financial_revenue'] / df['financial_total_assets'].replace(0, np.nan) 
        df['inventory_turnover'] = df['financial_cost_of_goods_sold'] / df['financial_inventory'].replace(0, np.nan) 
        
        # Cash flow ratios
        df['operating_cash_margin'] = df['financial_operating_cash_flow'] / df['financial_revenue'].replace(0, np.nan)
        df['free_cash_flow'] = df['financial_operating_cash_flow'].fillna(0) - df['financial_investing_cash_flow'].fillna(0)
        df['fcf_margin'] = df['free_cash_flow'] / df['financial_revenue'].replace(0, np.nan)
        
        fundamental_ratios = [
            'PE_ratio', 'PB_ratio', 'debt_to_equity', 'debt_to_assets', 'current_ratio', 
            'cash_ratio', 'quick_ratio', 'ROA', 'ROE', 'gross_margin', 'operating_margin', 
            'net_margin', 'asset_turnover', 'inventory_turnover', 'operating_cash_margin', 
            'free_cash_flow', 'fcf_margin'
        ]
        
        for col in fundamental_ratios:
            if col in df.columns and col not in self.fundamental_features: 
                self.fundamental_features.append(col)
                
        return df

    def add_growth_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add growth metrics"""
        # Ensure base cols exist for growth calculations
        base_growth_cols = ['financial_revenue', 'financial_net_income', 'financial_total_assets']
        # financial_earnings_per_share might not be directly available, better to calculate EPS as a ratio
        
        for col_prefix in base_growth_cols:
            if col_prefix not in df.columns: # Check for the financial_ prefix
                df[col_prefix] = np.nan 

        growth_features = []
        for col_prefix in base_growth_cols:
            # Need historical financial data for true growth rates, which is not directly from merge_financial_with_stock_data
            # if financial_df only provides latest. For this, we'd need to shift.
            # Assuming financial data is present for multiple periods per company due to merge strategy:
            if col_prefix in df.columns: 
                growth_col = f'{col_prefix}_growth'
                # Group by company_id and calculate percentage change based on previous financial period
                df[growth_col] = df.groupby('company_id')[col_prefix].pct_change() 
                growth_features.append(growth_col)
        
        # Also add EPS growth if 'EPS' was calculated in add_financial_ratios
        if 'PE_ratio' in df.columns: # Assuming PE_ratio implies EPS was calculated
            # EPS needs to be calculated first, then its growth
            if 'financial_net_income' in df.columns and 'financial_shares_outstanding' in df.columns:
                df['EPS'] = df['financial_net_income'] / df['financial_shares_outstanding'].replace(0, np.nan)
                eps_growth_col = 'EPS_growth'
                df[eps_growth_col] = df.groupby('company_id')['EPS'].pct_change()
                growth_features.append(eps_growth_col)
                if 'EPS' not in self.fundamental_features: self.fundamental_features.append('EPS')

        for col in growth_features:
            if col not in self.fundamental_features: 
                self.fundamental_features.append(col)
        
        return df

    def clean_and_validate_features(self, df: pd.DataFrame, training_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean and validate all engineered features.
        
        Args:
            df (pd.DataFrame): The DataFrame with engineered features.
            training_features (Optional[List[str]]): A list of feature names from the training set.
                                                     If provided (in prediction mode), ensures the output DataFrame
                                                     has these exact columns in the correct order, filling missing
                                                     ones and dropping extra ones.
        """
        logging.info("🧹 Cleaning and validating features...")
        initial_shape = df.shape

        # Identify all columns that should *not* be treated as features for imputation/clipping
        # These are usually metadata, IDs, or target labels
        non_feature_cols = [
            'trade_date', 'company_id', 'symbol', 'next_close_price', 
            'price_diff', 'label', 'previous_close', 'period_end_date', 'statement_type',
            'close_price', 'open_price', 'high_price', 'low_price', 'volume', 'market_cap' # Base columns, kept for calculations, but usually not treated as features for imputation in the same way as derived features.
        ]
        
        # Columns that MUST be numeric and should be filled carefully if NaNs exist
        critical_numeric_for_fill = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 'market_cap']
        
        # First, ensure base numeric columns are actually numeric and handle NaNs critically
        for col in critical_numeric_for_fill:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # For critical price/volume data, if NaN, drop the row or fill with 0 if it makes sense in your context
                # For a prediction scenario, dropping might be too aggressive if it means losing all data for a company.
                # For now, let's fill with 0 if it's NaN after coercion. You might want a more sophisticated strategy.
                if df[col].isnull().any():
                    logging.warning(f"NaNs found in critical column '{col}'. Filling with 0.")
                    df[col].fillna(0, inplace=True)
        
        # Apply cleaning for ALL current and potential technical/fundamental features
        all_potential_feature_cols = list(set(self.technical_features + self.fundamental_features))
        
        for col in all_potential_feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Clipping outliers
                if df[col].std(skipna=True) > 0 and df[col].count() > 1: 
                    mean_val = df[col].mean(skipna=True)
                    std_val = df[col].std(skipna=True)
                    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                        lower_bound = mean_val - 5 * std_val
                        upper_bound = mean_val + 5 * std_val
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Impute NaNs with median if available, otherwise 0
                median_val = df[col].median(skipna=True)
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
            else:
                # If a feature from self.technical_features/self.fundamental_features was expected but not created,
                # this is where we'd add it as NaN for consistency IF training_features is None.
                # If training_features IS provided, it will handle adding missing columns.
                pass # Handled by the training_features alignment below

        # --- Feature Alignment for Prediction/Consistent Output ---
        if training_features: # This block is active when `training_features` are provided (e.g., during prediction)
            logging.info(f"Aligning features to {len(training_features)} training features...")
            
            # Add any missing training features to df and fill with 0
            missing_cols = set(training_features) - set(df.columns)
            for col in missing_cols:
                df[col] = 0.0 # Use 0.0 for numerical consistency

            # Drop any columns in df that are NOT in training_features
            extra_cols = set(df.columns) - set(training_features)
            # Ensure we don't accidentally drop critical base columns or metadata
            extra_cols = [col for col in extra_cols if col not in non_feature_cols]
            df.drop(columns=list(extra_cols), errors='ignore', inplace=True)

            # Ensure the order of columns matches training_features
            df = df[training_features].copy()
            logging.info(f"Features aligned. Current columns: {df.columns.tolist()}")

        logging.info(f"Cleaned and validated features. Shape changed from {initial_shape} to {df.shape}")
        return df

    def create_labels(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target labels for prediction:
        1 (Buy) if price increases by threshold,
        -1 (Sell) if decreases by threshold,
        0 (Hold) otherwise.
        """
        logging.info("🎯 Creating target labels...")
        # Ensure data is sorted for correct shifting within groups
        df = df.sort_values(by=['company_id', 'trade_date'])

        # Calculate next day's close price for each company
        df['next_close_price'] = df.groupby('company_id')['close_price'].shift(-1)
        df['price_diff'] = df['next_close_price'] - df['close_price']
        df['price_change_percent'] = df['price_diff'] / df['close_price']

        def apply_label(row, threshold_val):
            if pd.isna(row['price_change_percent']) or row['close_price'] == 0:
                return np.nan # Cannot create label if next_close_price is missing or current price is zero
            elif row['price_change_percent'] >= threshold_val:
                return 1.0  # Buy
            elif row['price_change_percent'] <= -threshold_val:
                return -1.0 # Sell
            else:
                return 0.0  # Hold

        df['label'] = df.apply(apply_label, axis=1, threshold_val=threshold)
        logging.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        return df

    def create_features_and_labels(self, stock_df: pd.DataFrame, financial_df: Optional[pd.DataFrame] = None, 
                                   is_prediction: bool = False, training_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Main method to create all features and labels
        """
        
        self.technical_features = []
        self.fundamental_features = []
        self.all_features = [] # Reset this for each call
        
        logging.info("🔄 Starting feature engineering...")
        
        if 'trade_date' not in stock_df.columns:
            raise ValueError("stock_df must contain a 'trade_date' column.")
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        stock_df = stock_df.sort_values(by=['company_id', 'trade_date']).reset_index(drop=True)

        # Preserve the original index for metadata mapping later
        stock_df['original_index'] = stock_df.index

        # Merge financial data first if provided
        if financial_df is not None and not financial_df.empty:
            logging.info("💰 Merging financial data with stock data...")
            df = self.merge_financial_with_stock_data(stock_df, financial_df)
        else:
            df = stock_df.copy()
            logging.info("ℹ️ No financial data provided, using technical indicators only")
        
        df = df.sort_values(['company_id', 'trade_date'])
        
        # Ensure base price/volume/market_cap columns are numeric right after initial load/merge
        base_numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'market_cap']
        for col in base_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info("📈 Adding technical indicators...")
        df = self.add_moving_averages(df)
        df = self.add_exponential_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_stochastic_oscillator(df)
        df = self.add_volume_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_price_action_features(df)
        
        # Check for financial data and add fundamental indicators
        if 'financial_revenue' in df.columns and not df['financial_revenue'].isna().all():
            logging.info("💰 Adding fundamental indicators...")
            df = self.add_financial_ratios(df)
            df = self.add_growth_metrics(df)
        else:
            logging.warning("⚠️ Financial data not available or merged successfully for fundamental indicators (financial_revenue not found or all NaN).")

        # --- IMPORTANT NEW LOGIC: Preserve full metadata BEFORE clean_and_validate_features potentially strips columns ---
        # This is the key change to work around your existing clean_and_validate_features behavior.
        metadata_cols_to_preserve = ['company_id', 'trade_date', 'symbol', 'close_price', 'original_index',
                                     'open_price', 'high_price', 'low_price', 'volume', 'market_cap'] # Include all base stock data for metadata if needed
        
        # Ensure only columns actually present in df are considered for initial_metadata
        available_initial_metadata_cols = [col for col in metadata_cols_to_preserve if col in df.columns]
        initial_metadata = df[available_initial_metadata_cols].copy()
        
        logging.info("🧹 Cleaning and validating features...")
        # Your clean_and_validate_features will still get `df` (with all columns) and `training_features`.
        # It will likely return a `df_processed` that only contains the 92 `training_features` columns.
        df_processed = self.clean_and_validate_features(df, training_features=training_features)
        
        # --- Metadata extraction and final X, y preparation ---
        # Filter the initial_metadata based on the indices that remain in df_processed
        # This ensures metadata perfectly aligns with the rows in X.
        metadata = initial_metadata.loc[df_processed.index].copy()
        
        if not is_prediction:
            logging.info("🎯 Creating target labels for training...")
            df_labeled = self.create_labels(df_processed) 
            
            # Filter metadata based on valid labels from df_labeled
            valid_label_indices = df_labeled['label'].dropna().index
            df_final_for_training = df_labeled.loc[valid_label_indices].copy()
            metadata = metadata.loc[valid_label_indices].copy() # Filter metadata here
            
            # Dynamically determine which features are actually present in the DataFrame for training
            # This logic sets self.all_features for use outside the class (e.g., saving for prediction)
            all_potential_features_from_lists = list(set(self.technical_features + self.fundamental_features))
            
            # --- CRITICAL CHANGE FOR self.all_features definition ---
            # Exclude ALL base stock price/volume/market_cap columns from self.all_features,
            # as these are considered "metadata" or base data, not derived features for the model.
            # Your current 'base_stock_features_to_include' should only be used to ensure they are numeric.
            # They should not implicitly be added to the final 'all_features' list.
            excluded_from_model_features = ['trade_date', 'company_id', 'symbol', 'next_close_price', 
                                            'price_diff', 'price_change_percent', 'label', 'previous_close',
                                            'original_index', # These are the previous exclusions
                                            'open_price', 'high_price', 'low_price', 'close_price', 
                                            'volume', 'market_cap'] # <--- NEW EXCLUSIONS for model features
            
            # Combine all features, ensuring they exist in the dataframe and are not excluded
            self.all_features = [
                col for col in (all_potential_features_from_lists) # Only use derived features
                if col in df_final_for_training.columns and col not in excluded_from_model_features
            ]
            self.all_features.sort() # Keep features sorted for consistency

            # X for training: select only the actual features
            X = df_final_for_training[self.all_features].copy()
            y = df_final_for_training['label'].copy()
            
            logging.info(f"✅ Feature engineering complete for training!")
            logging.info(f"   📊 Technical features: {len(self.technical_features)}")
            logging.info(f"   💼 Fundamental features: {len(self.fundamental_features)}")
            logging.info(f"   🔢 Total features (dynamically selected): {len(self.all_features)}")
            logging.info(f"📋 Final dataset summary for training:")
            logging.info(f"   🔢 Samples: {len(X)}")
            logging.info(f"   📊 Features: {len(X.columns)}")
            logging.info(f"   🏷️ Label distribution: {y.value_counts().sort_index().to_dict()}")

        else: # If is_prediction is True
            logging.info("ℹ️ Skipping target label creation for prediction.")
            y = pd.Series([], dtype=float) # Return an empty Series for y in prediction mode

            if training_features is None:
                raise ValueError("training_features must be provided when is_prediction is True for prediction mode.")
            
            # X for prediction: df_processed already has columns aligned to training_features by your clean_and_validate_features
            X = df_processed[training_features].copy() 

            # The metadata DataFrame was already filtered based on df_processed's index earlier
            # So, metadata is already aligned with X.

            logging.info(f"✅ Feature engineering complete for prediction!")
            logging.info(f"   🔢 Samples for prediction: {len(X)}")
            logging.info(f"   📊 Features: {len(X.columns)}")
        
        # Return results, ensuring indices are reset for all three outputs
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        
        return X, y, metadata
    
    def get_feature_importance_groups(self) -> dict:
        """Return features grouped by category for analysis"""
        
        feature_groups = {
            'Moving Averages': [f for f in self.technical_features if 'MA_' in f or 'EMA_' in f],
            'Momentum': [f for f in self.technical_features if any(x in f for x in ['RSI', 'MACD', 'ROC', 'momentum', 'Stoch'])],
            'Volatility': [f for f in self.technical_features if any(x in f for x in ['BB_', 'ATR', 'volatility'])],
            'Volume': [f for f in self.technical_features if 'volume' in f or 'PVT' in f],
            'Price Action': [f for f in self.technical_features if any(x in f for x in ['daily_return', 'price_vs', 'hl_spread', 'gap', 'intraday'])],
            'Valuation': [f for f in self.fundamental_features if any(x in f for x in ['PE_', 'PB_', 'ratio'])],
            'Profitability': [f for f in self.fundamental_features if any(x in f for x in ['ROA', 'ROE', 'margin'])],
            'Liquidity': [f for f in self.fundamental_features if any(x in f for x in ['current_', 'cash_', 'quick_'])],
            'Growth': [f for f in self.fundamental_features if 'growth' in f],
            'Debt': [f for f in self.fundamental_features if 'debt' in f]
        }
        
        return {k: v for k, v in feature_groups.items() if v} 

    def save_feature_info(self, filepath: str = 'feature_info.txt'):
        """Save information about created features"""
        
        feature_groups = self.get_feature_importance_groups()
        
        with open(filepath, 'w') as f:
            f.write("STOCK PREDICTION FEATURES SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Features: {len(self.all_features)}\n")
            f.write(f"Technical Features: {len(self.technical_features)}\n")
            f.write(f"Fundamental Features: {len(self.fundamental_features)}\n\n")
            
            for group_name, features in feature_groups.items():
                f.write(f"{group_name} ({len(features)} features):\n")
                for feature in features:
                    f.write(f"  - {feature}\n")
                f.write("\n")
            
            f.write("ALL FEATURES LIST (Dynamically Selected for the Last Run):\n")
            f.write("-" * 20 + "\n")
            for i, feature in enumerate(self.all_features, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        print(f"💾 Feature information saved to: {filepath}")

# Convenience function for backward compatibility
def prepare_features_labels(stock_df: pd.DataFrame, financial_df: Optional[pd.DataFrame] = None, include_financials: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Backward compatibility function for existing training scripts
    """
    engineer = FeatureEngineer()
    
    financial_data = financial_df if include_financials else None
    X, y, metadata = engineer.create_features_and_labels(stock_df, financial_data)
    
    return X, y

# Main function for testing
def main():
    """Test the feature engineering module"""
    
    print("This is a standalone feature engineering module.")
    print("Import this module in your training script to use the FeatureEngineer class.")
    print("\nExample usage:")
    print("from data.feature_engineering import FeatureEngineer")
    print("engineer = FeatureEngineer()")
    print("X, y, metadata = engineer.create_features_and_labels(stock_df, financial_df)")

if __name__ == "__main__":
    main()