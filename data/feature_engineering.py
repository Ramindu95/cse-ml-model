import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple, List
import warnings
import logging 
warnings.filterwarnings('ignore')

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
        
        financial_df['period_end_date'] = pd.to_datetime(financial_df['period_end_date'])
        financial_df.columns = financial_df.columns.astype(str)
        financial_df['company_id'] = financial_df['company_id'].astype(str)

        financial_df_sorted = financial_df.sort_values(by=['company_id', 'period_end_date'], ascending=[True, False])
        
        for company_id_val in stock_df['company_id'].unique():
            company_id_val_str = str(company_id_val) 
            
            company_stock = stock_df[stock_df['company_id'] == company_id_val].copy()
            company_financial = financial_df_sorted[financial_df_sorted['company_id'] == company_id_val_str]
            
            if company_financial.empty:
                merged_data.append(company_stock)
                continue
            
            for idx, stock_row in company_stock.iterrows():
                stock_trade_date = stock_row['trade_date']
                
                relevant_financials = company_financial[
                    company_financial['period_end_date'] <= stock_trade_date
                ]
                
                if not relevant_financials.empty:
                    latest_financial_row = relevant_financials.iloc[0]
                    
                    for col_name in financial_df.columns:
                        if col_name not in ['company_id', 'period_end_date', 'statement_type']:
                            merged_col_name = f'financial_{col_name}'
                            if col_name in latest_financial_row.index:
                                company_stock.at[idx, merged_col_name] = latest_financial_row[col_name]
                                if merged_col_name not in self.fundamental_features:
                                    self.fundamental_features.append(merged_col_name)
                            else:
                                company_stock.at[idx, merged_col_name] = np.nan
                else:
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

    def add_moving_averages(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        for w in windows:
            col_name = f'MA_{w}'
            df[col_name] = df.groupby('company_id')['close_price'].transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )
            if col_name not in self.technical_features:
                self.technical_features.append(col_name)
        return df

    def add_exponential_moving_averages(self, df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        for w in windows:
            col_name = f'EMA_{w}'
            df[col_name] = df.groupby('company_id')['close_price'].transform(
                lambda x: x.ewm(span=w, adjust=False, min_periods=1).mean()
            )
            if col_name not in self.technical_features:
                self.technical_features.append(col_name)
        return df

    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        def safe_rsi(series):
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
        if not all(col in df.columns for col in ['close_price', 'volume', 'high_price', 'low_price']):
            logging.warning("Missing required price/volume columns for some volume indicators. Skipping affected indicators.")
            for col in ['volume_change_pct', 'volume_MA_20', 'volume_ratio', 'PVT', 'OBV', 'VWAP']:
                if col not in df.columns:
                    df[col] = np.nan
            return df

        df['volume_change_pct'] = df.groupby('company_id')['volume'].pct_change().fillna(0)
        df['volume_MA_20'] = df.groupby('company_id')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        ).fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume_MA_20']).replace([np.inf, -np.inf], 1).fillna(1)

        # FIX 1: Use correct PVT function name
        pvt_results = []
        for company_id, group_df in df.groupby('company_id'):
            try:
                # Changed from ta.volume.pvt_indicator to ta.volume.volume_price_trend
                pvt_series = ta.volume.volume_price_trend( 
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

        # FIX 2: Handle the OBV calculation more safely to avoid MultiIndex issues
        obv_results = []
        for company_id, group_df in df.groupby('company_id'):
            try:
                obv_series = ta.volume.on_balance_volume(
                    close=group_df['close_price'], 
                    volume=group_df['volume'], 
                    fillna=False
                )
                obv_results.append(obv_series)
            except Exception as e:
                logging.warning(f"Error calculating OBV for {company_id}: {e}. Using NaN values.")
                obv_series = pd.Series(np.nan, index=group_df.index)
                obv_results.append(obv_series)
        
        df['OBV'] = pd.concat(obv_results).reindex(df.index)
        
        df['VWAP'] = (df['volume'] * (df['high_price'] + df['low_price'] + df['close_price']) / 3).fillna(0) / df['volume'].fillna(1)

        volume_features = ['volume_change_pct', 'volume_MA_20', 'volume_ratio', 'PVT', 'OBV', 'VWAP']
        for feat in volume_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['price_volatility'] = df.groupby('company_id')['close_price'].transform(
            lambda x: x.pct_change().rolling(window=20, min_periods=1).std()
        ).fillna(0)
        
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
        if 'close_price' not in df.columns:
            logging.warning("Missing 'close_price' for price action features. Skipping.")
            return df 

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
            
            df['previous_close'] = df.groupby('company_id')['close_price'].shift(1)
            df['opening_gap'] = ((df['open_price'] - df['previous_close']) / df['previous_close'])
            price_action_features.append('opening_gap')
        
        for feat in price_action_features:
            if feat not in self.technical_features:
                self.technical_features.append(feat)
        return df

    def add_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        required_base_financial_cols = [
            'market_cap', 'net_income', 'total_equity', 'long_term_debt',
            'short_term_debt', 'total_assets', 'current_assets',
            'current_liabilities', 'cash_and_equivalents', 'inventory',
            'gross_profit', 'revenue', 'operating_income',
            'cost_of_goods_sold', 'operating_cash_flow', 
            'investing_cash_flow' 
        ]
        
        for col in required_base_financial_cols:
            if f'financial_{col}' not in df.columns:
                df[f'financial_{col}'] = np.nan 

        df['PE_ratio'] = df['market_cap'] / (df['financial_net_income'].replace(0, np.nan))
        df['PB_ratio'] = df['market_cap'] / df['financial_total_equity'].replace(0, np.nan)
        
        total_debt = (df['financial_long_term_debt'].fillna(0) + df['financial_short_term_debt'].fillna(0))
        df['debt_to_equity'] = total_debt / df['financial_total_equity'].replace(0, np.nan)
        df['debt_to_assets'] = total_debt / df['financial_total_assets'].replace(0, np.nan)
        
        df['current_ratio'] = df['financial_current_assets'] / df['financial_current_liabilities'].replace(0, np.nan)
        df['cash_ratio'] = df['financial_cash_and_equivalents'] / df['financial_current_liabilities'].replace(0, np.nan)
        df['quick_ratio'] = (df['financial_current_assets'] - df['financial_inventory'].fillna(0)) / df['financial_current_liabilities'].replace(0, np.nan)
        
        df['ROA'] = df['financial_net_income'] / df['financial_total_assets'].replace(0, np.nan) 
        df['ROE'] = df['financial_net_income'] / df['financial_total_equity'].replace(0, np.nan) 
        df['gross_margin'] = df['financial_gross_profit'] / df['financial_revenue'].replace(0, np.nan)
        df['operating_margin'] = df['financial_operating_income'] / df['financial_revenue'].replace(0, np.nan)
        df['net_margin'] = df['financial_net_income'] / df['financial_revenue'].replace(0, np.nan)
        
        df['asset_turnover'] = df['financial_revenue'] / df['financial_total_assets'].replace(0, np.nan) 
        df['inventory_turnover'] = df['financial_cost_of_goods_sold'] / df['financial_inventory'].replace(0, np.nan) 
        
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
        base_growth_cols = ['financial_revenue', 'financial_net_income', 'financial_total_assets']
        
        for col_prefix in base_growth_cols:
            if col_prefix not in df.columns:
                df[col_prefix] = np.nan 

        growth_features = []
        for col_prefix in base_growth_cols:
            if col_prefix in df.columns: 
                growth_col = f'{col_prefix}_growth'
                df[growth_col] = df.groupby('company_id')[col_prefix].pct_change() 
                growth_features.append(growth_col)
        
        if 'PE_ratio' in df.columns:
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

    def align_prediction_features(self, df: pd.DataFrame, training_features: List[str]) -> pd.DataFrame:
        """
        Specifically designed to align prediction data with training features.
        Handles missing financial data gracefully.
        """
        logging.info(f"🎯 Aligning prediction features to match training features...")
        
        # Get current feature columns (excluding metadata)
        non_feature_cols = [
            'trade_date', 'company_id', 'symbol', 'close_price', 'open_price', 
            'high_price', 'low_price', 'volume', 'market_cap', 'original_index'
        ]
        
        current_features = [col for col in df.columns if col not in non_feature_cols]
        missing_features = set(training_features) - set(current_features)
        
        if missing_features:
            logging.warning(f"Adding {len(missing_features)} missing features with appropriate defaults")
            
            # Add missing features with intelligent defaults
            for feature in missing_features:
                if any(keyword in feature.lower() for keyword in ['financial_', 'pe_ratio', 'pb_ratio', 'debt_to', 'current_ratio', 'cash_ratio', 'quick_ratio']):
                    # Financial features - use conservative defaults
                    if 'ratio' in feature.lower():
                        df[feature] = 1.0  # Neutral ratio
                    elif 'margin' in feature.lower():
                        df[feature] = 0.05  # 5% margin
                    elif 'growth' in feature.lower():
                        df[feature] = 0.0  # No growth
                    elif 'eps' in feature.lower():
                        df[feature] = 1.0  # $1 EPS
                    else:
                        df[feature] = 0.0
                else:
                    # Technical features - calculate if possible, otherwise use median
                    df[feature] = 0.0
        
        # Ensure we have only the training features in the correct order
        df_aligned = df[training_features].copy()
        
        # Final cleaning pass
        for col in df_aligned.columns:
            df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
            df_aligned[col] = df_aligned[col].replace([np.inf, -np.inf], np.nan)
            df_aligned[col] = df_aligned[col].fillna(0.0)
        
        logging.info(f"✅ Successfully aligned {len(training_features)} features for prediction")
        return df_aligned

    def clean_and_validate_features(self, df: pd.DataFrame, training_features: Optional[List[str]] = None) -> pd.DataFrame:
        logging.info("🧹 Cleaning and validating features...")
        initial_shape = df.shape

        non_feature_cols = [
            'trade_date', 'company_id', 'symbol', 'next_close_price', 
            'price_diff', 'label', 'previous_close', 'period_end_date', 'statement_type',
            'close_price', 'open_price', 'high_price', 'low_price', 'volume', 'market_cap'
        ]
        
        critical_numeric_for_fill = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 'market_cap']
        
        for col in critical_numeric_for_fill:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    logging.warning(f"NaNs found in critical column '{col}'. Filling with 0.")
                    df[col].fillna(0, inplace=True)
        
        # Handle feature alignment for prediction mode
        if training_features:
            logging.info(f"🔧 Aligning features to {len(training_features)} training features...")
            
            # First, ensure all training features exist in DataFrame
            missing_cols = set(training_features) - set(df.columns)
            if missing_cols:
                logging.warning(f"Missing {len(missing_cols)} features from training. Adding with default values: {list(missing_cols)[:10]}{'...' if len(missing_cols) > 10 else ''}")
                for col in missing_cols:
                    # Use more intelligent defaults based on feature type
                    if any(keyword in col.lower() for keyword in ['ratio', 'pe_', 'pb_', 'roe', 'roa']):
                        df[col] = 1.0  # Neutral ratio values
                    elif any(keyword in col.lower() for keyword in ['growth', 'change', 'return']):
                        df[col] = 0.0  # No growth/change
                    elif any(keyword in col.lower() for keyword in ['margin', 'percent']):
                        df[col] = 0.0  # Zero margin/percentage
                    else:
                        df[col] = 0.0  # Default to zero
            
            # Clean and process all potential feature columns (including newly added ones)
            all_potential_feature_cols = list(set(self.technical_features + self.fundamental_features + training_features))
            
            for col in all_potential_feature_cols:
                if col in df.columns and col not in non_feature_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Outlier handling
                    if df[col].std(skipna=True) > 0 and df[col].count() > 1: 
                        mean_val = df[col].mean(skipna=True)
                        std_val = df[col].std(skipna=True)
                        if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                            lower_bound = mean_val - 5 * std_val
                            upper_bound = mean_val + 5 * std_val
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    # Fill NaN values with appropriate defaults
                    if df[col].isnull().any():
                        if any(keyword in col.lower() for keyword in ['ratio', 'pe_', 'pb_', 'roe', 'roa']):
                            fill_value = 1.0  # Neutral ratio values
                        elif any(keyword in col.lower() for keyword in ['growth', 'change', 'return']):
                            fill_value = 0.0  # No growth/change
                        else:
                            median_val = df[col].median(skipna=True)
                            fill_value = median_val if pd.notna(median_val) else 0.0
                        
                        df[col].fillna(fill_value, inplace=True)
            
            # Remove extra columns not in training features
            extra_cols = set(df.columns) - set(training_features) - set(non_feature_cols)
            if extra_cols:
                logging.info(f"Removing {len(extra_cols)} extra columns not in training features")
                df.drop(columns=list(extra_cols), errors='ignore', inplace=True)
            
            # Ensure exact column order matches training
            try:
                df = df[training_features].copy()
                logging.info(f"✅ Features successfully aligned to training order. Shape: {df.shape}")
            except KeyError as e:
                missing_after_processing = set(training_features) - set(df.columns)
                logging.error(f"❌ Still missing features after processing: {missing_after_processing}")
                raise ValueError(f"Cannot align features. Missing: {missing_after_processing}")
                
        else:
            # Training mode - process all available features
            all_potential_feature_cols = list(set(self.technical_features + self.fundamental_features))
            
            for col in all_potential_feature_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    if df[col].std(skipna=True) > 0 and df[col].count() > 1: 
                        mean_val = df[col].mean(skipna=True)
                        std_val = df[col].std(skipna=True)
                        if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                            lower_bound = mean_val - 5 * std_val
                            upper_bound = mean_val + 5 * std_val
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    median_val = df[col].median(skipna=True)
                    df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

        logging.info(f"✅ Cleaned and validated features. Shape changed from {initial_shape} to {df.shape}")
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates target labels for classification.
        Assumes 'close_price' is available and sorted by company_id and trade_date.
        Labels: 0 (Sell), 1 (Hold), 2 (Buy) based on future price movement.
        """
        if 'close_price' not in df.columns:
            logging.error("Cannot create labels: 'close_price' column is missing.")
            df['label'] = np.nan
            return df

        df['next_close_price'] = df.groupby('company_id')['close_price'].shift(-1)
        
        df['price_diff'] = df['next_close_price'] - df['close_price']
        df['price_change_percent'] = (df['price_diff'] / df['close_price']) * 100

        buy_threshold = 0.5  
        sell_threshold = -0.5 

        conditions = [
            (df['price_change_percent'] >= buy_threshold),
            (df['price_change_percent'] <= sell_threshold)
        ]
        choices = [2, 0] 

        df['label'] = np.select(conditions, choices, default=1) 

        df.dropna(subset=['label'], inplace=True)
        
        logging.info(f"Labels created. Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
        return df

    def create_features_and_labels(self, stock_df: pd.DataFrame, financial_df: Optional[pd.DataFrame] = None, 
                                   is_prediction: bool = False, training_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Main method to create all features and labels
        """
        
        self.technical_features = []
        self.fundamental_features = []
        self.all_features = []
        
        logging.info("🔄 Starting feature engineering...")
        
        if 'trade_date' not in stock_df.columns:
            raise ValueError("stock_df must contain a 'trade_date' column.")
        
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        stock_df = stock_df.sort_values(by=['company_id', 'trade_date']).reset_index(drop=True)
        stock_df['original_index'] = stock_df.index # Preserve original index for metadata

        if financial_df is not None and not financial_df.empty:
            logging.info("💰 Merging financial data with stock data...")
            df = self.merge_financial_with_stock_data(stock_df, financial_df)
        else:
            df = stock_df.copy()
            logging.info("ℹ️ No financial data provided, using technical indicators only")
        
        # --- Crucial Pre-clean_and_validate_features Index & Column Reset ---
        # Ensure df is sorted and has a simple default integer index, and NO MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(drop=False) # Convert MultiIndex levels to columns, maintain data
            logging.info("MultiIndex detected and converted to columns for processing.")
        
        # Ensure a standard, unnamed RangeIndex for df before feature additions/cleaning
        df = df.reset_index(drop=True)
        df.index.name = None # Explicitly remove index name to prevent potential issues
        logging.info("DataFrame index reset and name removed for consistency.")

        # Re-sort to maintain order after potential MultiIndex reset, then re-reset index
        df = df.sort_values(['company_id', 'trade_date']).reset_index(drop=True)
        df.index.name = None # Ensure index name is still None after sort/reset
        logging.info("DataFrame resorted and index re-reset to ensure consistent order.")
        # ----------------------------------------------------------
        
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
        
        if 'financial_revenue' in df.columns and not df['financial_revenue'].isna().all():
            logging.info("💰 Adding fundamental indicators...")
            df = self.add_financial_ratios(df)
            df = self.add_growth_metrics(df)
        else:
            logging.warning("⚠️ Financial data not available or merged successfully for fundamental indicators (financial_revenue not found or all NaN).")

        # --- Preserve full metadata BEFORE clean_and_validate_features potentially strips columns ---
        metadata_cols_to_preserve = [
            'company_id', 'trade_date', 'symbol', 'close_price', 'original_index',
            'open_price', 'high_price', 'low_price', 'volume', 'market_cap'
        ] 
        
        available_initial_metadata_cols = [col for col in metadata_cols_to_preserve if col in df.columns]
        initial_metadata = df[available_initial_metadata_cols].copy()
        
        # Ensure initial_metadata has a clean RangeIndex and no name for later alignment
        initial_metadata = initial_metadata.reset_index(drop=True)
        initial_metadata.index.name = None
        
        logging.info("🧹 Cleaning and validating features...")
        df_processed = self.clean_and_validate_features(df, training_features=training_features)
        
        # --- Metadata extraction and final X, y preparation (more robust alignment) ---
        if df_processed.empty:
            logging.warning("df_processed is empty after cleaning and validation. Returning empty results.")
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

        # Align metadata using the 'original_index' column
        if 'original_index' in df_processed.columns:
            # Create a temporary index on 'original_index' for precise alignment
            temp_df_processed_indexed = df_processed.set_index('original_index', drop=False) # Keep original_index as column
            temp_initial_metadata_indexed = initial_metadata.set_index('original_index')

            # Reindex metadata to match rows present in df_processed
            metadata = temp_initial_metadata_indexed.reindex(temp_df_processed_indexed.index).reset_index(drop=False) # original_index back to column
            
            # Ensure df_processed itself also has a clean RangeIndex for model input, dropping original_index
            df_processed.drop(columns=['original_index'], inplace=True, errors='ignore')
            df_processed.reset_index(drop=True, inplace=True)
        else:
            # Fallback if original_index wasn't preserved (shouldn't happen)
            logging.warning("original_index not found in df_processed, falling back to direct index alignment. This might cause issues if rows were dropped.")
            metadata = initial_metadata.loc[df_processed.index].copy()
            metadata.reset_index(drop=True, inplace=True) 

        # Final check to ensure all outputs have simple RangeIndexes and no names
        metadata.reset_index(drop=True, inplace=True)
        metadata.index.name = None
        
        # The 'y' Series should also be reset
        y_final = pd.Series([], dtype=float) # Default empty series for prediction mode

        if not is_prediction:
            logging.info("🎯 Creating target labels for training...")
            df_labeled = self.create_labels(df_processed) # df_processed should have close_price for this

            # Filter metadata based on valid labels from df_labeled
            valid_label_indices = df_labeled['label'].dropna().index
            
            if valid_label_indices.empty:
                logging.warning("No valid labels after creation. Returning empty results for training.")
                return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

            df_final_for_training = df_labeled.loc[valid_label_indices].copy()
            
            # Re-align metadata to the rows that have valid labels
            if 'original_index' in df_final_for_training.columns and 'original_index' in metadata.columns:
                metadata = metadata[metadata['original_index'].isin(df_final_for_training['original_index'])].copy()
                metadata.reset_index(drop=True, inplace=True)
                # Drop original_index from df_final_for_training features, keep in metadata
                df_final_for_training.drop(columns=['original_index'], inplace=True, errors='ignore')

            # Dynamically determine which features are actually present in the DataFrame for training
            all_potential_features_from_lists = list(set(self.technical_features + self.fundamental_features))
            
            excluded_from_model_features = [
                'trade_date', 'company_id', 'symbol', 'next_close_price', 
                'price_diff', 'price_change_percent', 'label', 'previous_close',
                'original_index', # Exclude if it accidentally got into final training features
                'open_price', 'high_price', 'low_price', 'close_price', 
                'volume', 'market_cap'
            ] 
            
            self.all_features = [
                col for col in (all_potential_features_from_lists) 
                if col in df_final_for_training.columns and col not in excluded_from_model_features
            ]
            self.all_features.sort() 

            X = df_final_for_training[self.all_features].copy()
            y_final = df_final_for_training['label'].copy()
            
            logging.info(f"✅ Feature engineering complete for training!")
            logging.info(f"   📊 Technical features: {len(self.technical_features)}")
            logging.info(f"   💼 Fundamental features: {len(self.fundamental_features)}")
            logging.info(f"   🔢 Total features (dynamically selected): {len(self.all_features)}")
            logging.info(f"📋 Final dataset summary for training:")
            logging.info(f"   🔢 Samples: {len(X)}")
            logging.info(f"   📊 Features: {len(X.columns)}")
            logging.info(f"   🏷️ Label distribution: {y_final.value_counts().sort_index().to_dict()}")

        else: # If is_prediction is True
            logging.info("ℹ️ Skipping target label creation for prediction...")
            if training_features is None:
                raise ValueError("training_features must be provided when is_prediction is True for prediction mode.")
            
            # Use the specialized alignment method for predictions
            try:
                X = self.align_prediction_features(df_processed, training_features)
                logging.info(f"✅ Feature engineering complete for prediction!")
                logging.info(f"   🔢 Samples for prediction: {len(X)}")
                logging.info(f"   📊 Features: {len(X.columns)}")
            except Exception as e:
                logging.error(f"❌ Error in feature alignment for prediction: {str(e)}")
                # Fallback to basic alignment
                X = df_processed[training_features].copy() if all(col in df_processed.columns for col in training_features) else pd.DataFrame()
                if X.empty:
                    logging.error("Cannot create prediction features - too many missing columns")
                    return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
        
        # Final reset of indices for all returned DataFrames/Series
        X.reset_index(drop=True, inplace=True)
        y_final.reset_index(drop=True, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        
        return X, y_final, metadata
    
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

    def save_training_features(self, filepath: str = 'training_features.txt'):
        """Save the list of features used in training for prediction alignment"""
        import json
        
        feature_info = {
            'all_features': self.all_features,
            'technical_features': self.technical_features,
            'fundamental_features': self.fundamental_features,
            'feature_count': len(self.all_features)
        }
        
        with open(filepath, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logging.info(f"💾 Training features saved to: {filepath}")
    
    def load_training_features(self, filepath: str = 'training_features.txt') -> List[str]:
        """Load the list of features used in training"""
        import json
        
        try:
            with open(filepath, 'r') as f:
                feature_info = json.load(f)
            
            logging.info(f"📂 Loaded {feature_info['feature_count']} training features from: {filepath}")
            return feature_info['all_features']
        except FileNotFoundError:
            logging.warning(f"Training features file not found: {filepath}")
            return []
        except Exception as e:
            logging.error(f"Error loading training features: {str(e)}")
            return []

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