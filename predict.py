import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import re
from datetime import datetime, timedelta
import os # Import os for path manipulation

# Assuming data_loader.py and feature_engineering.py are in the 'data' directory
from data.data_loader import load_stock_data, load_financial_data
from data.feature_engineering import FeatureEngineer 

# Configure logging for better visibility during execution
# Set to INFO by default, but specific sections (like feature loading) will temporarily go to DEBUG
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == "__main__":
    MODEL_PATH = 'models/random_forest_model.pkl'
    FEATURE_INFO_PATH = 'results/feature_info.txt'

    try:
        # 1. Load the trained model
        model = joblib.load(MODEL_PATH)
        logging.info(f"Successfully loaded model from {MODEL_PATH}")

        # 2. Load the list of features the model was trained on
        training_features = []
        in_all_features_list_section = False
        found_all_features_list_header = False # New flag to correctly handle the separator line
        
        try:
            with open(FEATURE_INFO_PATH, 'r') as f:
                logging.info(f"Attempting to read features from {FEATURE_INFO_PATH}")
                # Temporarily set logging level to DEBUG to see detailed messages for this section
                initial_log_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.DEBUG) 
                
                for line_num, line in enumerate(f, 1):
                    stripped_line = line.strip()
                    logging.debug(f"Line {line_num} (stripped): '{stripped_line}'") # Stripped line

                    if "ALL FEATURES LIST" in stripped_line:
                        logging.info("Found 'ALL FEATURES LIST' section header.")
                        found_all_features_list_header = True
                        continue # Skip the header line itself

                    if found_all_features_list_header:
                        # If we've found the header, and the next line is the '----' separator, skip it
                        if stripped_line == "--------------------":
                            logging.debug("Skipping separator line after 'ALL FEATURES LIST' header.")
                            in_all_features_list_section = True # Now we are truly in the feature list content
                            continue

                        # Only start parsing features if we are past the header AND the separator
                        if in_all_features_list_section:
                            # Regex to match lines like " 1. feature_name" (handles leading space before number)
                            match = re.match(r'^\s*\d+\.\s*([\w_]+)\s*$', stripped_line) 
                            if match:
                                feature_name = match.group(1)
                                training_features.append(feature_name)
                                logging.debug(f"Extracted feature: '{feature_name}'")
                            # Stop reading if we hit an empty line or a line that doesn't match the feature format
                            elif not stripped_line or not re.match(r'^\s*\d+\.', stripped_line):
                                logging.info(f"End of feature list section detected on line {line_num}. Breaking. Stripped line: '{stripped_line}'")
                                break
                            else:
                                logging.warning(f"Line {line_num} in feature list section did not match expected feature format: '{stripped_line}'")
                                # Continue processing in case there are valid features after an unexpected line
                                pass 

        except FileNotFoundError:
            raise FileNotFoundError(f"Feature info file not found at {FEATURE_INFO_PATH}. "
                                    "Please ensure your training pipeline generates this file.")
        finally:
            # Reset logging level to its original value after this section
            if 'initial_log_level' in locals():
                logging.getLogger().setLevel(initial_log_level)
        
        logging.info(f"Finished reading feature info. Found {len(training_features)} features before deduplication.")
        if not training_features:
            raise ValueError("Could not extract training features from feature_info.txt. "
                             "Ensure 'ALL FEATURES LIST' section format is correct in your training output.")
        
        # Ensure unique features and maintain order from file (dict.fromkeys preserves insertion order)
        training_features = list(dict.fromkeys(training_features)) 
        logging.info(f"Loaded {len(training_features)} unique training features for prediction: {training_features}")

        # 3. Load all necessary historical data
        logging.info("Loading all historical stock data from the database... (This might take a moment)")
        all_historical_stock_data = load_stock_data()
        
        logging.info("Loading all historical financial data from the database... (This might take a moment)")
        all_historical_financial_data = load_financial_data()

        if all_historical_stock_data.empty:
            raise ValueError("No historical stock data loaded from the database. Cannot proceed with prediction.")
        if all_historical_financial_data.empty:
            logging.warning("No historical financial data loaded. Fundamental indicators will be skipped.")

        all_historical_stock_data['trade_date'] = pd.to_datetime(all_historical_stock_data['trade_date'])
        
        latest_data_date = all_historical_stock_data['trade_date'].max().date()
        logging.info(f"Latest available stock data date: {latest_data_date.strftime('%Y-%m-%d')}")

        # Determine the prediction basis date (e.g., today or the latest available data)
        # For a truly 'live' prediction, this would be the actual current date or yesterday's close.
        # For backtesting or predicting on the latest available historical data, use max trade_date.
        prediction_basis_date = latest_data_date 

        logging.info(f"Preparing to generate features for predictions based on data up to: {prediction_basis_date.strftime('%Y-%m-%d')}")

        # Define a window for indicator calculation (e.g., 90 days for MAs, RSI, etc.)
        # This ensures enough historical context for feature engineering
        max_indicator_window_days = 90 
        start_date_for_indicators = prediction_basis_date - timedelta(days=max_indicator_window_days)

        logging.info(f"Filtering stock data from {start_date_for_indicators.strftime('%Y-%m-%d')} "
                     f"up to and including {prediction_basis_date.strftime('%Y-%m-%d')} for feature engineering.")

        # Filter stock data to include only what's needed for feature engineering indicators
        full_stock_data_for_features = all_historical_stock_data[
            (all_historical_stock_data['trade_date'].dt.date >= start_date_for_indicators) &
            (all_historical_stock_data['trade_date'].dt.date <= prediction_basis_date)
        ].copy()

        # Filter financial data for relevant companies
        companies_in_scope = full_stock_data_for_features['company_id'].unique()
        all_historical_financial_data['period_end_date'] = pd.to_datetime(all_historical_financial_data['period_end_date'])
        financial_data_filtered = all_historical_financial_data[
            all_historical_financial_data['company_id'].isin(companies_in_scope)
        ].copy()
        
        if full_stock_data_for_features.empty:
            raise ValueError("No recent stock data available after filtering for feature engineering. "
                             "Ensure your database has data for the specified date range.")
        
        logging.info(f"Filtered stock data for feature engineering: {full_stock_data_for_features.shape}")
        logging.info(f"Filtered financial data for feature engineering: {financial_data_filtered.shape}")

        # 4. Instantiate FeatureEngineer and create features for prediction
        engineer = FeatureEngineer()
        
        # Call create_features_and_labels with is_prediction=True and training_features
        # The FeatureEngineer will now handle aligning the created features with the training features
        X_predicted_data, y_dummy, metadata_info = engineer.create_features_and_labels(
            stock_df=full_stock_data_for_features, 
            financial_df=financial_data_filtered, 
            is_prediction=True, # Flag for prediction mode
            training_features=training_features # Pass the features the model expects
        )
        
        logging.info(f"Feature engineering completed by FeatureEngineer for prediction. Shape: {X_predicted_data.shape}")

        # 5. Select the latest engineered data point for each company
        # X_predicted_data now contains the engineered features for the filtered date range.
        # metadata_info contains 'company_id', 'trade_date', 'original_index' etc.
        
        # Create a combined DataFrame for easier latest row selection and mapping
        # We merge X_predicted_data with metadata_info using their internal indices (which should align).
        combined_df = metadata_info.merge(X_predicted_data, left_index=True, right_index=True, how='inner')
        
        # Ensure combined_df is sorted by company and date to select the true latest row for each company
        combined_df = combined_df.sort_values(by=['company_id', 'trade_date'])

        # Get the index (from combined_df) of the row with the latest trade_date for each company
        idx_latest_per_company = combined_df.groupby('company_id')['trade_date'].idxmax()
        
        # Select the actual rows for prediction, which are the latest feature sets for each company
        X_final_predict_with_meta = combined_df.loc[idx_latest_per_company].copy()
        
        if X_final_predict_with_meta.empty:
            raise ValueError(f"No data found for prediction after selecting latest entries for basis date {prediction_basis_date.strftime('%Y-%m-%d')}. "
                             "This might mean insufficient historical data for indicators, or no recent data after processing.")

        # Extract only the features (columns present in training_features) and set 'company_id' as index
        # FeatureEngineer already ensured these columns are present and in order in X_predicted_data.
        X_final_predict = X_final_predict_with_meta.set_index('company_id')[training_features].copy() 

        logging.info(f"Prepared final data for prediction with shape: {X_final_predict.shape}")
        
        if X_final_predict.empty:
            raise ValueError("X_final_predict is empty after final alignment. No predictions can be made.")

        # 6. Make predictions
        predictions = model.predict(X_final_predict)
        prediction_probabilities = model.predict_proba(X_final_predict)

        # Map integer labels to readable actions
        # Ensure model.classes_ are sorted to match the order of probabilities
        model_classes_sorted = sorted(model.classes_)
        class_mapping = {cls: idx for idx, cls in enumerate(model_classes_sorted)}
        label_map_display = {-1.0: 'Sell', 0.0: 'Hold', 1.0: 'Buy'}

        predicted_labels = [label_map_display.get(p, 'Unknown') for p in predictions]

        prob_sell = prediction_probabilities[:, class_mapping[-1.0]] if -1.0 in class_mapping else np.array([np.nan] * len(predictions))
        prob_hold = prediction_probabilities[:, class_mapping[0.0]] if 0.0 in class_mapping else np.array([np.nan] * len(predictions))
        prob_buy = prediction_probabilities[:, class_mapping[1.0]] if 1.0 in class_mapping else np.array([np.nan] * len(predictions))

        # 7. Display results
        logging.info("\n--- Predictions ---")
        prediction_results = pd.DataFrame({
            'company_id': X_final_predict.index, # company_id is the index here
            'trade_date': X_final_predict_with_meta['trade_date'].values, # Get the trade_date for the prediction
            'Predicted_Action': predicted_labels,
            'Probability_Sell': prob_sell,
            'Probability_Hold': prob_hold,
            'Probability_Buy': prob_buy
        })
        prediction_results = prediction_results.reset_index(drop=False) # Reset index if company_id was the index

        logging.info("\n" + prediction_results.to_string())

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)