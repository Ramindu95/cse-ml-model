import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import json
from datetime import datetime, timedelta
import os

# Assuming data_loader.py and feature_engineering.py are in the 'data' directory
from data.data_loader import load_stock_data, load_financial_data
from data.feature_engineering import FeatureEngineer 

# Configure logging for better visibility during execution
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)

def load_training_features(feature_path: str) -> list:
    """
    Load training features from JSON file (preferred) or fallback to text file
    """
    # Try JSON format first (recommended)
    json_path = feature_path.replace('.txt', '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                feature_info = json.load(f)
            logging.info(f"✅ Loaded {feature_info['feature_count']} training features from JSON: {json_path}")
            return feature_info['all_features']
        except Exception as e:
            logging.warning(f"Failed to load JSON features from {json_path}: {e}")
    
    # Fallback to text format parsing
    if os.path.exists(feature_path):
        return load_features_from_text_file(feature_path)
    
    raise FileNotFoundError(f"Training features not found at {json_path} or {feature_path}")

def load_features_from_text_file(feature_path: str) -> list:
    """
    Load training features from the text file format (fallback method)
    """
    training_features = []
    in_all_features_list_section = False
    found_all_features_list_header = False
    
    try:
        with open(feature_path, 'r') as f:
            logging.info(f"Loading features from text file: {feature_path}")
            
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()

                if "ALL FEATURES LIST" in stripped_line:
                    logging.info("Found 'ALL FEATURES LIST' section header.")
                    found_all_features_list_header = True
                    continue

                if found_all_features_list_header:
                    if stripped_line.startswith("----"):
                        logging.debug("Skipping separator line after 'ALL FEATURES LIST' header.")
                        in_all_features_list_section = True
                        continue

                    if in_all_features_list_section:
                        # Parse feature lines like " 1. feature_name"
                        import re
                        match = re.match(r'^\s*\d+\.\s*([\w_]+)\s*$', stripped_line)
                        if match:
                            feature_name = match.group(1)
                            training_features.append(feature_name)
                        elif not stripped_line or not re.match(r'^\s*\d+\.', stripped_line):
                            logging.info(f"End of feature list section detected on line {line_num}.")
                            break
                        else:
                            logging.warning(f"Line {line_num} did not match expected format: '{stripped_line}'")

        # Remove duplicates while preserving order
        training_features = list(dict.fromkeys(training_features))
        logging.info(f"✅ Loaded {len(training_features)} unique training features from text file")
        return training_features
        
    except Exception as e:
        raise RuntimeError(f"Error parsing text file {feature_path}: {e}")

def get_company_info(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get basic company information for display purposes
    """
    company_info = stock_df.groupby('company_id').agg({
        'trade_date': 'max',
        'close_price': 'last',
        'volume': 'last'
    }).reset_index()
    
    company_info.columns = ['company_id', 'latest_date', 'latest_close', 'latest_volume']
    return company_info

def save_predictions_to_file(predictions_df: pd.DataFrame, output_dir: str = "results"):
    """
    Save predictions to a timestamped CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_predictions_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    predictions_df.to_csv(filepath, index=False)
    logging.info(f"💾 Predictions saved to: {filepath}")
    return filepath

def display_prediction_summary(predictions_df: pd.DataFrame):
    """
    Display a summary of predictions
    """
    logging.info("\n" + "="*80)
    logging.info("STOCK PREDICTIONS SUMMARY")
    logging.info("="*80)
    
    # Action distribution
    action_counts = predictions_df['Predicted_Action'].value_counts()
    logging.info(f"\n📊 Action Distribution:")
    for action, count in action_counts.items():
        percentage = (count / len(predictions_df)) * 100
        logging.info(f"   {action}: {count} companies ({percentage:.1f}%)")
    
    # High confidence predictions
    high_confidence = predictions_df[predictions_df['Max_Probability'] > 0.7]
    if not high_confidence.empty:
        logging.info(f"\n🎯 High Confidence Predictions (>70%):")
        for _, row in high_confidence.iterrows():
            logging.info(f"   Company {row['company_id']}: {row['Predicted_Action']} ({row['Max_Probability']:.1%})")
    
    # Companies to watch
    buy_signals = predictions_df[predictions_df['Predicted_Action'] == 'Buy'].sort_values('Probability_Buy', ascending=False)
    if not buy_signals.empty:
        logging.info(f"\n📈 Top Buy Signals:")
        for _, row in buy_signals.head(5).iterrows():
            logging.info(f"   Company {row['company_id']}: {row['Probability_Buy']:.1%} confidence")
    
    sell_signals = predictions_df[predictions_df['Predicted_Action'] == 'Sell'].sort_values('Probability_Sell', ascending=False)
    if not sell_signals.empty:
        logging.info(f"\n📉 Top Sell Signals:")
        for _, row in sell_signals.head(5).iterrows():
            logging.info(f"   Company {row['company_id']}: {row['Probability_Sell']:.1%} confidence")

def main():
    """Main prediction function"""
    
    # Configuration
    MODEL_PATH = 'models/random_forest_model.pkl'
    FEATURE_INFO_PATH = 'results/feature_info.txt'  # Primary path (your existing file)
    FALLBACK_FEATURE_PATH = 'models/training_features.json'  # Fallback to JSON format
    
    try:
        # 1. Load the trained model
        logging.info("🔄 Loading trained model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = joblib.load(MODEL_PATH)
        logging.info(f"✅ Successfully loaded model from {MODEL_PATH}")

        # 2. Load training features
        logging.info("🔄 Loading training features...")
        try:
            training_features = load_training_features(FEATURE_INFO_PATH)
        except FileNotFoundError:
            logging.warning(f"Primary feature file not found at {FEATURE_INFO_PATH}, trying fallback: {FALLBACK_FEATURE_PATH}")
            training_features = load_training_features(FALLBACK_FEATURE_PATH)
        
        if not training_features:
            raise ValueError("No training features loaded. Cannot proceed with prediction.")

        # 3. Load historical data
        logging.info("🔄 Loading historical data from database...")
        all_historical_stock_data = load_stock_data()
        all_historical_financial_data = load_financial_data()

        if all_historical_stock_data.empty:
            raise ValueError("No historical stock data loaded from the database.")
        
        if all_historical_financial_data.empty:
            logging.warning("⚠️ No historical financial data loaded. Only technical indicators will be used.")

        # Data preprocessing
        all_historical_stock_data['trade_date'] = pd.to_datetime(all_historical_stock_data['trade_date'])
        latest_data_date = all_historical_stock_data['trade_date'].max().date()
        logging.info(f"📅 Latest available data: {latest_data_date}")

        # 4. Prepare data for feature engineering
        prediction_basis_date = latest_data_date
        max_indicator_window_days = 90
        start_date_for_indicators = prediction_basis_date - timedelta(days=max_indicator_window_days)

        logging.info(f"🔄 Filtering data for feature engineering ({start_date_for_indicators} to {prediction_basis_date})...")
        
        full_stock_data_for_features = all_historical_stock_data[
            (all_historical_stock_data['trade_date'].dt.date >= start_date_for_indicators) &
            (all_historical_stock_data['trade_date'].dt.date <= prediction_basis_date)
        ].copy()

        if full_stock_data_for_features.empty:
            raise ValueError("No stock data available for the specified date range.")

        # Filter financial data for relevant companies
        companies_in_scope = full_stock_data_for_features['company_id'].unique()
        logging.info(f"🏢 Processing predictions for {len(companies_in_scope)} companies")
        
        financial_data_filtered = None
        if not all_historical_financial_data.empty:
            all_historical_financial_data['period_end_date'] = pd.to_datetime(all_historical_financial_data['period_end_date'])
            financial_data_filtered = all_historical_financial_data[
                all_historical_financial_data['company_id'].isin(companies_in_scope)
            ].copy()

        # 5. Feature engineering
        logging.info("🔄 Starting feature engineering...")
        engineer = FeatureEngineer()
        
        X_predicted_data, _, metadata_info = engineer.create_features_and_labels(
            stock_df=full_stock_data_for_features,
            financial_df=financial_data_filtered,
            is_prediction=True,
            training_features=training_features
        )
        
        if X_predicted_data.empty:
            raise ValueError("Feature engineering produced no data for prediction.")
        
        logging.info(f"✅ Feature engineering completed. Shape: {X_predicted_data.shape}")

        # 6. Select latest data point for each company
        logging.info("🔄 Selecting latest data points for prediction...")
        
        # Combine features with metadata for company/date selection
        combined_df = metadata_info.merge(X_predicted_data, left_index=True, right_index=True, how='inner')
        combined_df = combined_df.sort_values(by=['company_id', 'trade_date'])

        # Get latest entry for each company
        idx_latest_per_company = combined_df.groupby('company_id')['trade_date'].idxmax()
        latest_data_per_company = combined_df.loc[idx_latest_per_company].copy()
        
        if latest_data_per_company.empty:
            raise ValueError("No latest data points found for prediction.")

        # Extract features for prediction
        X_final_predict = latest_data_per_company[training_features].copy()
        logging.info(f"📊 Final prediction dataset shape: {X_final_predict.shape}")

        # 7. Make predictions
        logging.info("🔄 Making predictions...")
        predictions = model.predict(X_final_predict)
        prediction_probabilities = model.predict_proba(X_final_predict)

        # Map predictions to readable labels
        # Handle different label encodings (0,1,2 or -1,0,1)
        unique_predictions = np.unique(predictions)
        if set(unique_predictions).issubset({0, 1, 2}):
            # Standard encoding: 0=Sell, 1=Hold, 2=Buy
            label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
            class_order = [0, 1, 2]
        elif set(unique_predictions).issubset({-1, 0, 1}):
            # Alternative encoding: -1=Sell, 0=Hold, 1=Buy
            label_map = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
            class_order = [-1, 0, 1]
        else:
            # Generic handling
            sorted_classes = sorted(model.classes_)
            label_map = {cls: f'Action_{i}' for i, cls in enumerate(sorted_classes)}
            class_order = sorted_classes

        predicted_labels = [label_map.get(p, 'Unknown') for p in predictions]

        # Extract probabilities in correct order
        class_to_prob_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
        
        prob_columns = {}
        for cls in class_order:
            if cls in class_to_prob_idx:
                prob_idx = class_to_prob_idx[cls]
                action_name = label_map[cls]
                prob_columns[f'Probability_{action_name}'] = prediction_probabilities[:, prob_idx]

        # 8. Create results DataFrame
        logging.info("🔄 Preparing results...")
        
        prediction_results = pd.DataFrame({
            'company_id': latest_data_per_company['company_id'].values,
            'prediction_date': latest_data_per_company['trade_date'].dt.date.values,
            'Predicted_Action': predicted_labels,
            **prob_columns
        })
        
        # Add maximum probability for confidence assessment
        prediction_results['Max_Probability'] = prediction_probabilities.max(axis=1)
        
        # Sort by company_id for better readability
        prediction_results = prediction_results.sort_values('company_id').reset_index(drop=True)

        # 9. Display and save results
        display_prediction_summary(prediction_results)
        
        # Save detailed results
        output_file = save_predictions_to_file(prediction_results)
        
        # Display detailed results
        logging.info("\n" + "="*80)
        logging.info("DETAILED PREDICTIONS")
        logging.info("="*80)
        logging.info("\n" + prediction_results.to_string(index=False))
        
        logging.info(f"\n✅ Prediction completed successfully!")
        logging.info(f"📈 Processed {len(prediction_results)} companies")
        logging.info(f"💾 Results saved to: {output_file}")

    except Exception as e:
        logging.error(f"❌ An error occurred during prediction: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()