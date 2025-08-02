#!/usr/bin/env python3
"""
ML Training Script for Financial Data Extraction

This script handles training of ML models for financial data extraction,
data preparation, model evaluation, and continuous learning.
"""

import os
import sys
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import argparse
from dataclasses import dataclass, asdict
import pickle # Used for saving/loading models, though joblib is preferred for sklearn

# Import the ML models manager
# Assuming ml_models_financial is the file containing FinancialMLModelManager
# Renaming to ml_models_manager for consistency with previous responses
from ml_models_manager import FinancialMLModelManager, FinancialFieldExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    model_dir: str = "ml_models"
    data_dir: str = "training_data" # Directory for manually curated training data files
    db_path: str = "extraction_learning.db" # Path to the SQLite feedback database
    min_samples_per_field: int = 10 # Minimum samples required to train a model for a field
    validation_split: float = 0.2 # Proportion of data to use for validation during training
    retrain_threshold_days: int = 7 # How often to retrain models, in days
    performance_threshold: Dict[str, Dict[str, float]] = None # Thresholds for model performance
    
    def __post_init__(self):
        if self.performance_threshold is None:
            self.performance_threshold = {
                'numeric': {'mae': 100000.0},  # Max acceptable MAE for numeric fields (adjust based on currency scale)
                'categorical': {'accuracy': 0.80}  # Min accuracy for categorical fields
            }

class TrainingDataManager:
    """Manages training data collection and preparation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.db_path = config.db_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories"""
        Path(self.config.model_dir).mkdir(exist_ok=True)
        Path(self.config.data_dir).mkdir(exist_ok=True)
    
    def load_training_data_from_db(self) -> Dict[str, List[Tuple[str, Any]]]:
        """Load training data from feedback database"""
        training_data = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get feedback data with document texts
                # Note: 'extraction_data' column in 'extractions' table is assumed to contain
                # a JSON string which includes a 'full_text_context' field or similar
                # representing the text that was extracted from.
                query = """
                SELECT f.field_name, f.correct_value, e.extraction_data
                FROM feedback f
                JOIN extractions e ON f.document_hash = e.document_hash
                WHERE f.correct_value IS NOT NULL AND f.correct_value != ''
                ORDER BY f.timestamp DESC
                """
                
                cursor = conn.execute(query)
                feedback_data = cursor.fetchall()
                
                for field_name, correct_value, extraction_data_json in feedback_data:
                    try:
                        # Parse extraction data to get document text context
                        extraction_data = json.loads(extraction_data_json)
                        
                        # Use the 'full_text_context' stored during simulation/extraction
                        # This is crucial for feature generation in FinancialFieldExtractor
                        document_text_context = extraction_data.get('full_text_context')
                        
                        if not document_text_context:
                            # Fallback to synthetic text if context is missing (should not happen in real usage)
                            logger.warning(f"Missing 'full_text_context' for {field_name}, hash {feedback_entry['document_hash']}. Creating synthetic text.")
                            document_text_context = self._create_synthetic_text(field_name, correct_value, extraction_data)
                        
                        if field_name not in training_data:
                            training_data[field_name] = []
                        
                        training_data[field_name].append((document_text_context, correct_value))
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse extraction data for {field_name} (Error: {e}). Skipping entry.")
                        continue
                
        except sqlite3.Error as e:
            logger.error(f"Database error when loading training data: {e}")
        
        return training_data
    
    def _create_synthetic_text(self, field_name: str, value: Any, extraction_data: Dict) -> str:
        """
        Creates synthetic document text for training.
        This is a fallback if actual document context is not available in feedback.
        In a real system, you'd store the actual document snippets or full text.
        """
        company_name = extraction_data.get('company_name', 'Sample Company')
        
        # Define templates for various fields
        templates = {
            'revenue': f"The company {company_name} reported Revenue of {value} for the period. Total sales were high.",
            'profit_for_period': f"Net Profit for the period for {company_name} was {value}. Earnings after tax.",
            'operating_profit': f"Operating Profit for {company_name} amounted to {value}. EBIT was strong.",
            'basic_eps': f"Basic EPS of {company_name} was {value}. Earnings Per Share data.",
            'total_assets': f"Total Assets of {company_name} stood at {value}. Balance Sheet figures.",
            'total_liabilities': f"Total Liabilities for {company_name} were {value}. Debt and obligations.",
            'total_equity': f"Total Equity of {company_name} was {value}. Shareholders' capital.",
            'gross_profit': f"Gross Profit of {company_name} was {value}. Cost of sales considered.",
            'ebitda': f"EBITDA for {company_name} was {value}. Performance before non-operating items.",
            'net_cash_flow': f"Net Cash Flow from operations for {company_name} was {value}. Cash movements.",
            'report_type': f"This is an {value} financial report for {company_name}. Quarterly or Annual.",
            'audit_status': f"These financial statements for {company_name} are {value}. Audit opinion."
        }
        
        return templates.get(field_name, f"Generic text for {field_name}: {value} for {company_name}.")
    
    def load_training_data_from_files(self) -> Dict[str, List[Tuple[str, Any]]]:
        """Load training data from manually curated JSON files in data_dir"""
        training_data = {}
        data_dir = Path(self.config.data_dir)
        
        for file_path in data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                field_name = file_path.stem.replace('_data', '') # e.g., 'revenue_data.json' -> 'revenue'
                if field_name not in training_data:
                    training_data[field_name] = []
                
                for item in data:
                    text = item.get('text', '')
                    label = item.get('label')
                    if text and label is not None:
                        training_data[field_name].append((text, label))
                
                logger.info(f"Loaded {len(training_data[field_name])} samples for {field_name} from file.")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading training data from {file_path}: {e}")
            except FileNotFoundError:
                logger.warning(f"Training data file not found: {file_path}")
        
        return training_data
    
    def create_sample_training_data(self) -> Dict[str, List[Tuple[str, Any]]]:
        """
        Creates sample training data for testing and initial model training.
        This data is hardcoded and should be replaced by real, diverse data.
        """
        sample_data = {
            'revenue': [
                ("Revenue from operations LKR 1,234,567 thousand", 1234567.0),
                ("Total revenue amounted to 987,654 LKR thousands", 987654.0),
                ("Net sales revenue for the period 2,345,678", 2345678.0),
                ("Turnover during the year 3,456,789", 3456789.0),
                ("Operating revenue 4,567,890", 4567890.0),
                ("Revenue from continuing operations 5,678,901", 5678901.0),
                ("Total operating income 6,789,012", 6789012.0),
                ("Net revenue after discounts 7,890,123", 7890123.0),
                ("Gross sales revenue 8,901,234", 8901234.0),
                ("Revenue recognition 9,012,345", 9012345.0),
                ("Sales turnover 1,123,456", 1123456.0),
                ("Income from operations 2,234,567", 2234567.0),
                ("Total income 3,345,678", 3345678.0),
                ("Operating turnover 4,456,789", 4456789.0),
                ("Revenue for the financial year 5,567,890", 5567890.0)
            ],
            'profit_for_period': [
                ("Profit for the period LKR 123,456 thousand", 123456.0),
                ("Net income after tax 234,567", 234567.0),
                ("Profit after taxation 345,678", 345678.0),
                ("Net profit for the year 456,789", 456789.0),
                ("Earnings after tax 567,890", 567890.0),
                ("Profit attributable to shareholders 678,901", 678901.0),
                ("Net earnings 789,012", 789012.0),
                ("After-tax profit 890,123", 890123.0),
                ("Bottom line profit 901,234", 901234.0),
                ("Net income attributable to equity holders 112,345", 112345.0),
                ("Profit for the financial year 223,456", 223456.0),
                ("Net profit after all expenses 334,567", 334567.0),
                ("Comprehensive income 445,678", 445678.0),
                ("Retained earnings for the period 556,789", 556789.0),
                ("Profit after minority interest 667,890", 667890.0)
            ],
            'operating_profit': [
                ("Operating Profit amounted to 1,000,000", 1000000.0),
                ("EBIT was 950,000", 950000.0),
                ("Operational earnings 800,000", 800000.0),
                ("Profit before interest and tax 750,000", 750000.0),
                ("Core operating income 1,100,000", 1100000.0),
                ("Operating income 600,000", 600000.0),
                ("Profit from operations 700,000", 700000.0),
                ("Operating profit margin 0.15", 0.15), # Example of percentage, might need special handling
                ("Adjusted operating profit 1,200,000", 1200000.0),
                ("Operating loss (50,000)", -50000.0)
            ],
            'basic_eps': [
                ("Basic EPS was 2.55", 2.55),
                ("Diluted EPS 2.45", 2.45),
                ("Earnings Per Share 1.80", 1.80),
                ("EPS for the quarter 0.75", 0.75),
                ("Basic and diluted EPS 3.10", 3.10),
                ("EPS from continuing operations 2.90", 2.90),
                ("Weighted average basic EPS 1.50", 1.50),
                ("EPS (LKR) 0.90", 0.90),
                ("Basic earnings per share 4.00", 4.00),
                ("EPS (cents) 50", 0.50) # Example of cents, convert to LKR
            ],
            'total_assets': [
                ("Total Assets stood at 10,000,000", 10000000.0),
                ("Current and non-current assets 12,000,000", 12000000.0),
                ("Consolidated total assets 15,000,000", 15000000.0),
                ("Assets as at year-end 8,000,000", 8000000.0),
                ("Balance sheet total assets 11,500,000", 11500000.0),
                ("Total assets (LKR '000) 9,500,000", 9500000.0),
                ("Property, plant and equipment plus current assets 13,000,000", 13000000.0),
                ("Total assets increased to 14,000,000", 14000000.0),
                ("Assets (net of depreciation) 7,000,000", 7000000.0),
                ("Total assets including goodwill 16,000,000", 16000000.0)
            ],
            'total_liabilities': [
                ("Total Liabilities amounted to 5,000,000", 5000000.0),
                ("Current and non-current liabilities 6,000,000", 6000000.0),
                ("Consolidated total liabilities 7,500,000", 7500000.0),
                ("Liabilities as at year-end 4,500,000", 4500000.0),
                ("Balance sheet total liabilities 5,800,000", 5800000.0),
                ("Total liabilities (LKR '000) 4,900,000", 4900000.0),
                ("Long-term and short-term liabilities 6,200,000", 6200000.0),
                ("Total liabilities decreased to 5,500,000", 5500000.0),
                ("Liabilities and equity 8,000,000", 8000000.0),
                ("Total financial liabilities 7,000,000", 7000000.0)
            ],
            'total_equity': [
                ("Total Equity was 5,000,000", 5000000.0),
                ("Shareholders' equity 6,000,000", 6000000.0),
                ("Consolidated total equity 7,000,000", 7000000.0),
                ("Equity attributable to owners 4,500,000", 4500000.0),
                ("Balance sheet total equity 5,500,000", 5500000.0),
                ("Total equity (LKR '000) 4,800,000", 4800000.0),
                ("Retained earnings and capital 6,500,000", 6500000.0),
                ("Total equity increased to 7,200,000", 7200000.0),
                ("Equity and reserves 8,000,000", 8000000.0),
                ("Total equity including non-controlling interests 7,500,000", 7500000.0)
            ],
            'gross_profit': [
                ("Gross Profit amounted to 2,000,000", 2000000.0),
                ("Gross margin was 1,800,000", 1800000.0),
                ("Profit before operating expenses 2,100,000", 2100000.0),
                ("Gross profit from sales 1,900,000", 1900000.0),
                ("Total gross profit 2,200,000", 2200000.0)
            ],
            'ebitda': [
                ("EBITDA for the year was 1,500,000", 1500000.0),
                ("Earnings before interest, tax, depreciation and amortization 1,400,000", 1400000.0),
                ("Adjusted EBITDA 1,600,000", 1600000.0),
                ("Operating performance (EBITDA) 1,300,000", 1300000.0),
                ("Company's EBITDA 1,700,000", 1700000.0)
            ],
            'net_cash_flow': [
                ("Net Cash Flow from operations was 500,000", 500000.0),
                ("Net increase in cash and cash equivalents 450,000", 450000.0),
                ("Cash flow from operating activities 600,000", 600000.0),
                ("Total net cash flow 550,000", 550000.0),
                ("Net cash generated 480,000", 480000.0)
            ],
            'report_type': [
                ("Interim financial statements for the quarter", "Interim"),
                ("Annual report for the year ended 2023", "Annual"),
                ("Quarterly results announcement Q3", "Interim"),
                ("Full year financial statements", "Annual"),
                ("Half-yearly report", "Interim"),
                ("Consolidated annual report", "Annual"),
                ("First quarter report", "Interim"),
                ("Year-end financial statements", "Annual"),
                ("Semi-annual report", "Interim"),
                ("Annual general meeting report", "Annual")
            ],
            'audit_status': [
                ("These financial statements are unaudited.", "Unaudited"),
                ("The report has been audited by XYZ Auditors.", "Audited"),
                ("Figures are not audited.", "Unaudited"),
                ("Audited financial results.", "Audited"),
                ("Subject to audit.", "Unaudited"),
                ("Independent auditor's report attached.", "Audited"),
                ("Financial statements (unaudited).", "Unaudited"),
                ("Audited consolidated financial statements.", "Audited"),
                ("Not yet audited.", "Unaudited"),
                ("Audited by ABC & Co.", "Audited")
            ],
            'company_name': [
                ("Digital Mobility Solutions Lanka PLC", "Digital Mobility Solutions Lanka PLC"),
                ("PickMe", "PickMe"),
                ("ABC Limited", "ABC Limited"),
                ("XYZ Corporation", "XYZ Corporation"),
                ("Global Innovations Inc.", "Global Innovations Inc."),
                ("Tech Solutions Ltd.", "Tech Solutions Ltd."),
                ("Future Enterprises PLC", "Future Enterprises PLC"),
                ("Dynamic Holdings Co.", "Dynamic Holdings Co."),
                ("Pioneer Ventures Group", "Pioneer Ventures Group"),
                ("Apex Industries Limited", "Apex Industries Limited")
            ],
            'company_symbol': [
                ("PKME", "PKME"),
                ("ABC", "ABC"),
                ("XYZ", "XYZ"),
                ("GLIN", "GLIN"),
                ("TSL", "TSL"),
                ("FUE", "FUE"),
                ("DYN", "DYN"),
                ("PVG", "PVG"),
                ("APX", "APX"),
                ("LMN", "LMN")
            ]
        }
        
        return sample_data

class TrainingOrchestrator:
    """Orchestrates the training process for all ML models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_manager = TrainingDataManager(config)
        self.ml_manager = FinancialMLModelManager(model_dir=config.model_dir)
    
    def _get_last_training_timestamp(self, field_name: str) -> Optional[datetime]:
        """
        Retrieves the last training timestamp for a given field from a log or database.
        For simplicity, we'll assume a log file or a simple persistence mechanism.
        In a real system, this might be stored in the DB or as part of the model metadata.
        """
        log_file = Path(self.config.model_dir) / "training_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    if field_name in logs and 'last_trained' in logs[field_name]:
                        return datetime.fromisoformat(logs[field_name]['last_trained'])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading training log for {field_name}: {e}")
        return None

    def _update_last_training_timestamp(self, field_name: str, metrics: Dict[str, float]):
        """Updates the last training timestamp and performance metrics."""
        log_file = Path(self.config.model_dir) / "training_log.json"
        logs = {}
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode training log file {log_file}. Starting new log.")
        
        logs[field_name] = {
            'last_trained': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def _check_retrain_condition(self, field_name: str, current_samples: int) -> bool:
        """
        Determines if a model for a specific field needs retraining.
        Conditions:
        1. Not enough samples to train initially.
        2. Last trained time exceeds threshold.
        3. Model performance is below threshold (requires more advanced logging/evaluation).
        """
        if current_samples < self.config.min_samples_per_field:
            logger.info(f"Field '{field_name}': Insufficient samples ({current_samples}) for training. Skipping.")
            return False # Not enough data to train at all

        model_is_trained = self.ml_manager.get_model_status().get(field_name, False)
        
        if not model_is_trained:
            logger.info(f"Field '{field_name}': Model not yet trained. Initiating first training.")
            return True # Always train if not trained yet

        last_trained = self._get_last_training_timestamp(field_name)
        if last_trained is None:
            logger.info(f"Field '{field_name}': No last trained timestamp found. Retraining.")
            return True # Retrain if no timestamp (e.g., first run or log cleared)

        time_since_last_train = datetime.now() - last_trained
        if time_since_last_train > timedelta(days=self.config.retrain_threshold_days):
            logger.info(f"Field '{field_name}': Retraining due to time threshold ({time_since_last_train.days} days).")
            return True # Retrain if past threshold

        # TODO: Add logic to check model performance against self.config.performance_threshold
        # This would require loading historical performance metrics from the training_log.json
        # and comparing them. For now, we rely on time/sample count.

        logger.info(f"Field '{field_name}': No retraining needed (trained {time_since_last_train.days} days ago).")
        return False

    def run_training_cycle(self, force_train: bool = False, use_sample_data: bool = False):
        """
        Executes a full training cycle for all relevant ML models.
        
        Args:
            force_train: If True, forces retraining of all models regardless of conditions.
            use_sample_data: If True, uses hardcoded sample data instead of DB feedback.
        """
        logger.info("Starting ML training cycle.")
        
        if use_sample_data:
            training_data_dict = self.data_manager.create_sample_training_data()
            logger.info("Using hardcoded sample training data.")
        else:
            training_data_dict = self.data_manager.load_training_data_from_db()
            logger.info(f"Loaded training data from database. Fields with data: {list(training_data_dict.keys())}")
            
            # Optionally, load from files and merge (if you have manually curated datasets)
            file_data = self.data_manager.load_training_data_from_files()
            for field, data in file_data.items():
                if field not in training_data_dict:
                    training_data_dict[field] = []
                training_data_dict[field].extend(data)
                logger.info(f"Merged {len(data)} samples for {field} from files.")


        fields_to_train = []
        for field_name, data in training_data_dict.items():
            if force_train or self._check_retrain_condition(field_name, len(data)):
                fields_to_train.append(field_name)
            else:
                logger.info(f"Skipping training for {field_name} based on conditions.")

        if not fields_to_train:
            logger.info("No models need training at this time.")
            return

        logger.info(f"Models scheduled for training: {fields_to_train}")
        
        for field_name in fields_to_train:
            data = training_data_dict.get(field_name)
            if data:
                logger.info(f"Training model for {field_name}...")
                try:
                    metrics = self.ml_manager.models[field_name].train(
                        texts=[item[0] for item in data],
                        labels=[item[1] for item in data],
                        validation_split=self.config.validation_split
                    )
                    self.ml_manager.models[field_name].save(str(self.config.model_dir))
                    self._update_last_training_timestamp(field_name, metrics)
                    logger.info(f"Successfully trained and saved model for {field_name}. Metrics: {metrics}")
                except ValueError as ve:
                    logger.error(f"Training error for {field_name}: {ve}. Skipping this model.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during training for {field_name}: {e}. Skipping.")
            else:
                logger.warning(f"No data found for {field_name} despite being scheduled. Skipping.")

        logger.info("ML training cycle completed.")

def main():
    parser = argparse.ArgumentParser(description="ML Training Script for Financial Data Extraction.")
    parser.add_argument('--simulate-data', action='store_true',
                        help='Generate and use hardcoded sample training data instead of database/files.')
    parser.add_argument('--force-train', action='store_true',
                        help='Force retraining of all models regardless of conditions.')
    parser.add_argument('--model-dir', type=str, default="ml_models",
                        help='Directory to save/load ML models.')
    parser.add_argument('--db-path', type=str, default="extraction_learning.db",
                        help='Path to the SQLite feedback database.')
    
    args = parser.parse_args()

    config = TrainingConfig(
        model_dir=args.model_dir,
        db_path=args.db_path
    )

    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_training_cycle(force_train=args.force_train, use_sample_data=args.simulate_data)

if __name__ == "__main__":
    main()