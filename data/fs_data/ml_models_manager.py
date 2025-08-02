import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import re
from datetime import datetime
from dataclasses import dataclass

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    field_name: str
    model_params: Dict[str, Any]
    vectorizer_params: Dict[str, Any]
    scaler_params: Dict[str, Any]

class FinancialFieldExtractor:
    """
    ML model for extracting specific financial fields, now with support
    for layout-aware features.
    """
    
    def __init__(self, field_name: str, field_type: str = "numeric"):
        self.field_name = field_name
        self.field_type = field_type  # 'numeric' or 'categorical'
        self.model = None
        self.vectorizer = None # Will be fitted during train
        self.scaler = None     # Will be fitted during train
        self.label_encoder = None
        self.is_trained = False
        
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text content."""
        features = {
            'text_length': len(text),
            'line_count': text.count('\n'),
            'number_count': len(re.findall(r'\d+', text)),
            'word_count': len(text.split()),
            'parentheses_count': text.count('('),
            'percentage_count': text.count('%'),
            'dollar_count': text.count('$'),
            'lkr_count': text.count('LKR'),
            'usd_count': text.count('USD'),
            'comma_count': text.count(','),
            'decimal_count': text.count('.'),
            'negative_indicator': 1 if '(' in text and ')' in text else 0,
            'table_indicator': 1 if any(word in text.lower() for word in ['total', 'amount', 'balance', 'sum', 'statement', 'table']) else 0,
            'financial_terms': len(re.findall(r'\b(revenue|profit|loss|assets|liabilities|equity|eps|ebitda|income|expenses|cash|flow)\b', text.lower())),
            'date_patterns': len(re.findall(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', text)),
            'currency_amounts': len(re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        return features

    def _extract_visual_features(self, structured_text_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extracts features based on the visual layout and structure from structured_text_data.
        This method processes a list of text blocks with their bounding boxes,
        font info, etc., to derive features.
        """
        if not structured_text_data:
            return {
                'avg_font_size': 0.0,
                'has_bold_text': 0,
                'is_top_of_page': 0,
                'is_table_cell': 0, 
                'x_position_avg': 0.0, 
                'y_position_avg': 0.0, 
                'num_lines_in_block': 0,
                'max_font_size': 0.0,
                'min_x_position': 0.0,
                'max_x_position': 0.0,
                'min_y_position': 0.0,
                'max_y_position': 0.0,
            }

        font_sizes = [block.get('font_size', 0) for block in structured_text_data if 'font_size' in block]
        avg_font_size = np.mean(font_sizes) if font_sizes else 0.0
        max_font_size = np.max(font_sizes) if font_sizes else 0.0

        has_bold_text = 1 if any(block.get('is_bold', False) for block in structured_text_data) else 0
        
        # Bounding box features (assuming 'bbox' is [x0, y0, x1, y1])
        x_positions = [block['bbox'][0] for block in structured_text_data if 'bbox' in block and block['bbox']]
        y_positions = [block['bbox'][1] for block in structured_text_data if 'bbox' in block and block['bbox']]
        
        x_position_avg = np.mean(x_positions) if x_positions else 0.0
        y_position_avg = np.mean(y_positions) if y_positions else 0.0

        min_x_position = np.min(x_positions) if x_positions else 0.0
        max_x_position = np.max(x_positions) if x_positions else 0.0
        min_y_position = np.min(y_positions) if y_positions else 0.0
        max_y_position = np.max(y_positions) if y_positions else 0.0

        # Heuristic for top of page (e.g., if y0 is very low, meaning close to top)
        is_top_of_page = 1 if y_position_avg < 100 and y_position_avg > 0 else 0 # Assuming page height ~800 units

        # This would require actual table detection logic. For now, a simple keyword check.
        combined_text_for_table_check = " ".join([block.get('text', '') for block in structured_text_data]).lower()
        is_table_cell = 1 if any(term in combined_text_for_table_check for term in ['total', 'amount', 'balance', 'sum', 'statement', 'table']) else 0

        num_lines_in_block = len(set(block.get('line_id') for block in structured_text_data if 'line_id' in block))

        return {
            'avg_font_size': avg_font_size,
            'has_bold_text': has_bold_text,
            'is_top_of_page': is_top_of_page,
            'is_table_cell': is_table_cell,
            'x_position_avg': x_position_avg,
            'y_position_avg': y_position_avg,
            'num_lines_in_block': num_lines_in_block,
            'max_font_size': max_font_size,
            'min_x_position': min_x_position,
            'max_x_position': max_x_position,
            'min_y_position': min_y_position,
            'max_y_position': max_y_position,
        }
    
    def prepare_features(self, document_contexts: List[Union[str, List[Dict[str, Any]]]]) -> np.ndarray:
        """
        Prepare features for prediction. This method assumes self.vectorizer and self.scaler
        are already fitted.
        """
        if self.vectorizer is None or self.scaler is None:
            raise RuntimeError("Vectorizer or Scaler not fitted. Call train() first or load a trained model.")

        raw_texts = []
        visual_features_list = []

        for context in document_contexts:
            if isinstance(context, str):
                raw_texts.append(context)
                visual_features_list.append(self._extract_visual_features([]))
            elif isinstance(context, list) and all(isinstance(item, dict) for item in context):
                combined_text = " ".join([block.get('text', '') for block in context])
                raw_texts.append(combined_text)
                visual_features_list.append(self._extract_visual_features(context))
            else:
                logger.warning(f"Unexpected document context format. Skipping: {type(context)}")
                raw_texts.append("")
                visual_features_list.append(self._extract_visual_features([]))

        # Transform text features using the fitted vectorizer
        text_features = self.vectorizer.transform(raw_texts)
        
        # Numerical (text-derived) features
        numerical_text_features = []
        for text in raw_texts:
            features = self._extract_text_features(text)
            numerical_text_features.append(list(features.values()))
        numerical_text_features = np.array(numerical_text_features)

        # Visual features
        numerical_visual_features = pd.DataFrame(visual_features_list).values
        
        # Combine all features
        combined_features = np.hstack([
            text_features.toarray(),
            numerical_text_features,
            numerical_visual_features
        ])
        
        # Scale combined features using the fitted scaler
        scaled_features = self.scaler.transform(combined_features)
            
        return scaled_features
        
    def train(self, document_contexts: List[Union[str, List[Dict[str, Any]]]], labels: List[Any], validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the model. This method will re-initialize and re-fit the vectorizer and scaler.
        `document_contexts` can be a list of raw strings or structured data.
        """
        logger.info(f"Training model for {self.field_name} with {len(document_contexts)} samples")
        
        if len(document_contexts) == 0:
            raise ValueError(f"No document contexts provided for training {self.field_name}.")
        
        # --- Feature Preparation for Training (FIT AND TRANSFORM) ---
        
        raw_texts_for_fitting = []
        visual_features_for_fitting_list = []

        for context in document_contexts:
            if isinstance(context, str):
                raw_texts_for_fitting.append(context)
                visual_features_for_fitting_list.append(self._extract_visual_features([]))
            elif isinstance(context, list) and all(isinstance(item, dict) for item in context):
                combined_text = " ".join([block.get('text', '') for block in context])
                raw_texts_for_fitting.append(combined_text)
                visual_features_for_fitting_list.append(self._extract_visual_features(context))
            else:
                logger.warning(f"Unexpected document context format during training. Skipping: {type(context)}")
                raw_texts_for_fitting.append("")
                visual_features_for_fitting_list.append(self._extract_visual_features([]))

        # Re-initialize and fit TfidfVectorizer on the current training texts
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        text_features = self.vectorizer.fit_transform(raw_texts_for_fitting)

        # Extract numerical text features
        numerical_text_features = []
        for text in raw_texts_for_fitting:
            features = self._extract_text_features(text)
            numerical_text_features.append(list(features.values()))
        numerical_text_features = np.array(numerical_text_features)

        # Extract visual features
        numerical_visual_features = pd.DataFrame(visual_features_for_fitting_list).values
        
        # Combine all features before scaling
        combined_features = np.hstack([
            text_features.toarray(),
            numerical_text_features,
            numerical_visual_features
        ])

        # Re-initialize and fit StandardScaler on the combined features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(combined_features)
        
        # --- Label Processing ---
        
        # Handle labels based on field type
        if self.field_type == "numeric":
            processed_labels = []
            valid_indices = []
            
            for i, label in enumerate(labels):
                try:
                    if label is not None and label != "":
                        if isinstance(label, str):
                            cleaned = label.replace(',', '').replace(' ', '')
                            if '(' in cleaned and ')' in cleaned:
                                cleaned = '-' + cleaned.replace('(', '').replace(')', '')
                            cleaned = re.sub(r'[^\d\.\-]', '', cleaned)
                            if cleaned and cleaned != '-':
                                processed_labels.append(float(cleaned))
                                valid_indices.append(i)
                        else:
                            processed_labels.append(float(label))
                            valid_indices.append(i)
                except (ValueError, TypeError):
                    continue
            
            if len(processed_labels) < 2: # Minimum 2 samples for regression
                raise ValueError(f"Not enough valid labels for {self.field_name} after cleaning: {len(processed_labels)}. Need at least 2.")
            
            # Ensure features align with valid labels
            features_for_model = features_scaled[valid_indices]
            labels_for_model = np.array(processed_labels)
            
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
        else:  # categorical
            valid_labels = [str(label) for label in labels if label is not None and str(label).strip() != '']
            valid_indices = [i for i, label in enumerate(labels) if label is not None and str(label).strip() != '']

            if len(valid_labels) < 2: # Minimum 2 samples for classification
                 raise ValueError(f"Not enough valid labels for {self.field_name} after cleaning: {len(valid_labels)}. Need at least 2.")

            features_for_model = features_scaled[valid_indices]
            
            self.label_encoder = LabelEncoder()
            labels_for_model = self.label_encoder.fit_transform(valid_labels)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features_for_model, labels_for_model, test_size=validation_split, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        
        metrics = {}
        if self.field_type == "numeric":
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            metrics = {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}
            logger.info(f"Model {self.field_name} - MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}")
        else:
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            metrics = {'accuracy': accuracy, 'f1_score': f1}
            logger.info(f"Model {self.field_name} - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
        
        self.is_trained = True
        return metrics
    
    def predict(self, document_context: Union[str, List[Dict[str, Any]]]) -> Tuple[Any, float]:
        """Make prediction"""
        if not self.is_trained:
            logger.warning(f"Model for {self.field_name} is not trained. Cannot predict.")
            return None, 0.0
        
        try:
            features = self.prepare_features([document_context]) # prepare_features now handles transformation
            
            prediction = self.model.predict(features)[0]
            
            confidence = 0.5
            if hasattr(self.model, 'predict_proba') and self.field_type == "categorical":
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
            elif self.field_type == "numeric":
                # For regression, a simple heuristic for confidence based on prediction magnitude
                # This is a placeholder; more sophisticated confidence for regression is complex
                confidence = min(0.9, max(0.1, 1.0 / (1.0 + abs(prediction) / 1000000)))
            
            if self.field_type == "categorical" and self.label_encoder:
                prediction = self.label_encoder.inverse_transform([int(prediction)])[0]
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error for {self.field_name}: {e}")
            return None, 0.0
    
    def save(self, model_dir: str):
        """Save model to disk"""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        if self.is_trained:
            joblib.dump(self.model, model_dir / f"{self.field_name}_model.joblib")
            joblib.dump(self.vectorizer, model_dir / f"{self.field_name}_vectorizer.joblib")
            joblib.dump(self.scaler, model_dir / f"{self.field_name}_scaler.joblib")
            
            if self.label_encoder:
                joblib.dump(self.label_encoder, model_dir / f"{self.field_name}_encoder.joblib")
            
            logger.info(f"Model for {self.field_name} saved to {model_dir}")
    
    def load(self, model_dir: str) -> bool:
        """Load model from disk"""
        model_dir = Path(model_dir)
        
        try:
            self.model = joblib.load(model_dir / f"{self.field_name}_model.joblib")
            self.vectorizer = joblib.load(model_dir / f"{self.field_name}_vectorizer.joblib")
            self.scaler = joblib.load(model_dir / f"{self.field_name}_scaler.joblib")
            
            encoder_path = model_dir / f"{self.field_name}_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            self.is_trained = True
            logger.info(f"Model for {self.field_name} loaded from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for {self.field_name}: {e}")
            return False

class FinancialMLModelManager:
    """Manager for all financial extraction ML models"""
    
    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        
        self.field_configs = {
            'revenue': {'type': 'numeric', 'importance': 'high'},
            'profit_for_period': {'type': 'numeric', 'importance': 'high'},
            'operating_profit': {'type': 'numeric', 'importance': 'high'},
            'basic_eps': {'type': 'numeric', 'importance': 'high'},
            'total_assets': {'type': 'numeric', 'importance': 'medium'},
            'total_liabilities': {'type': 'numeric', 'importance': 'medium'},
            'total_equity': {'type': 'numeric', 'importance': 'medium'},
            'gross_profit': {'type': 'numeric', 'importance': 'medium'},
            'ebitda': {'type': 'numeric', 'importance': 'medium'},
            'net_cash_flow': {'type': 'numeric', 'importance': 'medium'},
            'report_type': {'type': 'categorical', 'importance': 'low'},
            'audit_status': {'type': 'categorical', 'importance': 'low'},
            'company_name': {'type': 'categorical', 'importance': 'low'},
            'company_symbol': {'type': 'categorical', 'importance': 'low'}
        }
        
        self._initialize_models()
        self._load_existing_models()
    
    def _initialize_models(self):
        """Initialize models for each field"""
        for field_name, config in self.field_configs.items():
            self.models[field_name] = FinancialFieldExtractor(
                field_name=field_name,
                field_type=config['type']
            )
    
    def _load_existing_models(self):
        """Load existing trained models"""
        for field_name in self.field_configs.keys():
            model_path = self.model_dir / f"{field_name}_model.joblib"
            if model_path.exists():
                self.models[field_name].load(str(self.model_dir))
    
    def train_model(self, field_name: str, training_data: List[Tuple[Union[str, List[Dict[str, Any]]], Any]]) -> bool:
        """
        Train a specific model.
        `training_data` now expects tuples where the first element can be
        raw text or structured text data.
        """
        if field_name not in self.models:
            logger.error(f"Unknown field: {field_name}")
            return False
        
        if len(training_data) < 2: # Changed from 10 to 2 for minimal sample data training
            logger.warning(f"Insufficient training data for {field_name}: {len(training_data)}. Need at least 2 samples.")
            return False
        
        document_contexts, labels = zip(*training_data)
        
        try:
            metrics = self.models[field_name].train(list(document_contexts), list(labels))
            self.models[field_name].save(str(self.model_dir))
            logger.info(f"Successfully trained model for {field_name}: {metrics}")
            return True
        except ValueError as ve:
            logger.error(f"Training error for {field_name}: {ve}. Skipping this model.")
            return False
        except Exception as e:
            logger.error(f"Failed to train model for {field_name}: {e}")
            return False
    
    def train_all_models(self, training_data_dict: Dict[str, List[Tuple[Union[str, List[Dict[str, Any]]], Any]]]) -> Dict[str, bool]:
        """Train all models"""
        results = {}
        for field_name, training_data in training_data_dict.items():
            results[field_name] = self.train_model(field_name, training_data)
        return results
    
    def predict(self, field_name: str, document_context: Union[str, List[Dict[str, Any]]]) -> Tuple[Any, float]:
        """Make prediction for a field"""
        if field_name not in self.models:
            return None, 0.0
        
        return self.models[field_name].predict(document_context)
    
    def predict_all(self, document_context: Union[str, List[Dict[str, Any]]]) -> Dict[str, Tuple[Any, float]]:
        """Make predictions for all fields"""
        predictions = {}
        for field_name in self.field_configs.keys():
            predictions[field_name] = self.predict(field_name, document_context)
        return predictions
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of all models"""
        return {field_name: model.is_trained for field_name, model in self.models.items()}
    
    def save_all_models(self):
        """Save all trained models"""
        for field_name, model in self.models.items():
            if model.is_trained:
                model.save(str(self.model_dir))
