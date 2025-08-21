import os
import numpy as np
import pandas as pd
import pickle
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import re
from datetime import datetime
from dataclasses import dataclass

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, f1_score, classification_report
from sklearn.pipeline import Pipeline
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
    """ML model for extracting specific financial fields"""
    
    def __init__(self, field_name: str, field_type: str = "numeric"):
        self.field_name = field_name
        self.field_type = field_type  # 'numeric' or 'categorical'
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False
        
    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text"""
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
            'table_indicator': 1 if any(word in text.lower() for word in ['total', 'amount', 'balance']) else 0,
            'financial_terms': len(re.findall(r'\b(revenue|profit|loss|assets|liabilities|equity|eps|ebitda)\b', text.lower())),
            'date_patterns': len(re.findall(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', text)),
            'currency_amounts': len(re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        return features
    
    def prepare_features(self, texts: List[str]) -> np.ndarray:
        """Prepare features for training/prediction"""
        # Text vectorization
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            text_features = self.vectorizer.fit_transform(texts)
        else:
            text_features = self.vectorizer.transform(texts)
        
        # Numerical features
        numerical_features = []
        for text in texts:
            features = self._extract_features(text)
            numerical_features.append(list(features.values()))
        
        numerical_features = np.array(numerical_features)
        
        # Combine features
        if hasattr(self.vectorizer, 'vocabulary_'):
            combined_features = np.hstack([text_features.toarray(), numerical_features])
        else:
            combined_features = numerical_features
            
        return combined_features
        
    def train(self, texts: List[str], labels: List[Any], validation_split: float = 0.2) -> Dict[str, float]:
        """Train the model"""
        logger.info(f"Training model for {self.field_name} with {len(texts)} samples")
        
        # Prepare features
        features = self.prepare_features(texts)
        
        # Handle labels based on field type
        if self.field_type == "numeric":
            # Clean and convert labels to float
            processed_labels = []
            valid_indices = []
            
            for i, label in enumerate(labels):
                try:
                    if label is not None and label != "":
                        # Clean financial values
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
            
            if len(processed_labels) < 5:
                raise ValueError(f"Not enough valid labels for {self.field_name}: {len(processed_labels)}")
            
            features = features[valid_indices]
            labels = np.array(processed_labels)
            
            # Choose regression model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
        else:  # categorical
            # Encode categorical labels
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform([str(label) for label in labels])
            else:
                labels = self.label_encoder.transform([str(label) for label in labels])
            
            # Choose classification model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42
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
            f1 = f1_score(y_val, y_pred, average='weighted')
            metrics = {'accuracy': accuracy, 'f1_score': f1}
            logger.info(f"Model {self.field_name} - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
        
        self.is_trained = True
        return metrics
    
    def predict(self, text: str) -> Tuple[Any, float]:
        """Make prediction"""
        if not self.is_trained:
            return None, 0.0
        
        try:
            features = self.prepare_features([text])
            features = self.scaler.transform(features)
            
            prediction = self.model.predict(features)[0]
            
            # Calculate confidence
            confidence = 0.5  # Default
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
            elif hasattr(self.model, 'score'):
                # For regression, use a simple heuristic
                confidence = min(0.9, max(0.1, 1.0 / (1.0 + abs(prediction) / 1000000)))
            
            # Decode if categorical
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
        
        # Define field configurations
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
    
    def train_model(self, field_name: str, training_data: List[Tuple[str, Any]]) -> bool:
        """Train a specific model"""
        if field_name not in self.models:
            logger.error(f"Unknown field: {field_name}")
            return False
        
        if len(training_data) < 10:
            logger.warning(f"Insufficient training data for {field_name}: {len(training_data)}")
            return False
        
        texts, labels = zip(*training_data)
        
        try:
            metrics = self.models[field_name].train(list(texts), list(labels))
            self.models[field_name].save(str(self.model_dir))
            logger.info(f"Successfully trained model for {field_name}: {metrics}")
            return True
        except Exception as e:
            logger.error(f"Failed to train model for {field_name}: {e}")
            return False
    
    def train_all_models(self, training_data_dict: Dict[str, List[Tuple[str, Any]]]) -> Dict[str, bool]:
        """Train all models"""
        results = {}
        for field_name, training_data in training_data_dict.items():
            results[field_name] = self.train_model(field_name, training_data)
        return results
    
    def predict(self, field_name: str, text: str) -> Tuple[Any, float]:
        """Make prediction for a field"""
        if field_name not in self.models:
            return None, 0.0
        
        return self.models[field_name].predict(text)
    
    def predict_all(self, text: str) -> Dict[str, Tuple[Any, float]]:
        """Make predictions for all fields"""
        predictions = {}
        for field_name in self.field_configs.keys():
            predictions[field_name] = self.predict(field_name, text)
        return predictions
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of all models"""
        return {field_name: model.is_trained for field_name, model in self.models.items()}
    
    def save_all_models(self):
        """Save all trained models"""
        for field_name, model in self.models.items():
            if model.is_trained:
                model.save(str(self.model_dir))

class TextClassificationModel:
    """Specialized model for text classification tasks"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_trained = False
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Train classification model"""
        # Prepare features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        features = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Split and evaluate
        X_train, X_val, y_train, y_val = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        self.is_trained = True
        return {'accuracy': accuracy, 'f1_score': f1}
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict class and confidence"""
        if not self.is_trained:
            return None, 0.0
        
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features)[0])
        
        label = self.label_encoder.inverse_transform([prediction])[0]
        return label, confidence

class RegressionModel:
    """Specialized model for regression tasks"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.is_trained = False
    
    def train(self, texts: List[str], values: List[float]) -> Dict[str, float]:
        """Train regression model"""
        # Prepare features
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        features = self.vectorizer.fit_transform(texts)
        
        # Scale features
        self.scaler = StandardScaler(with_mean=False)
        features = self.scaler.fit_transform(features)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        
        # Split and evaluate
        X_train, X_val, y_train, y_val = train_test_split(
            features, values, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        
        self.is_trained = True
        return {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}
    
    def predict(self, text: str) -> Tuple[float, float]:
        """Predict value and confidence"""
        if not self.is_trained:
            return None, 0.0
        
        features = self.vectorizer.transform([text])
        features = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        
        # Simple confidence heuristic for regression
        confidence = min(0.9, max(0.1, 1.0 / (1.0 + abs(prediction) / 1000000)))
        
        return prediction, confidence

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = FinancialMLModelManager()
    
    # Example training data (in practice, this would come from your database)
    sample_data = {
        'revenue': [
            ("Revenue from operations 1,234,567", 1234567.0),
            ("Total revenue 987,654", 987654.0),
            ("Net sales revenue 2,345,678", 2345678.0)
        ],
        'report_type': [
            ("Interim financial statements for the quarter", "Interim"),
            ("Annual report for the year ended", "Annual"),
            ("Quarterly results announcement", "Interim")
        ]
    }
    
    # Train models
    results = manager.train_all_models(sample_data)
    print("Training results:", results)
    
    # Make predictions
    test_text = "Revenue from continuing operations amounted to 5,678,901"
    prediction, confidence = manager.predict('revenue', test_text)
    print(f"Revenue prediction: {prediction}, confidence: {confidence}")