import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Import our modular components
from data.data_loader import load_stock_data, load_financial_data
from data.feature_engineering import FeatureEngineer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_DIR = 'models'
SCALER_DIR = 'scalers'
RESULTS_DIR = 'results'

def create_directories():
    """Create necessary directories"""
    for directory in [MODEL_DIR, SCALER_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_model_configs():
    """Define different model configurations to try"""
    return {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            },
            'scale_features': False
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'scale_features': False
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'scale_features': True
        }
    }

def handle_class_imbalance(y):
    """Calculate class weights to handle imbalanced data"""
    unique_classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    return dict(zip(unique_classes, class_weights))

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print(f"\nClassification Report:")
    target_names = ['Sell', 'Hold', 'Buy'] if len(np.unique(y_test)) == 3 else None
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        # Save detailed feature importance
        feature_importance.to_csv(
            os.path.join(RESULTS_DIR, f'{model_name}_feature_importance.csv'), 
            index=False
        )
        print(f"💾 Feature importance saved to: {RESULTS_DIR}/{model_name}_feature_importance.csv")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    }

def train_single_model(model_config, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    print(f"\n🔄 Training {model_name}...")
    
    # Handle feature scaling if required
    scaler = None
    if model_config['scale_features']:
        print(f"   🔄 Scaling features for {model_name}...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_file = os.path.join(SCALER_DIR, f'{model_name}_scaler.pkl')
        joblib.dump(scaler, scaler_file)
        print(f"   💾 Scaler saved to: {scaler_file}")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Handle class imbalance
    class_weights = handle_class_imbalance(y_train)
    print(f"   ⚖️ Class weights: {class_weights}")
    
    # Set class weights if the model supports it
    if hasattr(model_config['model'], 'class_weight'):
        model_config['model'].set_params(class_weight=class_weights)
    
    # Grid search for hyperparameter tuning
    print(f"   🔍 Performing hyperparameter tuning for {model_name}...")
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=3,  # 3-fold cross-validation
        scoring='f1_weighted',  # Use F1 score for imbalanced data
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"   ✅ Best parameters: {grid_search.best_params_}")
    print(f"   ✅ Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate the model
    results = evaluate_model(best_model, X_test_scaled, y_test, model_name, X_train.columns.tolist())
    
    # Save the model
    model_file = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
    joblib.dump(best_model, model_file)
    print(f"   💾 Model saved to: {model_file}")
    
    return {
        'model': best_model,
        'scaler': scaler,
        'results': results,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def compare_models(all_results):
    """Compare all trained models"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    for model_name, result in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'CV F1 Score': f"{result['cv_score']:.4f}",
            'Test Accuracy': f"{result['results']['accuracy']:.4f}",
            'Test F1 Score': f"{result['results']['f1_score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test F1 Score', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Model comparison saved to: {comparison_file}")
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n🏆 Best performing model: {best_model_name}")
    print(f"🎯 Test F1 Score: {comparison_df.iloc[0]['Test F1 Score']}")
    print(f"🎯 Test Accuracy: {comparison_df.iloc[0]['Test Accuracy']}")
    
    return best_model_name, comparison_df

def save_training_summary(engineer, X, y, metadata, all_results, best_model_name):
    """Save a comprehensive training summary"""
    
    # Get feature groups for analysis
    feature_groups = engineer.get_feature_importance_groups()
    
    summary_file = os.path.join(RESULTS_DIR, 'training_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("STOCK PREDICTION MODEL TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {len(X)} samples\n")
        f.write(f"Number of Features: {len(X.columns)}\n")
        f.write(f"Number of Companies: {metadata['company_id'].nunique()}\n")
        f.write(f"Date Range: {metadata['trade_date'].min()} to {metadata['trade_date'].max()}\n\n")
        
        f.write("LABEL DISTRIBUTION:\n")
        label_counts = y.value_counts().sort_index()
        label_names = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
        for label, count in label_counts.items():
            f.write(f"  {label_names.get(label, label)}: {count} ({count/len(y)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("FEATURE GROUPS:\n")
        for group_name, features in feature_groups.items():
            f.write(f"  {group_name}: {len(features)} features\n")
        f.write("\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"Best Model: {best_model_name}\n")
        for model_name, result in all_results.items():
            f.write(f"  {model_name}:\n")
            f.write(f"    CV F1 Score: {result['cv_score']:.4f}\n")
            f.write(f"    Test Accuracy: {result['results']['accuracy']:.4f}\n")
            f.write(f"    Test F1 Score: {result['results']['f1_score']:.4f}\n")
        f.write("\n")
        
        f.write("TOP 10 FEATURES (from best model):\n")
        best_result = all_results[best_model_name]
        if hasattr(best_result['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                f.write(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}\n")
    
    print(f"💾 Training summary saved to: {summary_file}")

def train():
    """Main training function"""
    print("🚀 Starting Stock Prediction Model Training")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load data
    print("\n📊 Loading data...")
    stock_data = load_stock_data()
    financial_data = load_financial_data()
    
    if stock_data.empty:
        print("❌ No stock data available. Please check your database connection and data.")
        return
    
    print(f"✅ Loaded stock data: {len(stock_data)} records")
    print(f"✅ Companies with stock data: {stock_data['company_id'].nunique()}")
    
    if not financial_data.empty:
        print(f"✅ Loaded financial data: {len(financial_data)} records")
        print(f"✅ Companies with financial data: {financial_data['company_id'].nunique()}")
    else:
        print("⚠️ No financial data available. Using technical analysis only.")
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    engineer = FeatureEngineer()
    X, y, metadata = engineer.create_features_and_labels(stock_data, financial_data)
    
    if len(X) == 0:
        print("❌ No data available after feature engineering. Please check your data and preprocessing.")
        return
    
    # Save feature information
    engineer.save_feature_info(os.path.join(RESULTS_DIR, 'feature_info.txt'))
    
    # --- ADD THIS LINE ---
    # Save the list of training features to a JSON file
    # NEW (Correct):
    engineer.save_training_features(os.path.join(MODEL_DIR, 'training_features.json'))
    print(f"💾 Training features list saved to: {MODEL_DIR}/training_features.json") # Keep the print
    # ---------------------
    
    # Split dataset
    print(f"\n📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train labels: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Test labels: {y_test.value_counts().sort_index().to_dict()}")
    
    # Train models
    print(f"\n🤖 Training models...")
    model_configs = get_model_configs()
    all_results = {}
    
    for model_name, config in model_configs.items():
        try:
            result = train_single_model(config, X_train, X_test, y_train, y_test, model_name)
            all_results[model_name] = result
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    if not all_results:
        print("❌ No models were successfully trained.")
        return
    
    # Compare models
    print("\n📈 Comparing models...")
    best_model_name, comparison_df = compare_models(all_results)
    
    # Save training summary
    print("\n💾 Saving training summary...")
    save_training_summary(engineer, X, y, metadata, all_results, best_model_name)
    
    # Final recommendations
    print(f"\n🎉 Training Complete!")
    print(f"📁 Results saved in: {RESULTS_DIR}/")
    print(f"📁 Models saved in: {MODEL_DIR}/")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📊 Use '{best_model_name}_model.pkl' for predictions")
    
    if all_results[best_model_name]['scaler']:
        print(f"🔧 Remember to use '{best_model_name}_scaler.pkl' for feature scaling")

def quick_train():
    """Quick training with just Random Forest for faster testing"""
    print("🚀 Quick Training Mode - Random Forest Only")
    print("="*50)
    
    create_directories()
    
    # Load data
    stock_data = load_stock_data()
    financial_data = load_financial_data()
    
    if stock_data.empty:
        print("❌ No stock data available.")
        return
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y, metadata = engineer.create_features_and_labels(stock_data, financial_data)
    
    if len(X) == 0:
        print("❌ No data available after feature engineering.")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest only
    rf_config = {
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'params': {'max_depth': [10, 20], 'min_samples_split': [2, 5]},
        'scale_features': False
    }
    
    result = train_single_model(rf_config, X_train, X_test, y_train, y_test, 'random_forest_quick')
    
    print(f"\n🎉 Quick training complete!")
    print(f"📊 Accuracy: {result['results']['accuracy']:.4f}")
    print(f"📊 F1 Score: {result['results']['f1_score']:.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_train()
    else:
        train()