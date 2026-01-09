"""
Model Pipeline Module
Handles model training, evaluation, and selection.
"""
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path  # <-- ADD THIS IMPORT
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import joblib
from collections import Counter

from .hyperparameter_tuner import HyperparameterTuner
from .analytics_pipeline import EnhancedMetricsCalculator

logger = logging.getLogger(__name__)

class ModelPipeline:
    """Complete model training and evaluation pipeline"""
    
    def __init__(self, test_size=0.2, random_state=42, cv_folds=5):
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = {}
        self.results = {}
        self.cv_scores = {}
        self.detailed_metrics = {}
        self.best_model_name = None
        self.best_model = None
        
    def train_test_split(self, X, y):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        logger.info(f"[DATA] Train set: {len(X_train)} samples")
        logger.info(f"[DATA] Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test, model_configs):
        """Train multiple models with comprehensive evaluation."""
        self.results = {}
        self.models = {}
        self.cv_scores = {}
        self.detailed_metrics = {}
        
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, config in model_configs.items():
            logger.info(f"\n  [TRAINING] Training {name}...")
            
            # Create and train model
            model = config['class'](**config['params'])
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred) * 100
            metrics_calculator = EnhancedMetricsCalculator()
            metrics = metrics_calculator.calculate_comprehensive_metrics(
                y_test, y_pred, y_proba, sorted(set(y_test))
            )
            
            # Cross-validation
            cv_acc = cross_val_score(model, np.vstack([X_train, X_test]), 
                                    np.concatenate([y_train, y_test]), 
                                    cv=kf, scoring='accuracy', n_jobs=-1)
            cv_mean = cv_acc.mean() * 100
            cv_std = cv_acc.std() * 100
            
            # Store results
            self.results[name] = acc
            self.models[name] = model
            self.cv_scores[name] = {
                'mean': cv_mean, 
                'std': cv_std, 
                'scores': cv_acc.tolist(),
                'stability': 'High' if cv_std < 5 else 'Moderate' if cv_std < 10 else 'Low'
            }
            self.detailed_metrics[name] = metrics
            
            logger.info(f"     [SUCCESS] Test Accuracy: {acc:.2f}%")
            logger.info(f"     [DATA] CV Accuracy: {cv_mean:.2f}% ± {cv_std:.2f}%")
            logger.info(f"     [INSIGHT] Performance Tier: {metrics['performance_interpretation']['overall_performance_tier']}")
        
        return self.results
    
    def select_best_model(self):
        """Select the best model based on performance."""
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        self.best_model_name = max(self.results, key=self.results.get)
        self.best_model = self.models[self.best_model_name]
        best_accuracy = self.results[self.best_model_name]
        best_metrics = self.detailed_metrics[self.best_model_name]
        
        logger.info(f"\n[🏆 BEST MODEL SELECTED]")
        logger.info(f"   Model: {self.best_model_name}")
        logger.info(f"   Accuracy: {best_accuracy:.2f}%")
        logger.info(f"   Performance Tier: {best_metrics['performance_interpretation']['overall_performance_tier']}")
        
        return self.best_model_name, self.best_model, best_accuracy, best_metrics
    
    def get_feature_importance(self, model, feature_names, top_n=20):
        """Extract feature importance from model."""
        important_features = []
        
        if hasattr(model, 'feature_importances_'):
            feat_imp = sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:top_n]
            
            important_features = [
                {
                    'feature': str(name), 
                    'importance': float(imp),
                    'rank': i + 1
                }
                for i, (name, imp) in enumerate(feat_imp)
            ]
            
            logger.info(f"[FEATURES] Top predictor: {important_features[0]['feature']}")
        
        return important_features
    
    def save_models(self, models_dir, version, best_model, preprocessor):
        """Save trained models and preprocessor."""
        import os
        
        # Ensure models_dir is a Path object
        if not isinstance(models_dir, Path):
            models_dir = Path(models_dir)
        
        # Create directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        version_file = models_dir / f"burnout_v{version}.pkl"
        latest_file = models_dir / "burnout_latest.pkl"
        preprocessor_file = models_dir / "preprocessor_latest.pkl"
        
        try:
            # Save versioned model
            joblib.dump(best_model, version_file)
            
            # Save latest model
            joblib.dump(best_model, latest_file)
            
            # Save preprocessor
            joblib.dump(preprocessor, preprocessor_file)
            
            logger.info(f"[SAVE] Models saved: {version_file}, {latest_file}, {preprocessor_file}")
            
            return [version_file, latest_file, preprocessor_file]
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save models: {e}")
            # Return empty list if saving fails
            return []

    
    def get_training_summary(self, X_train, y_train, original_row_count):
        """Create training summary."""
        return {
            'X_train_shape': X_train.shape,
            'y_train_distribution': dict(Counter(y_train)),
            'original_row_count': original_row_count,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds
        }