"""
Hyperparameter Tuning Module
Implements comprehensive hyperparameter optimization.
"""
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Advanced hyperparameter tuning with adaptive strategies"""
    
    # Enhanced model configurations with tuning grids
    MODEL_CONFIGS = {
        "Random Forest": {
            'class': RandomForestClassifier,
            'param_grid': {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'param_distributions': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'default_params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        "Decision Tree": {
            'class': DecisionTreeClassifier,
            'param_grid': {
                'max_depth': [3, 5, 8, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 8, 10],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None]
            },
            'param_distributions': {
                'max_depth': randint(3, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None]
            },
            'default_params': {
                'max_depth': 8,
                'min_samples_split': 15,
                'min_samples_leaf': 8,
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        "SVM": {
            'class': SVC,
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': [2, 3, 4],
                'class_weight': ['balanced', None]
            },
            'param_distributions': {
                'C': uniform(0.01, 100),
                'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 6)),
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': randint(2, 5),
                'class_weight': ['balanced', None]
            },
            'default_params': {
                'kernel': 'rbf',
                'C': 0.1,
                'gamma': 'scale',
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'max_iter': 1000
            }
        }
    }
    
    def __init__(self, tuning_strategy='adaptive', cv_folds=5, n_iter=50):
        """
        Initialize hyperparameter tuner.
        
        Args:
            tuning_strategy: 'grid', 'random', 'adaptive', or 'none'
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search
        """
        self.tuning_strategy = tuning_strategy
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.tuned_params = {}
        self.best_scores = {}
        
    def adaptive_parameter_selection(self, dataset_analysis):
        """Dynamically adjust parameters based on dataset characteristics."""
        n_samples = dataset_analysis['basic_statistics']['total_samples']
        n_features = dataset_analysis['basic_statistics']['total_features']
        class_balance = dataset_analysis['data_health_score']['component_scores'].get('balance_quality', 50)
        
        adaptive_params = {}
        
        # Adjust parameters based on dataset size
        if n_samples < 500:
            # Small dataset - simpler models
            adaptive_params['Random Forest'] = {
                'n_estimators': min(100, n_samples // 5),
                'max_depth': min(10, n_features * 2),
                'min_samples_split': max(5, n_samples // 100),
                'min_samples_leaf': max(2, n_samples // 200)
            }
            adaptive_params['SVM'] = {
                'C': 1.0,  # Less regularization for small datasets
                'kernel': 'linear'  # Simpler kernel
            }
        elif n_samples > 5000:
            # Large dataset - more complex models
            adaptive_params['Random Forest'] = {
                'n_estimators': 400,
                'max_depth': None,  # Unlimited depth
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            adaptive_params['SVM'] = {
                'C': 0.1,  # More regularization for large datasets
                'kernel': 'rbf'  # Complex kernel
            }
        
        # Adjust for class imbalance
        if class_balance < 60:  # Imbalanced data
            adaptive_params['Random Forest']['class_weight'] = 'balanced_subsample'
            adaptive_params['SVM']['class_weight'] = 'balanced'
        else:  # Balanced data
            adaptive_params['Random Forest']['class_weight'] = None
            adaptive_params['SVM']['class_weight'] = None
        
        # Adjust for high dimensionality
        if n_features > 50:
            adaptive_params['Random Forest']['max_features'] = 'sqrt'
            adaptive_params['SVM']['kernel'] = 'rbf'
        
        return adaptive_params
    
    def tune_single_model(self, model_name, X_train, y_train, adaptive_params=None):
        """Tune hyperparameters for a single model."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.MODEL_CONFIGS[model_name]
        model_class = config['class']
        
        # Start with default parameters
        base_params = config['default_params'].copy()
        
        # Apply adaptive parameters if available
        if adaptive_params and model_name in adaptive_params:
            base_params.update(adaptive_params[model_name])
        
        if self.tuning_strategy == 'none':
            logger.info(f"[TUNING] Skipping tuning for {model_name}, using adaptive parameters")
            return base_params, 0.0
        
        logger.info(f"[TUNING] Starting hyperparameter tuning for {model_name}")
        
        # Create base model
        model = model_class(**base_params)
        
        try:
            # Select tuning strategy
            if self.tuning_strategy == 'grid' and 'param_grid' in config:
                logger.info(f"[TUNING] Using GridSearchCV for {model_name}")
                search = GridSearchCV(
                    model,
                    config['param_grid'],
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    verbose=0
                )
            elif self.tuning_strategy == 'random' and 'param_distributions' in config:
                logger.info(f"[TUNING] Using RandomizedSearchCV for {model_name}")
                search = RandomizedSearchCV(
                    model,
                    config['param_distributions'],
                    n_iter=self.n_iter,
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                    scoring='balanced_accuracy',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
            elif self.tuning_strategy == 'adaptive':
                logger.info(f"[TUNING] Using adaptive strategy for {model_name}")
                # Use cross-validation with adaptive parameters
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                    scoring='balanced_accuracy',
                    n_jobs=-1
                )
                score = cv_scores.mean() if len(cv_scores) > 0 else 0.0
                logger.info(f"[TUNING] Adaptive CV score for {model_name}: {score:.4f}")
                return base_params, score
            else:
                logger.warning(f"[WARNING] No tuning strategy for {model_name}, using default params")
                return base_params, 0.0
            
            # Perform search
            logger.info(f"[TUNING] Fitting {model_name}...")
            search.fit(X_train, y_train)
            
            # Store results
            self.tuned_params[model_name] = search.best_params_
            self.best_scores[model_name] = search.best_score_
            
            logger.info(f"[TUNING] Best {model_name} params: {search.best_params_}")
            logger.info(f"[TUNING] Best CV score: {search.best_score_:.4f}")
            
            return search.best_params_, search.best_score_
            
        except Exception as e:
            logger.error(f"[ERROR] Tuning failed for {model_name}: {e}")
            # Return default parameters
            return base_params, 0.0
    
    def tune_all_models(self, X_train, y_train, dataset_analysis=None, models_to_tune=None):
        """Tune hyperparameters for all models."""
        if models_to_tune is None:
            models_to_tune = list(self.MODEL_CONFIGS.keys())
        
        # Get adaptive parameters if dataset analysis is available
        adaptive_params = None
        if dataset_analysis and self.tuning_strategy == 'adaptive':
            try:
                adaptive_params = self.adaptive_parameter_selection(dataset_analysis)
            except Exception as e:
                logger.error(f"[ERROR] Adaptive parameter selection failed: {e}")
                adaptive_params = None
        
        tuned_params = {}
        tuning_scores = {}
        
        for model_name in models_to_tune:
            try:
                logger.info(f"[TUNING] Tuning {model_name}...")
                params, score = self.tune_single_model(
                    model_name, X_train, y_train, adaptive_params
                )
                tuned_params[model_name] = params
                tuning_scores[model_name] = score
                logger.info(f"[TUNING] {model_name} tuning complete: score={score:.4f}")
            except Exception as e:
                logger.error(f"[ERROR] Tuning failed for {model_name}: {e}")
                # Use default parameters as fallback
                if model_name in self.MODEL_CONFIGS:
                    tuned_params[model_name] = self.MODEL_CONFIGS[model_name]['default_params']
                    tuning_scores[model_name] = 0.0
                else:
                    logger.warning(f"[WARNING] Model {model_name} not in configs, skipping")
        
        # If no models were tuned successfully, create empty dicts
        if not tuned_params:
            logger.warning("[WARNING] No models were tuned successfully")
        
        return tuned_params, tuning_scores
    
    def get_model_configurations(self, tuned_params=None):
        """Get model configurations with tuned parameters."""
        model_configs = {}
        
        for model_name, config in self.MODEL_CONFIGS.items():
            # Start with default parameters
            params = config['default_params'].copy()
            
            # Update with tuned parameters if available
            if tuned_params and model_name in tuned_params:
                params.update(tuned_params[model_name])
            
            model_configs[model_name] = {
                'class': config['class'],
                'params': params,
                'description': 'Tuned model with optimized hyperparameters',
                'simple_explanation': f'Optimized {model_name} for your dataset'
            }
        
        return model_configs
    
    def create_tuning_report(self):
        """Create a report of tuning results."""
        report = {
            'tuning_strategy': self.tuning_strategy,
            'cv_folds': self.cv_folds,
            'tuned_parameters': self.tuned_params,
            'best_scores': self.best_scores,
            'improvement_analysis': {}
        }
        
        # Calculate improvement over defaults
        for model_name in self.tuned_params:
            if model_name in self.best_scores:
                report['improvement_analysis'][model_name] = {
                    'best_score': self.best_scores[model_name],
                    'tuning_method': 'Grid Search' if self.tuning_strategy == 'grid' else 'Random Search',
                    'parameters_tuned': len(self.tuned_params[model_name])
                }
        
        return report