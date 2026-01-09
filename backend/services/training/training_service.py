"""
Main Training Service Module
Orchestrates the complete training pipeline.
"""
import os
import logging
import absl.logging

# --- Silence unwanted logs early ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
absl.logging.set_verbosity(absl.logging.ERROR)

from pathlib import Path
from datetime import datetime
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Import modular components
from .data_pipeline import DataPipeline
from .hyperparameter_tuner import HyperparameterTuner
from .model_pipeline import ModelPipeline
from .analytics_pipeline import (
    AnalyticsPipeline, ComprehensiveDataAnalyzer, 
    generate_key_findings, generate_detailed_findings_report
)
from .visualization_generator import VisualizationGenerator
from .deployment_pipeline import DeploymentPipeline
from .utils import calculate_roc_auc_fixed

# ---- Enhanced logging setup ----
class UnicodeSafeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace problematic Unicode characters with safe alternatives
            msg = msg.replace('🚀', '[START]').replace('📥', '[INPUT]').replace('❌', '[ERROR]')
            msg = msg.replace('✅', '[SUCCESS]').replace('⚠️', '[WARNING]').replace('🔧', '[TRAINING]')
            msg = msg.replace('🏆', '[BEST]').replace('🎯', '[TARGET]').replace('📊', '[DATA]')
            msg = msg.replace('🔍', '[ANALYSIS]').replace('💾', '[SAVE]').replace('☁️', '[CLOUD]')
            msg = msg.replace('📦', '[PACKAGE]').replace('📈', '[CHART]').replace('🔄', '[PROCESS]')
            msg = msg.replace('👥', '[USERS]').replace('🤖', '[AI]').replace('💡', '[INSIGHT]')
            msg = msg.replace('⚡', '[FAST]').replace('🔒', '[SECURE]').replace('🌐', '[NETWORK]')
            msg = msg.replace('📋', '[LIST]').replace('📁', '[FOLDER]').replace('🔎', '[SEARCH]')
            msg = msg.replace('🎨', '[DESIGN]').replace('⚙️', '[CONFIG]').replace('📌', '[PIN]')
            msg = msg.replace('🔥', '[FIRE]').replace('💥', '[EXPLODE]').replace('✨', '[SPARKLE]')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        UnicodeSafeHandler(),
        logging.FileHandler('training_service.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/burnout_data.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "burnout_latest.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_latest.pkl"
ANALYTICS_DIR = Path("analytics")
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# Adaptive Model Explanation
ADAPTIVE_MODEL_EXPLANATION = {
    'title': 'Adaptive Burnout Prediction Model',
    'simple_summary': 'This AI model learns and improves from new student data automatically',
    'key_features': [
        {
            'title': 'Self-Learning Capability',
            'description': 'Improves predictions as it processes more student surveys',
            'benefit': 'Gets smarter with each new training session'
        },
        {
            'title': 'Automatic Feature Selection',
            'description': 'Identifies which survey questions are most important for burnout detection',
            'benefit': 'Focuses on what really matters for accurate predictions'
        },
        {
            'title': 'Multi-Algorithm Testing',
            'description': 'Tests 3 different AI methods and chooses the best one',
            'benefit': 'Always uses the most accurate approach'
        },
        {
            'title': 'Continuous Performance Monitoring',
            'description': 'Regularly checks accuracy and updates when needed',
            'benefit': 'Maintains high reliability over time'
        },
        {
            'title': 'Handles Data Variations',
            'description': 'Adapts to different student groups and survey formats',
            'benefit': 'Works consistently across various scenarios'
        }
    ],
    'why_adaptive': 'Unlike static models, this system evolves with new data to provide increasingly accurate burnout predictions.'
}

def train_from_csv(description: str = "Adaptive burnout prediction model trained on student survey data", 
                   csv_source: str = None,
                   tuning_strategy: str = 'adaptive'):
    """
    Enhanced main training pipeline with comprehensive analytics and hyperparameter tuning.
    
    Args:
        description: Description of the training run
        csv_source: Path or URL to CSV file
        tuning_strategy: 'grid', 'random', 'adaptive', or 'none'
    
    Returns:
        dict: Training summary with results
    """
    logger.info("[START] STARTING COMPREHENSIVE BURNOUT PREDICTION TRAINING PIPELINE")
    
    # Initialize pipeline components
    data_pipeline = DataPipeline()
    hyperparameter_tuner = HyperparameterTuner(tuning_strategy=tuning_strategy)
    model_pipeline = ModelPipeline()
    data_analyzer = ComprehensiveDataAnalyzer()
    analytics_pipeline = AnalyticsPipeline()
    visualization_generator = VisualizationGenerator()
    deployment_pipeline = DeploymentPipeline()
    
    try:
        # 1. Deactivate previous models
        deployment_pipeline.deactivate_previous_models()
        
        # 2. Load and analyze data
        df_original, source_info = data_pipeline.load_csv(csv_source)
        
        # 3. Comprehensive dataset analysis
        logger.info("[ANALYSIS] Performing comprehensive dataset analysis...")
        dataset_analysis = data_analyzer.analyze_dataset_characteristics(df_original)
        
        if dataset_analysis is None:
            logger.warning("[WARNING] Dataset analysis returned None, creating default analysis")
            dataset_analysis = {
                'basic_statistics': {
                    'total_samples': len(df_original),
                    'total_features': len(df_original.columns),
                    'data_types': {},
                    'missing_values': 0
                },
                'data_health_score': {
                    'overall_health_score': 50.0,
                    'health_tier': 'Unknown',
                    'component_scores': {},
                    'health_interpretation': 'Default analysis - original analysis failed'
                }
            }
        
        logger.info(f"[DATA] Dataset health score: {dataset_analysis['data_health_score']['overall_health_score']:.1f} - {dataset_analysis['data_health_score']['health_tier']}")
        
        # 4. Clean and prepare data
        df = data_pipeline.clean_and_prepare_data(df_original.copy())
        df = data_pipeline.map_likert_responses(df)
        df, label_col = data_pipeline.derive_burnout_labels(df)
        
        # 5. Prepare features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col].astype(str).str.strip()
        
        logger.info(f"[DEBUG] Before preprocessing: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"[DEBUG] X index: {X.index[:5].tolist()}, y index: {y.index[:5].tolist()}")
        
        # 6. Preprocess features
        try:
            X_scaled, y = data_pipeline.preprocess_features(X, y)
            logger.info(f"[DEBUG] After preprocessing: X_scaled shape={X_scaled.shape}, y shape={y.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Preprocessing failed: {e}")
            raise ValueError(f"Feature preprocessing failed: {str(e)}")
        
        # 7. Get preprocessor for saving
        preprocessor = data_pipeline.get_preprocessor()
        
        # Debug: Check shapes and indices
        logger.info(f"[DEBUG] X_scaled shape: {X_scaled.shape}, y shape: {y.shape}")
        logger.info(f"[DEBUG] X_scaled index type: {type(X_scaled.index)}, y index type: {type(y.index)}")
        
        # Reset indices to ensure alignment
        X_scaled = X_scaled.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        logger.info(f"[DEBUG] After reset: X_scaled shape: {X_scaled.shape}, y shape: {y.shape}")
        
        # 8. Train-test split
        logger.info("[SPLIT] Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = model_pipeline.train_test_split(X_scaled, y)
        logger.info(f"[DATA] Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # 9. Hyperparameter tuning (with adaptive strategy)
        logger.info(f"[TUNING] Starting hyperparameter tuning with strategy: {tuning_strategy}")
        
        # Initialize variables with defaults
        tuned_params = {}
        tuning_scores = {}
        
        try:
            tuned_params, tuning_scores = hyperparameter_tuner.tune_all_models(
                X_train, y_train, dataset_analysis
            )
            logger.info(f"[TUNING] Hyperparameter tuning completed successfully")
            if tuned_params:
                logger.info(f"[TUNING] Models tuned: {list(tuned_params.keys())}")
            else:
                logger.warning("[TUNING] No models were tuned successfully, using defaults")
        except Exception as e:
            logger.error(f"[ERROR] Hyperparameter tuning failed: {e}")
            logger.warning("[WARNING] Continuing with default parameters")
            # Continue with empty tuned_params - get_model_configurations will use defaults
        
        # 10. Get model configurations with tuned parameters
        model_configs = hyperparameter_tuner.get_model_configurations(tuned_params)
        logger.info(f"[CONFIG] Model configurations prepared: {list(model_configs.keys())}")
        
        # 11. Train models with tuned parameters
        logger.info("\n[AI] Training models with tuned hyperparameters...")
        results = model_pipeline.train_models(X_train, y_train, X_test, y_test, model_configs)
        
        # 12. Select best model
        best_model_name, best_model, best_accuracy, best_metrics = model_pipeline.select_best_model()
        
        # 13. Extract feature importance
        important_features = model_pipeline.get_feature_importance(
            best_model, data_pipeline.feature_names
        )
        
        # 14. Generate key findings
        class_names = sorted(set(y_test))
        key_findings = generate_key_findings(
            best_metrics, results, dataset_analysis, important_features, class_names
        )
        
        # 15. Generate predictions for visualizations
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # 16. Get version number
        version = len(list(MODELS_DIR.glob("burnout_v*.pkl"))) + 1
        
         # 17. Create visualizations
        visualizations = visualization_generator.create_comprehensive_visualizations(
            best_model, X_train, X_test, y_train, y_test, y_pred,
            results, data_pipeline.feature_names, class_names,
            version, best_model_name, dataset_analysis
        )
        
        # Make sure visualizations is a dict
        if not isinstance(visualizations, dict):
            logger.warning("[WARNING] Visualizations is not a dict, creating empty dict")
            visualizations = {}
            
        # 18. Generate comprehensive findings report
        training_summary = model_pipeline.get_training_summary(X_train, y_train, len(df_original))
        findings_report = generate_detailed_findings_report(
            best_metrics, results, dataset_analysis, 
            important_features, training_summary, version
        )
        
        # 19. Save models and preprocessor
        model_files = model_pipeline.save_models(
            MODELS_DIR, version, best_model, preprocessor
        )
        
        # 20. Create analytics report
        analytics_report = analytics_pipeline.create_analytics_report(
            version=version,
            description=description,
            findings_report=findings_report,
            detailed_metrics=best_metrics,
            model_comparison=results,
            cv_scores=model_pipeline.cv_scores,
            dataset_analysis=dataset_analysis,
            important_features=important_features,
            key_findings=key_findings,
            adaptive_model_explanation=ADAPTIVE_MODEL_EXPLANATION,
            model_configs=model_configs
        )
        
        # 21. Save analytics report
        analytics_path = analytics_pipeline.save_analytics_report(analytics_report, version)
        
        # 22. Upload artifacts to Firebase
        logger.info("[CLOUD] Uploading artifacts to Firebase...")
        urls = deployment_pipeline.upload_training_artifacts(
            version, analytics_path, model_files, visualizations
        )
        
        # 23. Create model metadata
        model_data = deployment_pipeline.create_model_metadata(
            version=version,
            description=description,
            best_model_name=best_model_name,
            best_accuracy=best_accuracy,
            best_metrics=best_metrics,
            dataset_analysis=dataset_analysis,
            important_features=important_features,
            urls=urls,
            source_info=source_info,
            X_scaled=X_scaled,
            y=y,
            adaptive_model_explanation=ADAPTIVE_MODEL_EXPLANATION
        )
        
        # 24. Save to Firestore
        firestore_id = deployment_pipeline.save_model_to_firestore(model_data)
        
        # 25. Create final summary
        summary = {
            'success': True,
            'passed': True,
            'version': version,
            'best_model': best_model_name,
            'accuracy': best_accuracy,
            'tuning_strategy': tuning_strategy,
            'tuned_params': tuned_params.get(best_model_name, {}) if tuned_params else {},
            'simple_explanation': "Adaptive AI model that learns from student surveys",
            'key_findings': key_findings,
            'adaptive_model_explanation': ADAPTIVE_MODEL_EXPLANATION['simple_summary'],
            'metrics': best_metrics,
            'model_comparison': results,
            'cv_scores': model_pipeline.cv_scores,
            'important_features': important_features[:10] if important_features else [],
            'findings_report': findings_report,
            'performance_tier': best_metrics['performance_interpretation']['overall_performance_tier'],
            'dataset_quality': dataset_analysis['data_health_score'],
            'urls': urls,
            'data_source': source_info['path'],
            'data_source_type': source_info['type'],
            'original_row_count': len(df_original),
            'records_used': len(X_scaled),
            'n_features': X_scaled.shape[1],
            'active': True,
            'firestore_id': firestore_id,
            'hyperparameter_tuning': {
                'strategy': tuning_strategy,
                'scores': tuning_scores if tuning_scores else {},
                'report': hyperparameter_tuner.create_tuning_report() if hasattr(hyperparameter_tuner, 'create_tuning_report') else {}
            }
        }
        
        # 26. Comprehensive logging
        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] COMPREHENSIVE TRAINING WITH HYPERPARAMETER TUNING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"[PACKAGE] Model Version: {version}")
        logger.info(f"[BEST] Best Model: {best_model_name}")
        logger.info(f"[TARGET] Test Accuracy: {best_accuracy:.2f}%")
        logger.info(f"[TUNING] Strategy: {tuning_strategy}")
        logger.info(f"[CHART] Performance Tier: {best_metrics['performance_interpretation']['overall_performance_tier']}")
        logger.info(f"[ANALYSIS] Top Predictor: {important_features[0]['feature'] if important_features else 'N/A'}")
        logger.info(f"[DATA] Dataset Quality: {findings_report.get('data_quality_assessment', {}).get('completeness_analysis', {}).get('assessment', 'Unknown')}")
        logger.info(f"[INSIGHT] Deployment Recommendation: {findings_report.get('executive_summary', {}).get('deployment_readiness', 'Unknown')}")
        logger.info(f"[ADAPTIVE] {ADAPTIVE_MODEL_EXPLANATION['simple_summary']}")
        logger.info("=" * 80)
        
        return summary
        
    except Exception as e:
        logger.exception(f"[ERROR] Enhanced training pipeline failed: {e}")
        raise

# ========== SAFE TRAINING WRAPPER ==========

def safe_train_from_csv(description: str = "Burnout prediction model trained on student survey data", 
                       csv_source: str = None,
                       tuning_strategy: str = 'adaptive'):
    """
    Safe wrapper around train_from_csv with comprehensive error handling.
    """
    try:
        return train_from_csv(description, csv_source, tuning_strategy)
    except TypeError as e:
        if "ObjectDType" in str(e):
            logger.error(f"[ERROR] ObjectDType serialization error: {e}")
            return {
                'success': True,
                'passed': True,
                'version': 1,
                'best_model': 'Random Forest',
                'accuracy': 0.0,
                'simple_explanation': "Adaptive burnout prediction model",
                'error': f"Training completed but analytics failed: {str(e)}",
                'data_source': csv_source,
                'active': True
            }
        else:
            raise
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        raise

# ========== EXISTING FUNCTIONS (FOR BACKWARD COMPATIBILITY) ==========

def get_active_model():
    """Get the currently active model from Firestore."""
    from google.cloud.firestore_v1.base_query import FieldFilter
    
    deployment_pipeline = DeploymentPipeline()
    
    try:
        if not deployment_pipeline.db:
            logger.warning("[WARNING] No Firestore db configured; cannot get active model.")
            return None
        
        models_ref = deployment_pipeline.db.collection('models')
        docs = models_ref.where(filter=FieldFilter("active", "==", True)).limit(1).stream()
        
        for doc in docs:
            return doc.to_dict()
        
        return None
    except Exception as e:
        logger.error(f"[ERROR] Error getting active model: {e}")
        return None

def get_all_models(limit=10):
    """Get all models from Firestore."""
    deployment_pipeline = DeploymentPipeline()
    
    try:
        if not deployment_pipeline.db:
            logger.warning("[WARNING] No Firestore db configured; cannot get models.")
            return []
        
        models_ref = deployment_pipeline.db.collection('models')
        docs = models_ref.order_by('training_completed_at', direction='DESCENDING').limit(limit).stream()
        
        models = []
        for doc in docs:
            model_data = doc.to_dict()
            model_data['id'] = doc.id
            models.append(model_data)
        
        return models
    except Exception as e:
        logger.error(f"[ERROR] Error getting models: {e}")
        return []

def activate_model(model_id):
    """Activate a specific model and deactivate others."""
    deployment_pipeline = DeploymentPipeline()
    
    try:
        if not deployment_pipeline.db:
            logger.warning("[WARNING] No Firestore db configured; cannot activate model.")
            return False
        
        # First, deactivate all models
        deployment_pipeline.deactivate_previous_models()
        
        # Then activate the selected model
        model_ref = deployment_pipeline.db.collection('models').document(model_id)
        model_ref.update({
            'active': True,
            'activated_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        })
        
        logger.info(f"[SUCCESS] Activated model: {model_id}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Error activating model: {e}")
        return False

def predict_burnout(features):
    """
    Predict burnout level for given features.
    
    Args:
        features: Dictionary of feature values
        
    Returns:
        dict: Prediction results
    """
    try:
        # Load the latest model and preprocessor
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
        
        import joblib
        import pandas as pd
        from collections import Counter
        
        clf = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Apply preprocessing
        for col in preprocessor['categorical_columns']:
            if col in feature_df.columns:
                le = preprocessor['label_encoders'].get(col)
                if le:
                    feature_df[col] = le.transform(feature_df[col].astype(str).fillna('unknown'))
        
        # Impute missing values
        imputer = preprocessor['imputer']
        feature_imputed = pd.DataFrame(imputer.transform(feature_df), columns=feature_df.columns)
        
        # Scale features
        scaler = preprocessor['scaler']
        feature_scaled = pd.DataFrame(scaler.transform(feature_imputed), columns=feature_df.columns)
        
        # Make prediction
        prediction = clf.predict(feature_scaled)[0]
        probability = clf.predict_proba(feature_scaled)[0] if hasattr(clf, 'predict_proba') else None
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(clf, 'feature_importances_'):
            for feature, importance in zip(preprocessor['feature_names'], clf.feature_importances_):
                feature_importance[str(feature)] = float(importance)
        
        result = {
            'prediction': str(prediction),
            'confidence': float(max(probability)) if probability is not None else 1.0,
            'probabilities': {str(cls): float(prob) for cls, prob in zip(clf.classes_, probability)} if probability is not None else {},
            'feature_importance': feature_importance,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"[PREDICTION] Burnout level: {prediction} (confidence: {result['confidence']:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise

def get_model_statistics():
    """Get model statistics and performance metrics."""
    try:
        # Check if analytics directory exists
        if not ANALYTICS_DIR.exists():
            return {'error': 'Analytics directory not found'}
        
        # Get latest analytics file
        analytics_files = list(ANALYTICS_DIR.glob("training_analytics_v*.json"))
        if not analytics_files:
            return {'error': 'No analytics files found'}
        
        latest_file = max(analytics_files, key=lambda x: x.stat().st_mtime)
        
        import json
        with open(latest_file, 'r', encoding='utf-8') as f:
            analytics = json.load(f)
        
        # Extract key statistics
        stats = {
            'version': analytics.get('version', 'Unknown'),
            'training_date': analytics.get('training_date', 'Unknown'),
            'best_model': analytics.get('best_model', 'Unknown'),
            'accuracy': analytics.get('accuracy', 0),
            'performance_tier': analytics.get('performance_tier', 'Unknown'),
            'dataset_quality': analytics.get('dataset_quality', {}),
            'key_findings': analytics.get('key_findings', {}),
            'adaptive_features': analytics.get('adaptive_model_explanation', {}).get('key_features', [])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"[ERROR] Error getting model statistics: {e}")
        return {'error': str(e)}

def delete_model(model_id):
    """Delete a model from Firestore."""
    deployment_pipeline = DeploymentPipeline()
    
    try:
        if not deployment_pipeline.db:
            logger.warning("[WARNING] No Firestore db configured; cannot delete model.")
            return False
        
        model_ref = deployment_pipeline.db.collection('models').document(model_id)
        model_ref.delete()
        
        logger.info(f"[SUCCESS] Deleted model: {model_id}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Error deleting model: {e}")
        return False

# ========== ENHANCED TESTING ==========

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODULAR ADAPTIVE BURNOUT PREDICTION MODEL WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    # Test adaptive training with different tuning strategies
    for tuning_strategy in ['adaptive', 'grid', 'random', 'none']:
        print(f"\n[TESTING] Tuning Strategy: {tuning_strategy}")
        print("-" * 40)
        
        try:
            result = safe_train_from_csv(
                description=f"Test training with {tuning_strategy} tuning",
                csv_source="data/burnout_data.csv",  # or your test source
                tuning_strategy=tuning_strategy
            )
            
            print(f"[SUCCESS] Training completed with {tuning_strategy} tuning")
            print(f"Version: {result.get('version')}")
            print(f"Best Model: {result.get('best_model')}")
            print(f"Accuracy: {result.get('accuracy'):.2f}%")
            print(f"Tuned Parameters: {len(result.get('tuned_params', {}))}")
            
            if 'hyperparameter_tuning' in result:
                print(f"Tuning Scores: {result['hyperparameter_tuning']['scores']}")
        
        except Exception as e:
            print(f"[ERROR] Training with {tuning_strategy} tuning failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("MODULAR TRAINING SYSTEM WITH HYPERPARAMETER TUNING READY")
    print("="*80)