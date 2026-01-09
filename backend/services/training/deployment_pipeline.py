"""
Deployment Pipeline Module
Handles Firebase upload, Firestore metadata, and deployment.
"""
import os
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter  # <-- ADD THIS IMPORT
from google.cloud.firestore_v1.base_query import FieldFilter

from backend.services.firebase_service import db, bucket
from .utils import convert_to_native_types

logger = logging.getLogger(__name__)

class DeploymentPipeline:
    """Handles deployment of trained models to Firebase"""
    
    def __init__(self):
        self.urls = {
            'analytics': None,
            'models': {},
            'visualizations': {}
        }
    
    def upload_to_firebase_storage(self, file_path, destination_path):
        """Upload a file to Firebase Storage."""
        try:
            if not bucket:
                logger.warning("[WARNING] No Firebase Storage bucket configured; skipping upload.")
                return None
            
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            
            # Make the blob publicly accessible
            blob.make_public()
            
            logger.info(f"[CLOUD] Uploaded to Firebase Storage: {destination_path}")
            return blob.public_url
            
        except Exception as e:
            logger.error(f"[ERROR] Firebase Storage upload failed: {e}")
            return None
    
    def save_model_to_firestore(self, model_data):
        """Save model metadata to Firestore."""
        try:
            if not db:
                logger.warning("[WARNING] No Firestore db configured; skipping Firestore save.")
                return None
            
            # Convert to native types for Firestore
            firestore_data = convert_to_native_types(model_data)
            
            # Add timestamps
            firestore_data['created_at'] = datetime.utcnow()
            firestore_data['updated_at'] = datetime.utcnow()
            
            # Create document in models collection
            doc_ref = db.collection('models').document()
            doc_ref.set(firestore_data)
            
            logger.info(f"[FIRE] Model saved to Firestore: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"[ERROR] Firestore save failed: {e}")
            return None
    
    def upload_training_artifacts(self, version, analytics_path, model_files, visualizations):
        """Upload all training artifacts to Firebase."""
        urls = {
            'analytics': None,
            'models': {},
            'visualizations': {}
        }
        
        try:
            # Upload analytics report
            if analytics_path:
                # Convert to Path if it's a string
                if isinstance(analytics_path, str):
                    analytics_path = Path(analytics_path)
                
                if analytics_path.exists():
                    analytics_dest = f"analytics/training_analytics_v{version}.json"
                    urls['analytics'] = self.upload_to_firebase_storage(analytics_path, analytics_dest)
            
            # Upload model files
            for model_file in model_files:
                if model_file:
                    # Convert to Path if it's a string
                    if isinstance(model_file, str):
                        model_file = Path(model_file)
                    
                    if model_file.exists():
                        model_name = model_file.name
                        model_dest = f"models/v{version}/{model_name}"
                        urls['models'][model_name] = self.upload_to_firebase_storage(model_file, model_dest)
            
            # Upload visualizations
            for viz_name, viz_buffer in visualizations.items():
                if viz_buffer:
                    try:
                        # Save visualization to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_file.write(viz_buffer.getvalue())
                            temp_path = temp_file.name
                        
                        # Upload to Firebase
                        viz_dest = f"visualizations/v{version}/{viz_name}.png"
                        urls['visualizations'][viz_name] = self.upload_to_firebase_storage(temp_path, viz_dest)
                        
                        # Clean up temporary file
                        os.unlink(temp_path)
                        
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed to upload visualization {viz_name}: {e}")
            
            logger.info(f"[CLOUD] All artifacts uploaded for version {version}")
            return urls
            
        except Exception as e:
            logger.error(f"[ERROR] Artifact upload failed: {e}")
            return urls
    
    def deactivate_previous_models(self):
        """Mark all previous models as inactive in Firestore."""
        if not db:
            logger.warning("[WARNING] No Firestore db configured; skipping model deactivation.")
            return
        
        try:
            logger.info("[PROCESS] Deactivating previous models...")
            models_ref = db.collection('models')
            docs = models_ref.where(filter=FieldFilter("active", "==", True)).stream()
            
            count = 0
            for doc in docs:
                doc.reference.update({'active': False, 'deactivated_at': datetime.utcnow()})
                count += 1
            
            logger.info(f"Deactivated {count} previous model(s)")
        except Exception as e:
            logger.exception(f"[ERROR] Error deactivating previous models: {e}")
    
    def create_model_metadata(self, version, description, best_model_name, best_accuracy,
                            best_metrics, dataset_analysis, important_features, 
                            urls, source_info, X_scaled, y, adaptive_model_explanation):
        """Create model metadata for Firestore."""
        
        model_data = {
            'version': version,
            'description': description,
            'simple_description': "Adaptive AI burnout prediction model",
            'best_model': best_model_name,
            'accuracy': best_accuracy,
            'performance_tier': best_metrics['performance_interpretation']['overall_performance_tier'],
            'key_findings': {
                'model_performance': f"{best_accuracy:.1f}% accuracy",
                'best_algorithm': best_model_name,
                'dataset_quality': dataset_analysis['data_health_score']['health_tier']
            },
            'dataset_quality': dataset_analysis['data_health_score']['health_tier'],
            'dataset_health_score': dataset_analysis['data_health_score']['overall_health_score'],
            'n_features': X_scaled.shape[1],
            'n_samples': len(X_scaled),
            'class_distribution': dict(Counter(y)),
            'training_parameters': {
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation_folds': 5,
                'models_tested': ['Random Forest', 'Decision Tree', 'SVM']
            },
            'metrics_summary': {
                'accuracy': best_accuracy,
                'balanced_accuracy': best_metrics['basic_metrics']['balanced_accuracy']['value'],
                'precision_macro': best_metrics['basic_metrics']['precision_macro']['value'],
                'recall_macro': best_metrics['basic_metrics']['recall_macro']['value'],
                'f1_macro': best_metrics['basic_metrics']['f1_macro']['value'],
                'cohens_kappa': best_metrics['advanced_metrics']['cohens_kappa']['value']
            },
            'important_features': important_features[:10],
            'adaptive_features': adaptive_model_explanation['key_features'],
            'urls': urls,
            'data_source': source_info['path'],
            'data_source_type': source_info['type'],
            'active': True,
            'training_completed_at': datetime.utcnow()
        }
        
        return model_data