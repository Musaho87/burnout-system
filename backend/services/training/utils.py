"""
Utility functions for the training service.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def convert_to_native_types(obj):
    """
    Recursively convert numpy/pandas types to native Python types.
    """
    try:
        # Handle None values
        if obj is None:
            return None
        
        # Handle pandas NA values
        if pd.isna(obj):
            return None
        
        # Handle basic scalar types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return convert_to_native_types(obj.tolist())
        
        # Handle pandas Series and Index
        if isinstance(obj, (pd.Series, pd.Index)):
            return convert_to_native_types(obj.tolist())
        
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return {
                'columns': convert_to_native_types(obj.columns.tolist()),
                'data': convert_to_native_types(obj.values.tolist()),
                'shape': obj.shape
            }
        
        # Handle datetime objects
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        
        # Handle ObjectDType and other pandas dtypes
        if hasattr(obj, 'dtype'):
            return str(obj)
        
        # Handle dictionaries
        if isinstance(obj, dict):
            converted_dict = {}
            for k, v in obj.items():
                # Convert key to string if it's not a basic type
                if not isinstance(k, (str, int, float, bool)) and k is not None:
                    safe_key = str(k)
                else:
                    safe_key = k
                converted_dict[safe_key] = convert_to_native_types(v)
            return converted_dict
        
        # Handle lists, tuples, and other iterables
        if isinstance(obj, (list, tuple, set)):
            return [convert_to_native_types(item) for item in obj]
        
        # Handle other types - convert to string representation as last resort
        try:
            return str(obj)
        except:
            return f"<unserializable: {type(obj).__name__}>"
            
    except Exception as e:
        logger.warning(f"[WARNING] Error converting type {type(obj)} to native: {e}")
        try:
            return str(obj)
        except:
            return f"<conversion_error: {type(obj).__name__}>"

def ensure_string_keys(obj):
    """
    Ensure all dictionary keys are strings for JSON serialization.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            # Convert key to string
            safe_key = str(k) if not isinstance(k, (str, int, float, bool)) else k
            if isinstance(safe_key, (int, float, bool)):
                safe_key = str(safe_key)
            cleaned[safe_key] = ensure_string_keys(v)
        return cleaned
    elif isinstance(obj, list):
        return [ensure_string_keys(item) for item in obj]
    else:
        return convert_to_native_types(obj)

def clean_analytics_report(report):
    """
    Specifically clean the analytics report to ensure JSON serialization.
    """
    def deep_clean(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # Ensure keys are strings
                safe_key = str(k) if not isinstance(k, (str, int, float, bool)) else k
                if isinstance(safe_key, (int, float, bool)):
                    safe_key = str(safe_key)
                cleaned[safe_key] = deep_clean(v)
            return cleaned
        elif isinstance(obj, list):
            return [deep_clean(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.Index)):
            return deep_clean(obj.tolist())
        elif hasattr(obj, 'dtype'):
            return str(obj)
        else:
            return convert_to_native_types(obj)
    
    return deep_clean(report)

def calculate_roc_auc_fixed(y_true, y_proba, class_names):
    """
    Fixed ROC AUC calculation that handles all scenarios with robust error handling.
    """
    try:
        logger.info(f"[ROC_AUC_DEBUG] Starting ROC AUC calculation...")
        logger.info(f"[ROC_AUC_DEBUG] Class names: {class_names}")
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        # Ensure y_proba is 2D
        if len(y_proba.shape) == 1:
            logger.warning("[ROC_AUC_DEBUG] y_proba is 1D, reshaping to 2D")
            y_proba = y_proba.reshape(-1, 1)
        
        # Check for NaN or invalid values
        if np.any(np.isnan(y_proba)):
            logger.warning("[ROC_AUC_DEBUG] NaN found in y_proba, replacing with 0")
            y_proba = np.nan_to_num(y_proba)
        
        # Convert y_true to numerical labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        
        # For binary classification
        if len(class_names) == 2:
            logger.info("[ROC_AUC_DEBUG] Binary classification detected")
            try:
                # Use appropriate probability column
                if y_proba.shape[1] == 2:
                    from sklearn.metrics import roc_auc_score
                    auc_score = roc_auc_score(y_true_encoded, y_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_true_encoded, y_proba[:, 0])
                
                logger.info(f"[ROC_AUC_DEBUG] Binary AUC: {auc_score}")
                
                # SIMPLE, SERIALIZABLE RETURN
                return {
                    'binary_auc': float(auc_score),
                    'average_auc': float(auc_score),
                    'interpretation': 'Excellent' if auc_score > 0.8 else 'Good' if auc_score > 0.7 else 'Fair',
                    'score': float(auc_score),
                    'calculation_method': 'binary'
                }
            except Exception as e:
                logger.error(f"[ROC_AUC_DEBUG] Binary AUC failed: {e}")
                # Return simple error structure
                return {
                    'error': str(e),
                    'average_auc': 0.5,
                    'score': 0.5,
                    'interpretation': 'ROC AUC calculation failed',
                    'calculation_method': 'binary_fallback'
                }
        
        # For multi-class classification
        logger.info("[ROC_AUC_DEBUG] Multi-class classification detected")
        try:
            # Calculate both OVO and OVR AUC
            from sklearn.metrics import roc_auc_score
            auc_ovo = roc_auc_score(y_true_encoded, y_proba, multi_class='ovo', average='macro')
            auc_ovr = roc_auc_score(y_true_encoded, y_proba, multi_class='ovr', average='macro')
            
            avg_auc = (auc_ovo + auc_ovr) / 2
            
            logger.info(f"[ROC_AUC_DEBUG] OVO AUC: {auc_ovo}, OVR AUC: {auc_ovr}, Average: {avg_auc}")
            
            # SIMPLE, SERIALIZABLE RETURN
            result = {
                'auc_ovo': float(auc_ovo),
                'auc_ovr': float(auc_ovr),
                'average_auc': float(avg_auc),
                'score': float(avg_auc),  # Duplicate for compatibility
                'interpretation': 'Excellent class separation' if avg_auc > 0.8 else 'Good separation' if avg_auc > 0.7 else 'Fair separation',
                'calculation_method': 'multi_class_ovo_ovr_average'
            }
            
            return result
            
        except Exception as multi_error:
            logger.error(f"[ROC_AUC_DEBUG] Multi-class AUC failed: {multi_error}")
            
            # Fallback: Calculate per-class AUC
            try:
                auc_scores = []
                unique_classes = np.unique(y_true_encoded)
                
                for i in range(len(unique_classes)):
                    try:
                        # Create binary labels for this class
                        y_binary = (y_true_encoded == i).astype(int)
                        
                        # Check if we have both classes present
                        if len(np.unique(y_binary)) < 2:
                            auc_scores.append(0.5)
                            continue
                        
                        # Get probabilities for this class
                        if i < y_proba.shape[1]:
                            prob_class = y_proba[:, i]
                        else:
                            prob_class = y_proba[:, 0]
                        
                        class_auc = roc_auc_score(y_binary, prob_class)
                        auc_scores.append(float(class_auc))  # Convert to float immediately
                    except Exception:
                        auc_scores.append(0.5)
                
                avg_auc = float(np.mean(auc_scores)) if auc_scores else 0.5
                
                # SIMPLE, SERIALIZABLE RETURN
                return {
                    'per_class_auc': auc_scores,  # Already converted to floats
                    'average_auc': avg_auc,
                    'score': avg_auc,
                    'interpretation': 'Per-class AUC calculated',
                    'calculation_method': 'per_class_fallback'
                }
                
            except Exception as fallback_error:
                logger.error(f"[ROC_AUC_DEBUG] All AUC methods failed: {fallback_error}")
                # SIMPLE, SERIALIZABLE ERROR RETURN
                return {
                    'error': str(fallback_error),
                    'average_auc': 0.5,
                    'score': 0.5,
                    'interpretation': 'ROC AUC calculation failed',
                    'calculation_method': 'complete_fallback'
                }
            
    except Exception as e:
        logger.error(f"[ROC_AUC_DEBUG] ROC AUC calculation failed: {e}")
        # SIMPLE, SERIALIZABLE ERROR RETURN
        return {
            'error': str(e),
            'average_auc': 0.5,
            'score': 0.5,
            'interpretation': 'ROC AUC calculation failed',
            'calculation_method': 'exception_fallback'
        }