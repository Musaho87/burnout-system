"""
Analytics Pipeline Module
Handles comprehensive metrics calculation and reporting.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

from .utils import convert_to_native_types, ensure_string_keys, clean_analytics_report

logger = logging.getLogger(__name__)

class EnhancedMetricsCalculator:
    """Comprehensive metrics calculation with detailed interpretations"""
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, class_names=None):
        """Calculate comprehensive evaluation metrics with detailed analysis"""
        
        metrics = {
            'basic_metrics': {},
            'class_wise_metrics': {},
            'advanced_metrics': {},
            'model_diagnostics': {},
            'performance_interpretation': {},
            'comparative_analysis': {},
            'statistical_significance': {},
            'key_insights': {}
        }
        
        if class_names is None:
            class_names = sorted(set(y_true) | set(y_pred))
        
        # Ensure class names are strings
        class_names = [str(cls) for cls in class_names]
        y_true = [str(y) for y in y_true]
        y_pred = [str(y) for y in y_pred]
        
        # Basic metrics with detailed interpretations
        metrics['basic_metrics'] = EnhancedMetricsCalculator._calculate_basic_metrics(y_true, y_pred)
        
        # Class-wise metrics
        metrics['class_wise_metrics'] = EnhancedMetricsCalculator._calculate_class_wise_metrics(y_true, y_pred, class_names)
        
        # Advanced metrics
        metrics['advanced_metrics'] = EnhancedMetricsCalculator._calculate_advanced_metrics(y_true, y_pred, y_proba, class_names)
        
        # Model diagnostics
        metrics['model_diagnostics'] = EnhancedMetricsCalculator._calculate_model_diagnostics(y_true, y_pred, class_names)
        
        # Performance interpretation
        metrics['performance_interpretation'] = EnhancedMetricsCalculator._provide_performance_interpretation(metrics)
        
        # Comparative analysis
        metrics['comparative_analysis'] = EnhancedMetricsCalculator._perform_comparative_analysis(metrics, y_true)
        
        # Statistical significance
        metrics['statistical_significance'] = EnhancedMetricsCalculator._assess_statistical_significance(metrics, len(y_true))
        
        # Key insights
        metrics['key_insights'] = EnhancedMetricsCalculator._generate_key_insights(metrics, y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _calculate_basic_metrics(y_true, y_pred):
        """Calculate basic performance metrics"""
        try:
            acc = accuracy_score(y_true, y_pred) * 100
            balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
            
            return {
                'accuracy': {
                    'value': acc,
                    'interpretation': 'Excellent' if acc > 85 else 'Good' if acc > 70 else 'Fair' if acc > 55 else 'Poor'
                },
                'balanced_accuracy': {
                    'value': balanced_acc,
                    'interpretation': 'Excellent' if balanced_acc > 85 else 'Good' if balanced_acc > 70 else 'Fair' if balanced_acc > 55 else 'Poor'
                },
                'precision_macro': {
                    'value': precision_macro,
                    'interpretation': 'Excellent' if precision_macro > 85 else 'Good' if precision_macro > 70 else 'Fair' if precision_macro > 55 else 'Poor'
                },
                'recall_macro': {
                    'value': recall_macro,
                    'interpretation': 'Excellent' if recall_macro > 85 else 'Good' if recall_macro > 70 else 'Fair' if recall_macro > 55 else 'Poor'
                },
                'f1_macro': {
                    'value': f1_macro,
                    'interpretation': 'Excellent' if f1_macro > 85 else 'Good' if f1_macro > 70 else 'Fair' if f1_macro > 55 else 'Poor'
                }
            }
        except Exception as e:
            logger.error(f"[ERROR] Error calculating basic metrics: {e}")
            return {
                'accuracy': {'value': 0, 'interpretation': 'Calculation failed'},
                'balanced_accuracy': {'value': 0, 'interpretation': 'Calculation failed'},
                'precision_macro': {'value': 0, 'interpretation': 'Calculation failed'},
                'recall_macro': {'value': 0, 'interpretation': 'Calculation failed'},
                'f1_macro': {'value': 0, 'interpretation': 'Calculation failed'}
            }
    
    @staticmethod
    def _calculate_class_wise_metrics(y_true, y_pred, class_names):
        """Calculate metrics for each class"""
        try:
            class_metrics = {}
            for cls in class_names:
                # Create binary classification for this class
                y_true_binary = [1 if y == cls else 0 for y in y_true]
                y_pred_binary = [1 if y == cls else 0 for y in y_pred]
                
                if sum(y_true_binary) > 0:  # Check if class exists in true labels
                    tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
                    fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
                    fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_metrics[cls] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'support': sum(y_true_binary)
                    }
            
            return class_metrics
        except Exception as e:
            logger.error(f"[ERROR] Error calculating class-wise metrics: {e}")
            return {}
    
    @staticmethod
    def _calculate_advanced_metrics(y_true, y_pred, y_proba, class_names):
        """Calculate advanced metrics including ROC AUC"""
        try:
            metrics = {}
            
            # Cohen's Kappa
            kappa = cohen_kappa_score(y_true, y_pred)
            metrics['cohens_kappa'] = {
                'value': kappa,
                'interpretation': 'Almost Perfect' if kappa > 0.8 else 'Substantial' if kappa > 0.6 else 
                                 'Moderate' if kappa > 0.4 else 'Fair' if kappa > 0.2 else 'Slight' if kappa > 0 else 'Poor'
            }
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            metrics['matthews_corrcoef'] = {
                'value': mcc,
                'interpretation': 'Strong' if mcc > 0.5 else 'Moderate' if mcc > 0.3 else 
                                 'Weak' if mcc > 0.1 else 'No correlation'
            }
            
            return metrics
        except Exception as e:
            logger.error(f"[ERROR] Error calculating advanced metrics: {e}")
            return {
                'cohens_kappa': {'value': 0, 'interpretation': 'Calculation failed'},
                'matthews_corrcoef': {'value': 0, 'interpretation': 'Calculation failed'}
            }
    
    @staticmethod
    def _calculate_model_diagnostics(y_true, y_pred, class_names):
        """Perform model diagnostics"""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=class_names)
            
            diagnostics = {
                'confusion_matrix': cm.tolist(),
                'prediction_distribution': dict(Counter(y_pred)),
                'true_distribution': dict(Counter(y_true))
            }
            
            return diagnostics
        except Exception as e:
            logger.error(f"[ERROR] Error calculating model diagnostics: {e}")
            return {}
    
    @staticmethod
    def _provide_performance_interpretation(metrics):
        """Provide overall performance interpretation"""
        try:
            acc = metrics['basic_metrics']['accuracy']['value']
            
            if acc > 85:
                tier = 'Excellent'
                recommendation = 'Model is ready for production deployment'
            elif acc > 70:
                tier = 'Good'
                recommendation = 'Model performs well, consider minor improvements'
            elif acc > 55:
                tier = 'Fair'
                recommendation = 'Model needs improvement before deployment'
            else:
                tier = 'Poor'
                recommendation = 'Significant improvement needed'
            
            return {
                'overall_performance_tier': tier,
                'recommendation': recommendation,
                'summary': f"Model achieves {acc:.1f}% accuracy, classified as {tier.lower()} performance"
            }
        except Exception as e:
            logger.error(f"[ERROR] Error providing performance interpretation: {e}")
            return {
                'overall_performance_tier': 'Unknown',
                'recommendation': 'Analysis failed',
                'summary': 'Performance interpretation failed'
            }
    
    @staticmethod
    def _perform_comparative_analysis(metrics, y_true):
        """Perform comparative analysis"""
        try:
            baseline_accuracy = 100 / len(set(y_true))  # Random baseline
            actual_accuracy = metrics['basic_metrics']['accuracy']['value']
            improvement = ((actual_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            
            return {
                'baseline_accuracy': baseline_accuracy,
                'improvement_over_baseline': improvement,
                'relative_performance': 'Much better than random' if improvement > 50 else 
                                       'Better than random' if improvement > 20 else 
                                       'Similar to random' if improvement > -20 else 
                                       'Worse than random'
            }
        except Exception as e:
            logger.error(f"[ERROR] Error performing comparative analysis: {e}")
            return {}
    
    @staticmethod
    def _assess_statistical_significance(metrics, n_samples):
        """Assess statistical significance"""
        try:
            accuracy = metrics['basic_metrics']['accuracy']['value'] / 100
            se = np.sqrt(accuracy * (1 - accuracy) / n_samples)
            ci_95 = 1.96 * se * 100  # 95% confidence interval
            
            return {
                'standard_error': se * 100,
                'confidence_interval_95': f"±{ci_95:.2f}%",
                'margin_of_error': ci_95,
                'reliability': 'High' if ci_95 < 5 else 'Medium' if ci_95 < 10 else 'Low'
            }
        except Exception as e:
            logger.error(f"[ERROR] Error assessing statistical significance: {e}")
            return {}
    
    @staticmethod
    def _generate_key_insights(metrics, y_true, y_pred):
        """Generate key insights from metrics"""
        try:
            insights = []
            
            # Insight 1: Overall performance
            acc = metrics['basic_metrics']['accuracy']['value']
            insights.append(f"Model achieves {acc:.1f}% overall accuracy")
            
            # Insight 2: Class balance
            class_dist = Counter(y_true)
            if len(class_dist) > 0:
                min_class = min(class_dist.values())
                max_class = max(class_dist.values())
                balance_ratio = min_class / max_class if max_class > 0 else 0
                
                if balance_ratio > 0.7:
                    insights.append("Dataset shows good class balance")
                elif balance_ratio > 0.4:
                    insights.append("Moderate class imbalance detected")
                else:
                    insights.append("Significant class imbalance - consider balancing techniques")
            
            # Insight 3: Prediction consistency
            cm = np.array(metrics['model_diagnostics'].get('confusion_matrix', []))
            if cm.size > 0:
                diag_sum = np.trace(cm)
                total = cm.sum()
                if total > 0:
                    correct_rate = diag_sum / total
                    insights.append(f"{correct_rate:.1%} of predictions are correct")
            
            return insights
        except Exception as e:
            logger.error(f"[ERROR] Error generating key insights: {e}")
            return ["Insight generation failed"]

class ComprehensiveDataAnalyzer:
    """Enhanced data analysis with comprehensive statistics and insights"""
    
    @staticmethod
    def analyze_dataset_characteristics(df):
        """Comprehensive dataset analysis with detailed findings"""
        if df is None or df.empty:
            logger.warning("[WARNING] Empty or None dataframe provided for analysis")
            return {
                'basic_statistics': {
                    'total_samples': 0,
                    'total_features': 0,
                    'data_types': {},
                    'missing_values': 0
                },
                'data_health_score': {
                    'overall_health_score': 0,
                    'health_tier': 'Unknown',
                    'component_scores': {},
                    'health_interpretation': 'No data available for analysis'
                }
            }
        
        try:
            analysis = {}
            
            # Basic statistics
            analysis['basic_statistics'] = {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'data_types': df.dtypes.apply(lambda x: str(x)).to_dict(),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 100
            }
            
            # Data health scoring
            health_scores = {}
            
            # 1. Completeness score
            missing_percentage = analysis['basic_statistics']['missing_percentage']
            completeness_score = max(0, 100 - missing_percentage)
            health_scores['completeness'] = {
                'score': completeness_score,
                'status': 'Excellent' if completeness_score > 90 else 
                         'Good' if completeness_score > 75 else 
                         'Fair' if completeness_score > 50 else 'Poor'
            }
            
            # 2. Diversity score (unique values per column)
            diversity_scores = []
            for col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                    diversity_scores.append(min(100, unique_ratio * 100))
                else:
                    # For numeric columns, use coefficient of variation
                    if len(df[col]) > 0 and df[col].std() > 0 and df[col].mean() != 0:
                        cv = df[col].std() / abs(df[col].mean())
                        diversity_scores.append(min(100, 100 / (1 + cv)))
                    else:
                        diversity_scores.append(50)
            
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 50
            health_scores['diversity'] = {
                'score': avg_diversity,
                'status': 'Excellent' if avg_diversity > 70 else 
                         'Good' if avg_diversity > 50 else 
                         'Fair' if avg_diversity > 30 else 'Poor'
            }
            
            # 3. Balance score for categorical columns
            balance_scores = []
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols[:5]:  # Check first 5 categorical columns
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 1:
                    # Use Gini impurity as balance measure
                    gini = 1 - sum(value_counts ** 2)
                    balance_scores.append(gini * 100)
            
            balance_score = np.mean(balance_scores) if balance_scores else 50
            health_scores['balance'] = {
                'score': balance_score,
                'status': 'Excellent' if balance_score > 70 else 
                         'Good' if balance_score > 50 else 
                         'Fair' if balance_score > 30 else 'Poor'
            }
            
            # Calculate overall health score
            weights = {'completeness': 0.4, 'diversity': 0.3, 'balance': 0.3}
            overall_score = sum(health_scores[k]['score'] * weights[k] for k in weights.keys())
            
            # Determine health tier
            if overall_score >= 80:
                health_tier = 'Excellent'
            elif overall_score >= 60:
                health_tier = 'Good'
            elif overall_score >= 40:
                health_tier = 'Fair'
            else:
                health_tier = 'Poor'
            
            analysis['data_health_score'] = {
                'overall_health_score': overall_score,
                'health_tier': health_tier,
                'component_scores': health_scores,
                'health_interpretation': f"Dataset health is {health_tier.lower()} with an overall score of {overall_score:.1f}/100"
            }
            
            # Data quality insights
            analysis['quality_insights'] = {
                'missing_data_patterns': {
                    'total_missing': int(df.isnull().sum().sum()),
                    'columns_with_missing': (df.isnull().sum() > 0).sum(),
                    'recommendation': 'Consider imputation for missing values' if missing_percentage > 5 else 'Minimal missing data'
                },
                'data_type_analysis': {
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                    'recommendation': 'Good mix of data types' if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'Consider feature engineering'
                }
            }
            
            logger.info(f"[ANALYSIS] Dataset analysis completed: {len(df)} samples, {len(df.columns)} features")
            logger.info(f"[ANALYSIS] Health score: {overall_score:.1f}/100 ({health_tier})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"[ERROR] Dataset analysis failed: {e}")
            return {
                'basic_statistics': {
                    'total_samples': len(df) if df is not None else 0,
                    'total_features': len(df.columns) if df is not None else 0,
                    'error': str(e)
                },
                'data_health_score': {
                    'overall_health_score': 0,
                    'health_tier': 'Analysis Failed',
                    'component_scores': {},
                    'health_interpretation': f'Analysis failed: {str(e)}'
                }
            }

def generate_key_findings(metrics, model_results, dataset_analysis, important_features, class_names):
    """Generate comprehensive key findings for the model."""
    try:
        findings = {
            'performance_summary': f"Model achieves {metrics['basic_metrics']['accuracy']['value']:.1f}% accuracy",
            'best_model': max(model_results, key=model_results.get),
            'dataset_quality': dataset_analysis.get('data_health_score', {}).get('health_tier', 'Unknown'),
            'top_features': important_features[:3] if important_features else [],
            'class_distribution': f"{len(class_names)} classes identified",
            'recommendation': metrics['performance_interpretation'].get('recommendation', 'No recommendation')
        }
        
        return findings
    except Exception as e:
        logger.error(f"[ERROR] Error generating key findings: {e}")
        return {
            'performance_summary': 'Findings generation failed',
            'best_model': 'Unknown',
            'dataset_quality': 'Unknown',
            'top_features': [],
            'class_distribution': 'Unknown',
            'recommendation': 'Analysis failed'
        }

def generate_detailed_findings_report(metrics, model_results, dataset_analysis, 
                                    important_features, training_summary, version):
    """Generate a comprehensive findings report."""
    try:
        report = {
            'executive_summary': {
                'model_version': version,
                'training_date': datetime.utcnow().isoformat(),
                'best_model': max(model_results, key=model_results.get) if model_results else 'None',
                'overall_accuracy': metrics['basic_metrics']['accuracy']['value'],
                'performance_tier': metrics['performance_interpretation']['overall_performance_tier'],
                'deployment_readiness': 'Ready' if metrics['basic_metrics']['accuracy']['value'] > 70 else 'Needs Improvement'
            },
            'data_quality_assessment': {
                'health_score': dataset_analysis.get('data_health_score', {}).get('overall_health_score', 0),
                'health_tier': dataset_analysis.get('data_health_score', {}).get('health_tier', 'Unknown'),
                'completeness_analysis': {
                    'missing_percentage': dataset_analysis.get('basic_statistics', {}).get('missing_percentage', 0),
                    'assessment': 'Complete' if dataset_analysis.get('basic_statistics', {}).get('missing_percentage', 100) < 5 else 'Incomplete'
                }
            },
            'model_performance': {
                'comparative_results': model_results,
                'detailed_metrics': metrics['basic_metrics'],
                'advanced_metrics': metrics['advanced_metrics']
            },
            'feature_analysis': {
                'top_predictors': important_features[:10] if important_features else [],
                'total_features_analyzed': len(important_features) if important_features else 0
            },
            'training_details': training_summary
        }
        
        return report
    except Exception as e:
        logger.error(f"[ERROR] Error generating detailed findings report: {e}")
        return {
            'executive_summary': {
                'error': f'Report generation failed: {str(e)}',
                'model_version': version,
                'training_date': datetime.utcnow().isoformat()
            }
        }

class AnalyticsPipeline:
    """Orchestrates analytics generation and reporting"""
    
    def __init__(self, analytics_dir="analytics"):
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_analytics_report(self, analytics_report, version):
        """Save analytics report with enhanced serialization."""
        analytics_path = self.analytics_dir / f"training_analytics_v{version}.json"
        
        try:
            # Clean and convert the analytics report
            cleaned_report = clean_analytics_report(analytics_report)
            final_report = ensure_string_keys(cleaned_report)
            
            # Test serialization
            json.dumps(final_report)
            
            # Save the cleaned report
            with open(analytics_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"[SAVE] Analytics report saved successfully: {analytics_path}")
            return analytics_path
            
        except Exception as json_error:
            logger.error(f"[ERROR] JSON serialization failed: {json_error}")
            # Save minimal report
            minimal_report = {
                'version': version,
                'training_date': datetime.utcnow().isoformat(),
                'best_model': analytics_report.get('best_model', 'Unknown'),
                'accuracy': analytics_report.get('accuracy', 0),
                'simple_explanation': "Adaptive burnout prediction model",
                'error': f"Full report unavailable: {str(json_error)}"
            }
            with open(analytics_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_report, f, indent=2, ensure_ascii=False)
            return analytics_path
    
    def create_analytics_report(self, version, description, findings_report, 
                              detailed_metrics, model_comparison, cv_scores,
                              dataset_analysis, important_features, key_findings,
                              adaptive_model_explanation, model_configs):
        """Create comprehensive analytics report."""
        
        analytics_report = {
            'version': version,
            'training_date': datetime.utcnow().isoformat(),
            'description': description,
            'simple_description': "Adaptive AI model that learns from student surveys to predict burnout risk",
            'findings_report': findings_report,
            'detailed_metrics': detailed_metrics,
            'model_comparison': model_comparison,
            'cv_scores': cv_scores,
            'dataset_analysis': dataset_analysis,
            'important_features': important_features,
            'key_findings': key_findings,
            'adaptive_model_explanation': adaptive_model_explanation,
            'model_configs': {
                name: {
                    'simple_explanation': config.get('simple_explanation', 'No explanation'),
                    'description': config.get('description', 'No description')
                }
                for name, config in model_configs.items()
            }
        }
        
        return analytics_report