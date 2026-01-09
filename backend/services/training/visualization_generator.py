"""
Visualization Generator Module
Creates comprehensive visualizations for model evaluation.
"""
import io
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# Color palette
COLOR_PALETTE = {
    'primary': '#2E7D32',
    'secondary': '#1976D2',
    'accent': '#F57C00',
    'danger': '#D32F2F',
    'success': '#388E3C',
    'warning': '#FFA000',
    'info': '#0288D1',
    'neutral': '#757575',
    'light_gray': '#E0E0E0'
}

class VisualizationGenerator:
    """Generates comprehensive visualizations for model analysis"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def create_comprehensive_visualizations(self, clf, X_train, X_test, y_train, y_test, y_pred,
                                          model_results, feature_names, class_names,
                                          version, best_model_name, dataset_analysis=None):
        """Create comprehensive visualizations for model evaluation."""
        visualizations = {}
        
        try:
            # 1. Confusion Matrix Analysis
            visualizations['comprehensive_confusion_matrix'] = self._create_confusion_matrix_visualization(
                y_test, y_pred, class_names, best_model_name, version
            )
            
            # 2. ROC Curves
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)
                visualizations['roc_curves'] = self._create_roc_curve_visualization(
                    y_test, y_proba, class_names, best_model_name, version
                )
            
            # 3. Learning Curve
            visualizations['learning_curve'] = self._create_learning_curve_visualization(
                clf, X_train, y_train, best_model_name, version
            )
            
            # 4. Feature Importance
            if hasattr(clf, 'feature_importances_'):
                feat_imp = sorted(
                    zip(feature_names, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True
                )[:20]
                important_features = [
                    {'feature': str(name), 'importance': float(imp)}
                    for name, imp in feat_imp
                ]
                visualizations['adaptive_model_explanation'] = self._create_adaptive_model_explanation_graph(
                    best_model_name, version, important_features
                )
            
            # 5. Model Comparison Dashboard
            visualizations['model_analysis_dashboard'] = self._create_model_dashboard(
                model_results, y_test, y_pred, class_names,
                clf, feature_names, best_model_name, version
            )
            
            # 6. Data Quality Analysis
            if dataset_analysis:
                visualizations['data_quality_analysis'] = self._create_data_quality_visualization(
                    dataset_analysis, version
                )
            
            # 7. Prediction Confidence
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)
                visualizations['prediction_confidence'] = self._create_prediction_confidence_visualization(
                    y_proba, class_names, best_model_name, version
                )
            
        except Exception as e:
            logger.error(f"[ERROR] Error creating visualizations: {e}")
        
        return visualizations
    
    def _create_confusion_matrix_visualization(self, y_test, y_pred, class_names, model_name, version):
        """Create confusion matrix visualization with analytics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Main confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=ax1, cbar_kws={'label': 'Number of Predictions'})
        
        ax1.set_title(f'Confusion Matrix - {model_name}\nVersion {version}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, 
                ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Normalized Confusion Matrix\n(True Label Basis)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Precision and recall by class
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        
        x = range(len(class_names))
        width = 0.35
        ax3.bar(x, precision, width, label='Precision', color=COLOR_PALETTE['primary'], alpha=0.7)
        ax3.bar([i + width for i in x], recall, width, label='Recall', color=COLOR_PALETTE['accent'], alpha=0.7)
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision and Recall by Class', fontsize=14, fontweight='bold')
        ax3.set_xticks([i + width/2 for i in x])
        ax3.set_xticklabels(class_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error analysis
        error_rates = []
        for i in range(len(class_names)):
            total = cm[i, :].sum()
            correct = cm[i, i]
            error_rate = (total - correct) / total if total > 0 else 0
            error_rates.append(error_rate)
        
        bars = ax4.bar(class_names, error_rates, color=COLOR_PALETTE['danger'], 
                      alpha=0.7, edgecolor='black')
        ax4.set_title('Error Rate by Class', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Error Rate', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        fig.suptitle(f'Comprehensive Confusion Matrix Analysis - Version {version}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return buf
    
    def _create_roc_curve_visualization(self, y_true, y_proba, class_names, model_name, version):
        """Create ROC curve visualization for multi-class classification."""
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Binarize the output for multi-class
            y_true_bin = label_binarize(y_true, classes=class_names)
            n_classes = len(class_names)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot ROC curve for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                # Choose color based on AUC performance
                if roc_auc > 0.9:
                    color = COLOR_PALETTE['success']
                elif roc_auc > 0.8:
                    color = COLOR_PALETTE['info']
                elif roc_auc > 0.7:
                    color = COLOR_PALETTE['warning']
                else:
                    color = COLOR_PALETTE['danger']
                
                ax.plot(fpr, tpr, lw=2, color=color,
                       label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
            ax.set_title(f'ROC Curves - {model_name}\nVersion {version}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add interpretation text
            avg_auc = np.mean([auc(roc_curve(y_true_bin[:, i], y_proba[:, i])[0], 
                                   roc_curve(y_true_bin[:, i], y_proba[:, i])[1]) 
                              for i in range(n_classes)])
            
            interpretation = f"Average AUC: {avg_auc:.3f}\n"
            if avg_auc > 0.9:
                interpretation += "Excellent discrimination ability"
            elif avg_auc > 0.8:
                interpretation += "Good discrimination ability"
            elif avg_auc > 0.7:
                interpretation += "Fair discrimination ability"
            else:
                interpretation += "Limited discrimination ability"
            
            ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] ROC curve visualization failed: {e}")
            return None
    
    def _create_learning_curve_visualization(self, model, X, y, model_name, version):
        """Create learning curve to show model adaptation with more data."""
        try:
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy', random_state=42
            )
            
            train_scores_mean = np.mean(train_scores, axis=1) * 100
            train_scores_std = np.std(train_scores, axis=1) * 100
            test_scores_mean = np.mean(test_scores, axis=1) * 100
            test_scores_std = np.std(test_scores, axis=1) * 100
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot training scores
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1,
                           color=COLOR_PALETTE['primary'], label='Training ± Std')
            ax.plot(train_sizes, train_scores_mean, 'o-', color=COLOR_PALETTE['primary'],
                   linewidth=2, markersize=8, label='Training Accuracy')
            
            # Plot test scores
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                           test_scores_mean + test_scores_std, alpha=0.1,
                           color=COLOR_PALETTE['accent'], label='CV ± Std')
            ax.plot(train_sizes, test_scores_mean, 'o-', color=COLOR_PALETTE['accent'],
                   linewidth=2, markersize=8, label='Cross-validation Accuracy')
            
            ax.set_xlabel('Number of Training Examples', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
            ax.set_title(f'Learning Curve - {model_name}\nShows Model Adaptation with Data Volume', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] Learning curve visualization failed: {e}")
            return None
    
    def _create_adaptive_model_explanation_graph(self, model_name, version, important_features):
        """Create visualization explaining adaptive model features."""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Extract top features
            features = [feat['feature'] for feat in important_features[:10]]
            importances = [feat['importance'] for feat in important_features[:10]]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color=COLOR_PALETTE['primary'], alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
            ax.set_title(f'Top 10 Predictive Features - {model_name}\nVersion {version}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add importance values on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', ha='left', va='center', fontsize=9)
            
            # Add explanatory text
            explanation = "These features were automatically identified as most important\nfor predicting burnout risk in the current dataset."
            ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] Adaptive model explanation graph failed: {e}")
            return None
    
    def _create_prediction_confidence_visualization(self, y_proba, class_names, model_name, version):
        """Create visualization showing prediction confidence distribution."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # 1. Confidence distribution histogram
            max_confidences = np.max(y_proba, axis=1)
            axes[0].hist(max_confidences, bins=20, color=COLOR_PALETTE['primary'], 
                        alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[0].set_title('Distribution of Prediction Confidence', 
                            fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Average confidence by class
            avg_confidence_by_class = []
            for i, class_name in enumerate(class_names):
                if i < y_proba.shape[1]:
                    class_confidences = y_proba[:, i]
                    avg_confidence_by_class.append(np.mean(class_confidences))
                else:
                    avg_confidence_by_class.append(0)
            
            bars = axes[1].bar(range(len(class_names)), avg_confidence_by_class,
                             color=COLOR_PALETTE['accent'], alpha=0.7)
            axes[1].set_xlabel('Classes', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
            axes[1].set_title('Average Confidence by Class', fontsize=13, fontweight='bold')
            axes[1].set_xticks(range(len(class_names)))
            axes[1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # 3. Confidence vs prediction correctness (placeholder)
            axes[2].text(0.5, 0.5, 'Confidence-Accuracy Analysis\nWould require true labels comparison',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[2].set_title('Confidence vs Accuracy', fontsize=13, fontweight='bold')
            axes[2].axis('off')
            
            # 4. Low confidence predictions analysis
            low_confidence_threshold = 0.7
            low_conf_predictions = np.sum(max_confidences < low_confidence_threshold)
            total_predictions = len(max_confidences)
            low_conf_percentage = (low_conf_predictions / total_predictions) * 100
            
            labels = ['Low Confidence', 'High Confidence']
            sizes = [low_conf_predictions, total_predictions - low_conf_predictions]
            colors = [COLOR_PALETTE['warning'], COLOR_PALETTE['success']]
            
            axes[3].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, wedgeprops={'edgecolor': 'black'})
            axes[3].set_title(f'Predictions by Confidence Level\n(Threshold: {low_confidence_threshold})',
                            fontsize=13, fontweight='bold')
            
            fig.suptitle(f'Prediction Confidence Analysis - {model_name} (v{version})',
                        fontsize=16, fontweight='bold', y=0.98)
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] Prediction confidence visualization failed: {e}")
            return None
    
    def _create_model_dashboard(self, model_results, y_test, y_pred, class_names,
                              model, feature_names, best_model_name, version):
        """Create comprehensive model analysis dashboard."""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Model comparison bar chart
            ax1 = fig.add_subplot(gs[0, :])
            models = list(model_results.keys())
            accuracies = list(model_results.values())
            
            bars = ax1.bar(models, accuracies, color=COLOR_PALETTE['primary'], alpha=0.7)
            ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim([0, 100])
            
            # Highlight best model
            best_idx = models.index(best_model_name)
            bars[best_idx].set_color(COLOR_PALETTE['success'])
            bars[best_idx].set_alpha(0.9)
            
            # Add accuracy values on bars
            for bar, accuracy in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 2. Feature importance if available
            ax2 = fig.add_subplot(gs[1, 0])
            if hasattr(model, 'feature_importances_'):
                top_n = min(8, len(feature_names))
                feat_imp = sorted(zip(feature_names, model.feature_importances_),
                                key=lambda x: x[1], reverse=True)[:top_n]
                
                features = [name for name, _ in feat_imp]
                importances = [imp for _, imp in feat_imp]
                
                y_pos = np.arange(len(features))
                ax2.barh(y_pos, importances, color=COLOR_PALETTE['accent'])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(features)
                ax2.set_xlabel('Importance')
                ax2.set_title(f'Top {top_n} Features')
                ax2.invert_yaxis()
            else:
                ax2.text(0.5, 0.5, 'Feature Importance\nNot Available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Feature Importance')
            
            # 3. Class distribution
            ax3 = fig.add_subplot(gs[1, 1])
            unique_classes, class_counts = np.unique(y_test, return_counts=True)
            ax3.pie(class_counts, labels=unique_classes, autopct='%1.1f%%',
                   colors=sns.color_palette("husl", len(unique_classes)))
            ax3.set_title('Class Distribution in Test Set')
            
            # 4. Metrics summary
            ax4 = fig.add_subplot(gs[1, 2])
            metrics_text = f"Best Model: {best_model_name}\n"
            metrics_text += f"Version: {version}\n\n"
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            try:
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                metrics_text += f"Precision: {precision:.3f}\n"
                metrics_text += f"Recall: {recall:.3f}\n"
                metrics_text += f"F1-Score: {f1:.3f}\n"
            except:
                metrics_text += "Metrics calculation failed\n"
            
            ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='center')
            ax4.set_title('Performance Metrics')
            ax4.axis('off')
            
            # 5. Additional analysis placeholder
            ax5 = fig.add_subplot(gs[2, :])
            analysis_text = f"Model Analysis Dashboard\n"
            analysis_text += f"• Best model selected: {best_model_name}\n"
            analysis_text += f"• Test accuracy: {model_results[best_model_name]:.1f}%\n"
            analysis_text += f"• Number of classes: {len(class_names)}\n"
            analysis_text += f"• Version: {version}"
            
            ax5.text(0.1, 0.5, analysis_text, transform=ax5.transAxes,
                    fontsize=12, verticalalignment='center')
            ax5.set_title('Model Analysis Summary')
            ax5.axis('off')
            
            fig.suptitle(f'Model Analysis Dashboard - Version {version}',
                        fontsize=18, fontweight='bold', y=0.95)
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] Model dashboard creation failed: {e}")
            return None
    
    def _create_data_quality_visualization(self, dataset_analysis, version):
        """Create visualization showing data quality analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # 1. Health score gauge
            ax1 = axes[0]
            health_score = dataset_analysis['data_health_score']['overall_health_score']
            health_tier = dataset_analysis['data_health_score']['health_tier']
            
            # Create gauge chart
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            ax1.plot(theta, r, color='black', linewidth=2)
            
            # Fill based on health score
            fill_theta = np.linspace(0, np.pi * (health_score / 100), 100)
            ax1.fill_between(fill_theta, 0, 1, alpha=0.3, color=COLOR_PALETTE['primary'])
            
            ax1.set_title(f'Dataset Health Score: {health_score:.1f}/100', fontsize=14, fontweight='bold')
            ax1.text(0, -0.2, f'Tier: {health_tier}', ha='center', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 2. Component scores
            ax2 = axes[1]
            component_scores = dataset_analysis['data_health_score'].get('component_scores', {})
            components = list(component_scores.keys())
            scores = [component_scores[c].get('score', 0) for c in components]
            colors = [COLOR_PALETTE['success'] if s > 70 else 
                     COLOR_PALETTE['info'] if s > 50 else 
                     COLOR_PALETTE['warning'] if s > 30 else 
                     COLOR_PALETTE['danger'] for s in scores]
            
            bars = ax2.bar(components, scores, color=colors, alpha=0.7)
            ax2.set_xlabel('Components', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax2.set_title('Component Health Scores', fontsize=14, fontweight='bold')
            ax2.set_ylim([0, 100])
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Missing data analysis
            ax3 = axes[2]
            missing_percentage = dataset_analysis.get('basic_statistics', {}).get('missing_percentage', 0)
            labels = ['Complete Data', 'Missing Data']
            sizes = [100 - missing_percentage, missing_percentage]
            colors_pie = [COLOR_PALETTE['success'], COLOR_PALETTE['warning']]
            
            ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'black'})
            ax3.set_title('Data Completeness', fontsize=14, fontweight='bold')
            
            # 4. Data type distribution
            ax4 = axes[3]
            basic_stats = dataset_analysis.get('basic_statistics', {})
            data_types = basic_stats.get('data_types', {})
            
            if data_types:
                type_counts = {}
                for dtype in data_types.values():
                    dtype_str = str(dtype)
                    if 'int' in dtype_str or 'float' in dtype_str:
                        type_counts['Numeric'] = type_counts.get('Numeric', 0) + 1
                    elif 'object' in dtype_str or 'category' in dtype_str:
                        type_counts['Categorical'] = type_counts.get('Categorical', 0) + 1
                    else:
                        type_counts['Other'] = type_counts.get('Other', 0) + 1
                
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                
                ax4.bar(types, counts, color=COLOR_PALETTE['accent'], alpha=0.7)
                ax4.set_xlabel('Data Type', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax4.set_title('Data Type Distribution', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'Data Type Analysis\nNot Available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Data Type Distribution')
            
            fig.suptitle(f'Data Quality Analysis - Version {version}',
                        fontsize=18, fontweight='bold', y=0.95)
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return buf
        except Exception as e:
            logger.warning(f"[WARNING] Data quality visualization failed: {e}")
            return None