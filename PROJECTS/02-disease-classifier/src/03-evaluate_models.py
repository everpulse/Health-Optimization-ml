"""
Model Evaluation Module - PROFESSIONAL VERSION
Evaluates all trained models on test data with comprehensive metrics
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc, 
    roc_auc_score
)


def load_test_data(data_dir):
    """Load test data"""
    test_path = Path(data_dir) / 'test_data.csv'
    
    try:
        df = pd.read_csv(test_path)
        print(f"✅ Loaded test data: {test_path}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {test_path} not found!")
        print("   Please run 01-split_data.py first!")
        sys.exit(1)


def feature_engineering(df):
    """Create the same features as in training"""
    print("\n" + "="*60)
    print("🛠️ FEATURE ENGINEERING (Test Data)")
    print("="*60)
    
    # Interaction features
    df['BMI_Age_Interaction'] = (df['BMI'] * df['Age']) / 100
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1)
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
    
    # Polynomial features
    df['BMI_Squared'] = df['BMI'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    
    # Risk categories
    df['Age_Risk'] = pd.cut(df['Age'], bins=[0, 30, 45, 100], labels=[0, 1, 2])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 25, 30, 100], labels=[0, 1, 2])
    df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 100, 126, 200], labels=[0, 1, 2])
    
    print(f"✅ Created 8 engineered features")
    print(f"   Total features: {df.shape[1] - 1}")
    
    return df


def prepare_test_data(df):
    """Separate features and target"""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    print("\n" + "="*60)
    print("🎯 TEST DATA PREPARED")
    print("="*60)
    print(f"Features: {X.shape}")
    print(f"Target:   {y.shape}")
    print(f"Diabetic (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"Healthy  (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    return X, y


def load_scaler(models_dir):
    """Load the fitted scaler from training"""
    scaler_path = Path(models_dir) / 'scaler.pkl'
    
    try:
        scaler = joblib.load(scaler_path)
        print(f"\n✅ Loaded scaler: {scaler_path}")
        return scaler
    except FileNotFoundError:
        print(f"❌ Error: {scaler_path} not found!")
        print("   Please run 02-train_models.py first!")
        sys.exit(1)


def scale_test_data(X_test, scaler):
    """Apply the same scaling transformation as training"""
    X_scaled = scaler.transform(X_test)
    X_scaled = pd.DataFrame(X_scaled, columns=X_test.columns)
    print("✅ Test features scaled with training scaler")
    return X_scaled


def load_model(model_name, models_dir):
    """Load a trained model"""
    model_path = Path(models_dir) / model_name
    
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"⚠️  Warning: {model_path} not found!")
        return None


def evaluate_model(model, X_test, y_test, model_display_name):
    """Evaluate a single model on test data"""
    if model is None:
        return None
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities (if available)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        y_pred_proba = None
        roc_auc = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'Model': model_display_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Specificity': specificity,
        'ROC_AUC': roc_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def evaluate_all_models(X_test, y_test, models_dir):
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("🧊 EVALUATING MODELS ON TEST DATA...")
    print("="*60)
    
    models_to_evaluate = [
        ('logistic_regression.pkl', 'Logistic Regression'),
        ('decision_tree.pkl', 'Decision Tree'),
        ('random_forest.pkl', 'Random Forest'),
        ('knn.pkl', 'K-Nearest Neighbors'),
        ('gradient_boosting.pkl', 'Gradient Boosting'),
        ('svm.pkl', 'Support Vector Machine'),
        ('voting_classifier.pkl', 'Voting Classifier')
    ]
    
    all_metrics = []
    
    for model_file, display_name in models_to_evaluate:
        model = load_model(model_file, models_dir)
        if model is not None:
            metrics = evaluate_model(model, X_test, y_test, display_name)
            all_metrics.append(metrics)
            print(f" Evaluated: {display_name}")
    
    return all_metrics


def create_test_visualizations(all_metrics, y_test, results_dir):
    """Create comprehensive test evaluation visualizations"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Sort metrics by accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['Accuracy'], reverse=True)
    model_names = [m['Model'] for m in sorted_metrics]
    
    # 1. Confusion Matrices (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Test Set Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    for idx, m in enumerate(sorted_metrics):
        row, col = idx // 4, idx % 4
        cm = m['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                   xticklabels=['Healthy', 'Diabetic'],
                   yticklabels=['Healthy', 'Diabetic'],
                   ax=axes[row, col], cbar=False)
        
        axes[row, col].set_title(f"{m['Model']}",fontweight='bold')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'test_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Comparison (All Metrics)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D']
    
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] * 100 for m in sorted_metrics]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' '), 
                     color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Test Set Performance Comparison - All Metrics', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors_roc = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    for idx, m in enumerate(sorted_metrics):
        if m['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, m['y_pred_proba'])
            roc_auc = m['ROC_AUC']
            ax.plot(fpr, tpr, color=colors_roc[idx], lw=2, 
                   label=f"{m['Model']} (AUC = {roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curves - Test Set', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'test_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy Ranking
    fig, ax = plt.subplots(figsize=(12, 6))
    
    accuracies = [m['Accuracy'] * 100 for m in sorted_metrics]
    colors_bars = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 70 else '#e74c3c' 
                   for acc in accuracies]
    
    bars = ax.barh(model_names, accuracies, color=colors_bars, alpha=0.8)
    
    # Add value labels
    for i, acc in enumerate(accuracies):
        ax.text(acc + 1, i, f'{acc:.2f}%', 
               va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Ranking by Test Accuracy', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    # Add performance zones
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='Excellent (>80%)')
    ax.axvline(x=70, color='orange', linestyle='--', alpha=0.3, label='Good (>70%)')
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'test_accuracy_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Precision-Recall Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precisions = [m['Precision'] * 100 for m in sorted_metrics]
    recalls = [m['Recall'] * 100 for m in sorted_metrics]
    
    scatter = ax.scatter(recalls, precisions, c=range(len(model_names)), 
                        cmap='viridis', s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(model_names):
        ax.annotate(name, (recalls[i], precisions[i]), 
                   fontsize=9, ha='right', va='bottom')
    
    ax.set_xlabel('Recall (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Precision (%)', fontweight='bold', fontsize=12)
    ax.set_title('Precision vs Recall Trade-off', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(results_dir / 'test_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Test visualizations saved to: {results_dir}")
    print("   - test_confusion_matrices.png")
    print("   - test_performance_comparison.png")
    print("   - test_roc_curves.png")
    print("   - test_accuracy_ranking.png")
    print("   - test_precision_recall.png")


def print_test_summary(all_metrics):
    """Print comprehensive test evaluation summary"""
    print("\n" + "="*60)
    print("☰ TEST SET PERFORMANCE SUMMARY")
    print("="*60)
    
    # Sort by accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['Accuracy'], reverse=True)
    
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Specificity':>12} {'ROC-AUC':>10}")
    print("-" * 80)
    
    for m in sorted_metrics:
        roc_auc_str = f"{m['ROC_AUC']*100:.2f}%" if m['ROC_AUC'] is not None else "N/A"
        print(f"{m['Model']:<30} {m['Accuracy']*100:>9.2f}% {m['Precision']*100:>9.2f}% {m['Recall']*100:>9.2f}% {m['F1_Score']*100:>9.2f}% {m['Specificity']*100:>11.2f}% {roc_auc_str:>10}")
    
    # Best model
    best_model = sorted_metrics[0]
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model['Model']}")
    print("="*60)
    print(f"Test Accuracy:  {best_model['Accuracy']*100:.2f}%")
    print(f"Test Precision: {best_model['Precision']*100:.2f}%")
    print(f"Test Recall:    {best_model['Recall']*100:.2f}%")
    print(f"Test F1-Score:  {best_model['F1_Score']*100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (TN): {best_model['TN']}")
    print(f"  False Positives (FP): {best_model['FP']}")
    print(f"  False Negatives (FN): {best_model['FN']}")
    print(f"  True Positives  (TP): {best_model['TP']}")


def save_results_to_csv(all_metrics, results_dir):
    """Save test results to CSV"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Prepare data for CSV
    results_data = []
    for m in all_metrics:
        results_data.append({
            'Model': m['Model'],
            'Accuracy': m['Accuracy'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'F1_Score': m['F1_Score'],
            'Specificity': m['Specificity'],
            'ROC_AUC': m['ROC_AUC'],
            'True_Negatives': m['TN'],
            'False_Positives': m['FP'],
            'False_Negatives': m['FN'],
            'True_Positives': m['TP']
        })
    
    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('Accuracy', ascending=False)
    
    csv_path = results_dir / 'test_results.csv'
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n💾 Test results saved to: {csv_path}")


def main():
    """Main execution"""
    print("📋 MODEL EVALUATION ON TEST DATA")
    print("="*60)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results'
    
    # Load and prepare test data
    df_test = load_test_data(data_dir)
    df_test = feature_engineering(df_test)
    X_test, y_test = prepare_test_data(df_test)
    
    # Load scaler and scale test data
    scaler = load_scaler(models_dir)
    X_test_scaled = scale_test_data(X_test, scaler)
    
    # Evaluate all models
    all_metrics = evaluate_all_models(X_test_scaled, y_test, models_dir)
    
    # Create visualizations
    create_test_visualizations(all_metrics, y_test, results_dir)
    
    # Print summary
    print_test_summary(all_metrics)
    
    # Save results to CSV
    save_results_to_csv(all_metrics, results_dir)
    
    print("\n" + "="*60)
    print("✔ EVALUATION COMPLETED!")
    print("="*60)
    print(f"📁 Visualizations saved in: {results_dir}")
    print(f"💾 Results CSV saved in: {results_dir}")


if __name__ == "__main__":
    main()