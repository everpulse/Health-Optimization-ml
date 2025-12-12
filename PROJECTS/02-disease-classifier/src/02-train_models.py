"""
Model Training Module - PROFESSIONAL VERSION
Trains multiple ML models with proper validation and metrics
"""

import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Validation
from sklearn.model_selection import cross_val_score

# Handle imbalanced data
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_train_data(data_dir):
    """Load training data"""
    train_path = Path(data_dir) / 'train_data.csv'
    
    try:
        df = pd.read_csv(train_path)
        print(f"✅ Loaded training data: {train_path}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {train_path} not found!")
        print("   Please run 01-split_data.py first!")
        sys.exit(1)


def feature_engineering(df):
    """Create meaningful features"""
    print("\n" + "="*60)
    print("🛠️ FEATURE ENGINEERING")
    print("="*60)
    
    # Interaction features
    df['BMI_Age_Interaction'] = (df['BMI'] * df['Age']) / 100
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1)
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
    
    # Polynomial features for key variables
    df['BMI_Squared'] = df['BMI'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    
    # Risk categories
    df['Age_Risk'] = pd.cut(df['Age'], bins=[0, 30, 45, 100], labels=[0, 1, 2])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 25, 30, 100], labels=[0, 1, 2])
    df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 100, 126, 200], labels=[0, 1, 2])
    
    print(f"✅ Created 8 engineered features")
    print(f"   Total features: {df.shape[1] - 1}")
    
    return df


def prepare_train_data(df):
    """Separate features and target, and apply scaling"""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    print("\n" + "="*60)
    print("🎯 TRAINING DATA PREPARED")
    print("="*60)
    print(f"Features: {X.shape}")
    print(f"Target:   {y.shape}")
    print(f"Diabetic (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"Healthy  (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"\nFeatures scaled with StandardScaler")
    
    return X_scaled, y, scaler


def handle_imbalance(X, y):
    """Handle class imbalance with SMOTE"""
    print("\n" + "="*60)
    print("⚖️ HANDLING CLASS IMBALANCE")
    print("="*60)
    
    # Before SMOTE
    n_healthy = (y == 0).sum()
    n_diabetic = (y == 1).sum()
    total = len(y)
    print("\nBefore SMOTE:")
    print(f"  🔹 Healthy : {n_healthy} ({n_healthy/total*100:.1f}%)")
    print(f"  🔸 Diabetic: {n_diabetic} ({n_diabetic/total*100:.1f}%)")
    
    # SMOTE with careful parameters
    smote = SMOTE(
        sampling_strategy=0.85,
        random_state=42, 
        k_neighbors=5
    )
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # After SMOTE
    n_healthy_new = (y_balanced == 0).sum()
    n_diabetic_new = (y_balanced == 1).sum()
    total_new = len(y_balanced)
    print("\nAfter SMOTE:")
    print(f"  🔹 Healthy : {n_healthy_new} ({n_healthy_new/total_new*100:.1f}%)")
    print(f"  🔸 Diabetic: {n_diabetic_new} ({n_diabetic_new/total_new*100:.1f}%)")
    
    print(f"\n✅ Classes balanced to ~85% ratio")
    
    return X_balanced, y_balanced


def evaluate_training_performance(model, X_train, y_train, model_name):
    """Evaluate model on training data with full metrics"""
    y_pred = model.predict(X_train)
    
    # Calculate all metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1 = f1_score(y_train, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'confusion_matrix': cm
    }


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression"""
    model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        penalty='l2'
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'Logistic Regression')
    
    return model, metrics


def train_decision_tree(X_train, y_train):
    """Train Decision Tree"""
    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=15,
        criterion='entropy',
        random_state=42,
        ccp_alpha=0.01
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'Decision Tree')
    
    return model, metrics


def train_random_forest(X_train, y_train):
    """Train Random Forest"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        ccp_alpha=0.005
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'Random Forest')
    
    return model, metrics


def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors"""
    model = KNeighborsClassifier(
        n_neighbors=30,
        weights='uniform',
        metric='manhattan',
        p=1,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'K-Nearest Neighbors')
    
    return model, metrics


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting"""
    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=25,
        min_samples_leaf=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'Gradient Boosting')
    
    return model, metrics


def train_svm(X_train, y_train):
    """Train Support Vector Machine"""
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    metrics = evaluate_training_performance(model, X_train, y_train, 'Support Vector Machine')
    
    return model, metrics


def train_voting_classifier(models_dict, X_train, y_train):
    """Train Voting Classifier"""
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models_dict['random_forest']),
            ('gb', models_dict['gradient_boosting']),
            ('svm', models_dict['svm'])
        ],
        voting='soft',
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    metrics = evaluate_training_performance(voting_clf, X_train, y_train, 'Voting Classifier')
    
    return voting_clf, metrics


def save_model(model, model_name, models_dir):
    """Save trained model"""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / model_name
    joblib.dump(model, model_path)
    
    print(f"💾 Saved: {model_path}")


def create_visualizations(all_metrics, results_dir):
    """Create comprehensive visualizations"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Sort metrics by CV accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['CV_Mean'], reverse=True)
    model_names = [m['Model'] for m in sorted_metrics]
    
    # 1. Confusion Matrices (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    for idx, m in enumerate(sorted_metrics):
        row, col = idx // 4, idx % 4
        cm = m['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                   xticklabels=['Healthy', 'Diabetic'],
                   yticklabels=['Healthy', 'Diabetic'],
                   ax=axes[row, col], cbar=False)
        
        axes[row, col].set_title(f"{m['Model']}", fontweight='bold')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Comparison (All Metrics)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'CV_Mean']
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
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. CV Accuracy with Error Bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cv_means = [m['CV_Mean'] * 100 for m in sorted_metrics]
    cv_stds = [m['CV_Std'] * 100 for m in sorted_metrics]
    
    bars = ax.barh(model_names, cv_means, xerr=cv_stds, 
                   color='#3A86FF', alpha=0.7, capsize=5)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        ax.text(mean + std + 1, i, f'{mean:.2f}% ± {std:.2f}%', 
               va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Cross-Validation Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('5-Fold Cross-Validation Performance', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'cv_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training vs CV Accuracy (Overfitting Detection)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    train_acc = [m['Accuracy'] * 100 for m in sorted_metrics]
    cv_acc = [m['CV_Mean'] * 100 for m in sorted_metrics]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_acc, width, label='CV Accuracy', 
                   color='#4ECDC4', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Training vs Cross-Validation Accuracy (Overfitting Check)', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'overfitting_check.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualizations saved to: {results_dir}")
    print("   - confusion_matrices.png")
    print("   - performance_comparison.png")
    print("   - cv_accuracy.png")
    print("   - overfitting_check.png")


def print_training_summary(all_metrics):
    """Print comprehensive training summary table"""
    print("\n" + "="*80)
    print("📊 TRAINING PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort by CV accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['CV_Mean'], reverse=True)
    
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'CV-Acc':>10}")
    print("-" * 95)
    
    for m in sorted_metrics:
        print(f"{m['Model']:<30} "
              f"{m['Accuracy']*100:>9.2f}% "
              f"{m['Precision']*100:>9.2f}% "
              f"{m['Recall']*100:>9.2f}% "
              f"{m['F1_Score']*100:>9.2f}% "
              f"{m['CV_Mean']*100:>9.2f}%")


def main():
    """Main execution"""
    print("📋 MODEL TRAINING OUTPUT")
    print("="*60)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results'
    
    # Load and prepare data
    df_train = load_train_data(data_dir)
    df_train = feature_engineering(df_train)
    X_train, y_train, scaler = prepare_train_data(df_train)
    
    save_model(scaler, 'scaler.pkl', models_dir)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    
    # Train all models and collect metrics
    all_metrics = []
    models = {}
    
    print("\n" + "="*60)
    print("🚀 TRAINING MODELS...")
    print("="*60)
    
    model, metrics = train_logistic_regression(X_train_balanced, y_train_balanced)
    models['logistic_regression'] = model
    all_metrics.append(metrics)
    save_model(model, 'logistic_regression.pkl', models_dir)
    
    model, metrics = train_decision_tree(X_train_balanced, y_train_balanced)
    models['decision_tree'] = model
    all_metrics.append(metrics)
    save_model(model, 'decision_tree.pkl', models_dir)
    
    model, metrics = train_random_forest(X_train_balanced, y_train_balanced)
    models['random_forest'] = model
    all_metrics.append(metrics)
    save_model(model, 'random_forest.pkl', models_dir)
    
    model, metrics = train_knn(X_train_balanced, y_train_balanced)
    models['knn'] = model
    all_metrics.append(metrics)
    save_model(model, 'knn.pkl', models_dir)
    
    model, metrics = train_gradient_boosting(X_train_balanced, y_train_balanced)
    models['gradient_boosting'] = model
    all_metrics.append(metrics)
    save_model(model, 'gradient_boosting.pkl', models_dir)
    
    model, metrics = train_svm(X_train_balanced, y_train_balanced)
    models['svm'] = model
    all_metrics.append(metrics)
    save_model(model, 'svm.pkl', models_dir)
    
    voting_clf, metrics = train_voting_classifier(models, X_train_balanced, y_train_balanced)
    all_metrics.append(metrics)
    save_model(voting_clf, 'voting_classifier.pkl', models_dir)
    
    # Create visualizations
    create_visualizations(all_metrics, results_dir)
    
    # Print summary
    print_training_summary(all_metrics)
    
    print("\n" + "="*60)
    print("✔ TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Visualizations saved in: {results_dir}")


if __name__ == "__main__":
    main()