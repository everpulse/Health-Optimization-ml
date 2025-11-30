"""
Model Training Module for Disease Classification
Trains 4 different ML models and saves them
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
import sys
import os

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score


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


def prepare_data(df):
    """Separate features and target"""
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    print("\n" + "="*60)
    print("🎯 DATA PREPARED")
    print("="*60)
    print(f"Features: {X.shape}")
    print(f"Target:   {y.shape}")
    print(f"Positive cases (Diabetes): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"Negative cases (Healthy):  {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    
    return X, y


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("\n" + "="*60)
    print("📈 TRAINING: Logistic Regression")
    print("="*60)
    print("📚 About: Simple linear model for classification")
    print("💡 Best for: Linearly separable data, fast training")
    
    start_time = time.time()
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    print(f"✅ Training completed in {train_time:.2f} seconds")
    print(f"📊 Training accuracy: {train_accuracy*100:.2f}%")
    
    return model, train_accuracy, train_time


def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    print("\n" + "="*60)
    print("🌳 TRAINING: Decision Tree")
    print("="*60)
    print("📚 About: Tree-based model with if-then rules")
    print("💡 Best for: Easy to interpret, handles non-linear data")
    
    start_time = time.time()
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    print(f"✅ Training completed in {train_time:.2f} seconds")
    print(f"📊 Training accuracy: {train_accuracy*100:.2f}%")
    
    return model, train_accuracy, train_time


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("🏞️ TRAINING: Random Forest")
    print("="*60)
    print("📚 About: Ensemble of many decision trees")
    print("💡 Best for: High accuracy, robust to overfitting")
    
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    print(f"✅ Training completed in {train_time:.2f} seconds")
    print(f"📊 Training accuracy: {train_accuracy*100:.2f}%")
    
    return model, train_accuracy, train_time


def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors model"""
    print("\n" + "="*60)
    print("👥 TRAINING: K-Nearest Neighbors")
    print("="*60)
    print("📚 About: Classifies based on similar neighbors")
    print("💡 Best for: Simple concept, no training phase")
    
    start_time = time.time()
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    print(f"✅ Training completed in {train_time:.2f} seconds")
    print(f"📊 Training accuracy: {train_accuracy*100:.2f}%")
    
    return model, train_accuracy, train_time


def save_model(model, model_name, models_dir):
    """Save trained model to disk"""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    return model_path


def save_results_summary(results, output_dir):
    """Save training results summary"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df_results = pd.DataFrame(results)
    output_path = output_dir / 'training_results.csv'
    df_results.to_csv(output_path, index=False)
    
    return output_path


def main():
    """Main execution"""
    print("🤖 MACHINE LEARNING MODEL TRAINING")
    print("Training 4 different classification models...")
    print("="*60)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results'
    
    # Step 1: Load data
    df_train = load_train_data(data_dir)
    
    # Step 2: Prepare data
    X_train, y_train = prepare_data(df_train)
    
    # Step 3: Train all models
    results = []
    
    # Model 1: Logistic Regression
    lr_model, lr_acc, lr_time = train_logistic_regression(X_train, y_train)
    lr_path = save_model(lr_model, 'logistic_regression', models_dir)
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': lr_acc * 100,
        'Time': lr_time,
        'Saved_File': os.path.basename(lr_path)
    })
    
    # Model 2: Decision Tree
    dt_model, dt_acc, dt_time = train_decision_tree(X_train, y_train)
    dt_path = save_model(dt_model, 'decision_tree', models_dir)
    results.append({
        'Model': 'Decision Tree',
        'Accuracy': dt_acc * 100,
        'Time': dt_time,
        'Saved_File': os.path.basename(dt_path)
    })
    
    # Model 3: Random Forest
    rf_model, rf_acc, rf_time = train_random_forest(X_train, y_train)
    rf_path = save_model(rf_model, 'random_forest', models_dir)
    results.append({
        'Model': 'Random Forest',
        'Accuracy': rf_acc * 100,
        'Time': rf_time,
        'Saved_File': os.path.basename(rf_path)
    })
    
    # Model 4: K-Nearest Neighbors
    knn_model, knn_acc, knn_time = train_knn(X_train, y_train)
    knn_path = save_model(knn_model, 'knn', models_dir)
    results.append({
        'Model': 'K-Nearest Neighbors',
        'Accuracy': knn_acc * 100,
        'Time': knn_time,
        'Saved_File': os.path.basename(knn_path)
    })
    
    # Step 4: Summary with clean formatting
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    # Create DataFrame for display
    df_results = pd.DataFrame(results)
    
    # Format columns
    df_results['Accuracy'] = df_results['Accuracy'].apply(lambda x: f"{x:.2f}%")
    df_results['Time'] = df_results['Time'].apply(lambda x: f"{x:.3f}s")
    
    # Display with proper formatting
    print(f"\n{'Model':<25} {'Accuracy':>12} {'Time':>12} {'Saved File':<30}")
    print("-" * 60)
    for _, row in df_results.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:>12} {row['Time']:>12} {row['Saved_File']:<30}")
    
    # Save results (with original numeric values)
    results_for_csv = []
    for r in results:
        results_for_csv.append({
            'Model': r['Model'],
            'Training_Accuracy': f"{r['Accuracy']:.2f}%",
            'Training_Time': f"{r['Time']:.3f}s",
            'Saved_File': r['Saved_File']
        })
    
    results_path = save_results_summary(results_for_csv, results_dir)
    print(f"\n💾 Results saved to: {results_path}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Models saved in: {models_dir}")
    print(f"📊 Results saved in: {results_dir}")
    


if __name__ == "__main__":
    main()