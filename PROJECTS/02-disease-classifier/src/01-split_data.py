"""
Data Splitting Module for ML Training
Splits dataset into training and testing sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys


def load_data(filepath):
    """Load cleaned diabetes dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded: {filepath}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found - {filepath}")
        print("   Please copy diabetes_cleaned.csv to data/ folder!")
        sys.exit(1)


def split_features_target(df, target_column='Outcome'):
    """
    Split dataframe into features (X) and target (y)
    
    Args:
        df: DataFrame
        target_column: Name of target column
        
    Returns:
        X: Features (all columns except target)
        y: Target (outcome column)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print("\n" + "="*60)
    print("🎯 FEATURES & TARGET SEPARATION")
    print("="*60)
    print(f"Features (X): {X.shape[1]} columns")
    print(f"Target (y):   {y.shape[0]} samples")
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Target values:   {y.unique()}")
    
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set (default 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Keep same class distribution in train/test
    )
    
    print("\n" + "="*60)
    print("</>  TRAIN/TEST SPLIT")
    print("="*60)
    print(f"Total samples:     {len(X)}")
    print(f"Training samples:  {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Testing samples:   {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\n📊 Class distribution:")
    print(f"Train - Healthy: {(y_train==0).sum()}, Diabetes: {(y_train==1).sum()}")
    print(f"Test  - Healthy: {(y_test==0).sum()}, Diabetes: {(y_test==1).sum()}")
    
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir):
    """Save train/test splits to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine X and y for saving
    train_data = X_train.copy()
    train_data['Outcome'] = y_train.values
    
    test_data = X_test.copy()
    test_data['Outcome'] = y_test.values
    
    # Save
    train_path = output_dir / 'train_data.csv'
    test_path = output_dir / 'test_data.csv'
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print("\n" + "="*60)
    print("💾 SAVED SPLITS")
    print("="*60)
    print(f"✅ Training data: {train_path}")
    print(f"✅ Testing data:  {test_path}")

def show_sample_data(X_train, y_train, n=5):

    print("\n" + "="*60)
    print(f"SAMPLE TRAINING DATA (first {n} rows)")
    print("="*60)

    sample = X_train.head(n).copy()
    sample["Outcome"] = y_train.head(n).values
    sample = sample.reset_index().rename(columns={"index": "Row"})

    for i, row in sample.iterrows():
        print(
            f"Row {row['Row']:>3} | "
            f"P:{row['Pregnancies']:<2}  "
            f"G:{row['Glucose']:<5} "
            f"BP:{row['BloodPressure']:<5} "
            f"Skin:{row['SkinThickness']:<5} "
            f"Ins:{row['Insulin']:<5} "
            f"BMI:{row['BMI']:<5} "
            f"Dpf:{row['DiabetesPedigreeFunction']:<4} "
            f"Age:{row['Age']:<2} "
            f"Out:{row['Outcome']}"
        )     

def main():
    """Main execution"""
    print("𝄜  DATASET SPLITTING FOR MACHINE LEARNING")
    print("="*60)
    
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'diabetes_cleaned.csv'
    
    # Step 1: Load data
    df = load_data(input_file)
    
    # Step 2: Separate features and target
    X, y = split_features_target(df)
    
    # Step 3: Split into train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Step 4: Save splits
    save_splits(X_train, X_test, y_train, y_test, data_dir)
    
    # Step 5: Show sample
    show_sample_data(X_train, y_train)
    
    print("\n" + "="*60)
    print("✔️ DATA SPLITTING COMPLETED!")


if __name__ == "__main__":
    main()