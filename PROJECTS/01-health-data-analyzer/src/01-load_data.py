"""
Data Loading and Initial Analysis Module for Diabetes Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_diabetes_data(filepath):
    """
    Load Pima Indians Diabetes Dataset
    
    Args:
        filepath (str or Path): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe with proper column names
    """
    columns = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age',
        'Outcome'
    ]
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath, names=columns, header=0)
    print(f"Successfully loaded: {filepath.name}")
    
    return df


def basic_info(df):
    """
    Display basic information about the dataset
    """
    print("\n" + "=" * 60)
    print("Dataset Overview")
    print("=" * 60)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\n" + "=" * 60)
    print("Data Types")
    print("=" * 60)
    print(df.dtypes)
    
    print("\n" + "=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)
    print(df.describe())
    
    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n" + "=" * 60)
    print("Class Distribution")
    print("=" * 60)
    if 'Outcome' in df.columns:
        outcome_counts = df['Outcome'].value_counts()
        total = len(df)
        print(f"Healthy (0): {outcome_counts.get(0, 0)} ({outcome_counts.get(0, 0)/total*100:.1f}%)")
        print(f"Diabetes (1): {outcome_counts.get(1, 0)} ({outcome_counts.get(1, 0)/total*100:.1f}%)")


def check_data_quality(df):
    """
    Check for data quality issues like zero values and outliers
    """
    print("\n" + "=" * 60)
    print("Data Quality Check")
    print("=" * 60)
    
    # Check for zero values in columns that shouldn't be zero
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\nZero values (potential missing data):")
    for col in zero_columns:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"  {col}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")
    
    # Check for unusual values
    print("\nValue ranges:")
    if 'Age' in df.columns:
        print(f"  Age: {df['Age'].min()} to {df['Age'].max()} years")
    if 'BMI' in df.columns:
        print(f"  BMI: {df['BMI'].min():.1f} to {df['BMI'].max():.1f}")
    if 'Glucose' in df.columns:
        print(f"  Glucose: {df['Glucose'].min()} to {df['Glucose'].max()}")


if __name__ == "__main__":
    """
    Main execution
    """
    print("Starting Diabetes Dataset Analysis...\n")
    
    try:
        # Try multiple possible paths
        possible_paths = [
            '../data/diabetes.csv',
            'data/diabetes.csv',
            '../../data/diabetes.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = load_diabetes_data(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError(
                "diabetes.csv not found in common paths!\n"
                "Please place the file in one of these locations:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )
        
        # Display information
        basic_info(df)
        
        # Check quality
        check_data_quality(df)
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure diabetes.csv is in the data/ folder.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
