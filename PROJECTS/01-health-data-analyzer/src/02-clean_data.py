"""
Data Cleaning Module for Diabetes Dataset
Handles missing values (zeros) and prepares data for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_data(filepath):
    """Load the diabetes dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded: {filepath}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found - {filepath}")
        sys.exit(1)


def identify_zeros(df):
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = {}

    print("\n" + "="*60)
    print("🔍 ZERO VALUES ANALYSIS . . .")
    print("="*60)
    
    print(f"{'Column':20s} | {'Zero Count':10s} | {'Percentage':10s}")
    print("-"*50)

    for col in zero_columns:
        if col in df.columns:
            count = (df[col] == 0).sum()
            percentage = (count / len(df)) * 100
            zero_counts[col] = count
            
            if count > 0:
                print(f"{col:20s} | {count:10d} | {percentage:10.1f}%")
    
    return zero_counts



def replace_zeros_with_nan(df, columns):
    """
    Replace zeros with NaN for specified columns
    
    Args:
        df: DataFrame
        columns: List of column names to process
        
    Returns:
        DataFrame with zeros replaced by NaN
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
    
    return df_clean


def show_missing_summary(df):
    """Display summary of missing values"""
    print("\n" + "="*60)
    print("📊 MISSING VALUES SUMMARY")
    print("="*60)
    
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values!")
    else:
        print("\n  Column                 Missing    Percentage")
        print("  " + "-"*50)
        for col in missing[missing > 0].index:
            count = missing[col]
            pct = (count / len(df)) * 100
            print(f"  {col:20s}  {count:6d}     {pct:5.1f}%")
        print(f"\n  Total missing cells: {missing.sum()}")


def impute_with_median(df, columns):
    """
    Impute missing values with median for specified columns
    
    Strategy: Use median because it's robust to outliers
    """
    df_imputed = df.copy()
    
    print("\n" + "="*60)
    print("🔧 IMPUTATION WITH MEDIAN")
    print("="*60)
    
    for col in columns:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            median_value = df_imputed[col].median()
            missing_count = df_imputed[col].isnull().sum()
            df_imputed[col] = df_imputed[col].fillna(median_value)
            print(f"  {col:20s}: filled {missing_count:3d} values with {median_value:.1f}")
    
    return df_imputed


def compare_before_after(df_original, df_clean):
    """Compare statistics before and after cleaning"""
    print("\n" + "="*60)
    print("📈 BEFORE vs AFTER STATISTICS")
    print("="*60)
    
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n  Column               Before Mean   After Mean    Difference")
    print("  " + "-"*60)
    
    for col in cols_to_check:
        if col in df_original.columns:
            before_mean = df_original[col].mean()
            after_mean = df_clean[col].mean()
            diff = after_mean - before_mean
            print(f"  {col:18s}   {before_mean:10.2f}   {after_mean:10.2f}   {diff:+10.2f}")


def save_cleaned_data(df, output_path):
    """Save cleaned dataset"""
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")


def main():
    """Main execution"""
    print("🧹 DIABETES DATA CLEANING PIPELINE")
    print("="*60)
    
    # File paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'diabetes.csv'
    output_file = data_dir / 'diabetes_cleaned.csv'
    
    # Step 1: Load data
    df = load_data(input_file)
    df_original = df.copy()  # Keep original for comparison
    
    # Step 2: Identify zeros
    zero_counts = identify_zeros(df)
    
    # Step 3: Replace zeros with NaN
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = replace_zeros_with_nan(df, columns_to_clean)
    
    # Step 4: Show missing values
    show_missing_summary(df_clean)
    
    # Step 5: Impute with median
    df_clean = impute_with_median(df_clean, columns_to_clean)
    
    # Step 6: Verify no missing values remain
    show_missing_summary(df_clean)
    
    # Step 7: Compare before/after
    compare_before_after(df_original, df_clean)
    
    # Step 8: Save cleaned data
    save_cleaned_data(df_clean, output_file)
    
    print("\n" + "="*60)
    print("✅ DATA CLEANING COMPLETED!")
    print("="*60)
    print(f"  Original shape: {df_original.shape}")
    print(f"  Cleaned shape:  {df_clean.shape}")
    print(f"  Rows removed:   0")
    print(f"  Values imputed: {df_original.isnull().sum().sum()} → {df_clean.isnull().sum().sum()}")
    print("="*60)


if __name__ == "__main__":
    main()
