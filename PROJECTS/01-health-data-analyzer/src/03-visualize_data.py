"""
Data Visualization Module for Diabetes Dataset
Creates comprehensive visualizations for exploratory analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(filepath):
    """Load cleaned dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded: {filepath}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found - {filepath}")
        print("   Please run clean_data.py first!")
        sys.exit(1)


def plot_outcome_distribution(df, save_path):
    """Plot the distribution of diabetes outcomes"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    outcome_counts = df['Outcome'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(['Healthy', 'Diabetes'], outcome_counts.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(outcome_counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pie chart
    axes[1].pie(outcome_counts.values, labels=['Healthy', 'Diabetes'], autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05),
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Outcome Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '01_outcome_distribution.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 01_outcome_distribution.png")
    plt.show()
    plt.close()


def plot_feature_distributions(df, save_path):
    """Plot distributions of all numeric features"""
    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Histogram with KDE
        axes[idx].hist(df[col], bins=30, alpha=0.6, color='skyblue', edgecolor='black')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.1f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.1f}')
        
        axes[idx].set_xlabel(f"{col}\n--------------------", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[8])
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path / '02_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 02_feature_distributions.png")
    plt.show()
    plt.close()


def plot_outcome_comparison(df, save_path):
    """Compare features between healthy and diabetes groups"""
    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Box plot
        df.boxplot(column=col, by='Outcome', ax=axes[idx], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        axes[idx].set_xlabel('Outcome (0=Healthy, 1=Diabetes)', fontsize=9, fontweight='bold')
        axes[idx].set_ylabel(col, fontsize=9, fontweight='bold')
        axes[idx].set_title('')  # Remove auto title
        axes[idx].grid(alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[8])
    
    plt.suptitle('Feature Comparison by Outcome', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / '03_outcome_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 03_outcome_comparison.png")
    plt.show()
    plt.close()


def plot_correlation_matrix(df, save_path):
    """Plot correlation matrix heatmap"""
    # Calculate correlation
    corr = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / '04_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 04_correlation_matrix.png")
    plt.show()
    plt.close()


def plot_age_analysis(df, save_path):
    """Detailed age analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Age distribution by outcome
    healthy = df[df['Outcome'] == 0]['Age']
    diabetes = df[df['Outcome'] == 1]['Age']
    
    axes[0].hist(healthy, bins=20, alpha=0.6, label='Healthy', color='green', edgecolor='black')
    axes[0].hist(diabetes, bins=20, alpha=0.6, label='Diabetes', color='red', edgecolor='black')
    axes[0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Age Distribution by Outcome', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Diabetes rate by age group
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '>50'])
    diabetes_rate = df.groupby('AgeGroup', observed=False)['Outcome'].mean() * 100
    
    axes[1].bar(range(len(diabetes_rate)), diabetes_rate.values, 
                color=['#3498db', '#9b59b6', '#e67e22', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(diabetes_rate)))
    axes[1].set_xticklabels(diabetes_rate.index, fontsize=11)
    axes[1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Diabetes Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Diabetes Rate by Age Group', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(diabetes_rate.values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '05_age_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 05_age_analysis.png")
    plt.show()
    plt.close()


def plot_glucose_bmi_scatter(df, save_path):
    """Scatter plot of Glucose vs BMI colored by outcome"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(df['BMI'], df['Glucose'], 
                        c=df['Outcome'], cmap='RdYlGn_r',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('BMI (kg/m²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
    ax.set_title('Glucose vs BMI (colored by Outcome)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Outcome (0 = Healthy | 1 = Diabetes)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '06_glucose_bmi_scatter.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: 06_glucose_bmi_scatter.png")
    plt.show()
    plt.close()


def create_summary_statistics(df, save_path):
    """Create and save summary statistics table"""
    summary = df.groupby('Outcome').describe().T
    
    # Save CSV
    summary.to_csv(save_path / 'summary_statistics.csv')
    print("💾 Saved: summary_statistics.csv")

    print("\n" + "=" * 60)
    print("🧮 SUMMARY STATISTICS BY OUTCOME")
    print("=" * 60)

    print(f"{'Feature':<25} | {'Statistic':<10} | {'Outcome 0':>12} | {'Outcome 1':>12}")
    print("-" * 60)

    features = summary.index.get_level_values(0).unique()

    for feature in features:
        rows = summary.loc[feature]
        middle = len(rows) // 2

        for i, (stat, row) in enumerate(rows.iterrows()):
            # print feature name only once at the middle row
            feature_label = feature if i == middle else ""
            print(f"{feature_label:<25} | {stat:<10} | {row[0]:>12.6f} | {row[1]:>12.6f}")

        print("-" * 60)

def main():
    """Main execution"""
    print("📊 DIABETES DATA VISUALIZATION")
    print("=" * 60)
    
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    input_file = data_dir / 'diabetes_cleaned.csv'
    
    # Load data
    df = load_data(input_file)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("🛠  Creating visualizations . . .")
    print("=" * 60)
    
    plot_outcome_distribution(df, results_dir)
    plot_feature_distributions(df, results_dir)
    plot_outcome_comparison(df, results_dir)
    plot_correlation_matrix(df, results_dir)
    plot_age_analysis(df, results_dir)
    plot_glucose_bmi_scatter(df, results_dir)
    
    # Summary statistics
    create_summary_statistics(df, results_dir)
    
    print("\n" + "=" * 60)
    print("✔️  VISUALIZATION COMPLETED!")
    print(f"   All figures saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()