# 🏥 Health Data Analyzer

Exploratory data analysis and visualization pipeline for the Pima Indians Diabetes Database.

## 🎯 Project Overview

This project focuses on data exploration, cleaning, and visualization of clinical biomarkers to understand patterns and relationships in diabetes data.

**Objectives:**
- ✅ Load and validate diabetes dataset
- ✅ Assess data quality and identify issues
- ✅ Clean and preprocess data (handle missing values)
- ✅ Perform exploratory data analysis (EDA)
- ✅ Create comprehensive visualizations
- ✅ Generate statistical summaries

## 📊 Dataset

**Pima Indians Diabetes Database**
- **Source:** UCI Machine Learning Repository / Kaggle
- **Samples:** 768 female patients
- **Features:** 8 clinical measurements
- **Target:** Binary diabetes outcome (0=negative, 1=positive)

### Clinical Features

| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of pregnancies | count |
| Glucose | Plasma glucose concentration | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin fold thickness | mm |
| Insulin | 2-Hour serum insulin | mu U/ml |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Diabetes heredity score | score |
| Age | Age | years |

## 📁 Project Structure

```
01-health-data-analyzer/
├── README.md                   # Project documentation
├── data/
│   ├── diabetes.csv           # Original dataset
│   └── diabetes_cleaned.csv   # Preprocessed dataset
├── src/
│   ├── 01-load_data.py        # Data loading and exploration
│   ├── 02-clean_data.py       # Data cleaning and imputation
│   └── 03-visualize_data.py   # Statistical visualizations
├── notebooks/
│   └── analysis.ipynb         # Exploratory analysis (planned)
└── results/
    ├── 01_outcome_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_outcome_comparison.png
    ├── 04_correlation_matrix.png
    ├── 05_age_analysis.png
    ├── 06_glucose_bmi_scatter.png
    └── summary_statistics.csv
```

## 🚀 Setup & Usage

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd PROJECTS/01-health-data-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis Pipeline

#### 1️⃣ Load and Explore Data
```bash
python src/01-load_data.py
```
**Output:** Basic statistics, data types, missing values analysis

#### 2️⃣ Clean and Preprocess Data
```bash
python src/02-clean_data.py
```
**Output:** `data/diabetes_cleaned.csv` with imputed values

#### 3️⃣ Generate Visualizations
```bash
python src/03-visualize_data.py
```
**Output:** 6 visualization files + summary statistics in `results/`

## 🔬 Analysis Pipeline

### Phase 1: Data Loading & Exploration ✅
**Script:** `01-load_data.py`

- Data loading and validation
- Basic statistics and data types
- Initial quality assessment
- Class distribution analysis

**Key Findings:**
- 768 samples with 8 clinical features
- Binary outcome: 65% healthy, 35% diabetes
- Detected zero values in 5 features

---

### Phase 2: Data Cleaning ✅
**Script:** `02-clean_data.py`

- Identified zero values as missing data
- Applied median imputation strategy
- Generated cleaned dataset
- Compared before/after statistics

**Preprocessing Details:**
- Zero values detected: Glucose (0.7%), BloodPressure (4.6%), SkinThickness (29.6%), Insulin (48.7%), BMI (1.4%)
- Strategy: Median imputation (robust to outliers)
- Result: Complete dataset with 768 samples retained

---

### Phase 3: Visualization & Statistical Analysis ✅
**Script:** `03-visualize_data.py`

Generated visualizations:
1. **Outcome Distribution** - Class balance visualization
2. **Feature Distributions** - Histograms with mean/median
3. **Outcome Comparison** - Box plots by health status
4. **Correlation Matrix** - Feature relationships heatmap
5. **Age Analysis** - Diabetes rate by age group
6. **Glucose vs BMI Scatter** - Key predictor relationships

**Statistical Output:**
- Summary statistics by outcome (CSV)
- Mean, median, std, min/max for all features

## 📈 Key Insights

### Data Quality
- **Original Dataset:** 768 samples, 9 features (8 predictors + 1 outcome)
- **Missing Data Pattern:** Zero values in 5 features (biological impossibility)
- **Solution:** Median imputation preserves distribution while handling outliers

### Distribution Patterns
- **Age Distribution:** Right-skewed, median age ~29 years
- **Glucose Levels:** Higher in diabetes group (mean: ~142 vs ~110 mg/dL)
- **BMI:** Elevated in diabetes group (mean: ~35 vs ~30 kg/m²)

### Correlations
- **Strongest predictors:** Glucose (r=0.47), BMI (r=0.29), Age (r=0.24)
- **Feature relationships:** Moderate correlation between related features (Age-Pregnancies, SkinThickness-BMI)
- **Outcome correlation:** Glucose shows strongest association with diabetes

### Age-Based Risk
- **<30 years:** ~17% diabetes prevalence
- **30-40 years:** ~24% diabetes prevalence  
- **40-50 years:** ~34% diabetes prevalence
- **>50 years:** ~45% diabetes prevalence

Clear trend: Diabetes risk increases significantly with age.

## 📊 Visualizations

All visualizations are available in the `results/` directory:

1. **Outcome Distribution**: Class balance visualization
2. **Feature Distributions**: Histograms with mean/median for all features
3. **Outcome Comparison**: Box plots comparing healthy vs diabetes groups
4. **Correlation Matrix**: Heatmap showing feature relationships
5. **Age Analysis**: Age distribution and diabetes rate by age group
6. **Glucose vs BMI Scatter**: Relationship between two key predictors

## 🛠️ Technologies

- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (upcoming)
- **Analysis:** Jupyter Notebook (planned)

## 📚 References

1. [Pima Indians Diabetes Database - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
2. [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
3. Scikit-learn Documentation
4. Clinical diabetes diagnosis guidelines

## 🤝 Contributing

This is a portfolio project demonstrating data analysis and visualization skills.

## 📄 License

This project is open source and available for educational purposes.

---

**Status:** ✅ Complete  
**Last Updated:** January 2025  
**Next Project:** Machine Learning Classification (See: `02-disease-classifier`)