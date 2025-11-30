# 🤖 Disease Classifier

Machine learning classification models for diabetes prediction using preprocessed clinical data.

## 🎯 Project Overview

This project implements multiple machine learning classification algorithms to predict diabetes outcomes based on clinical biomarkers. The project uses the cleaned dataset from the Health Data Analyzer and trains four different models for comparison.

**Objectives:**
- ✅ Split data into training and testing sets
- ✅ Train multiple classification models
- ✅ Compare model performance
- ✅ Save trained models for deployment

## 📊 Dataset

**Source:** Preprocessed data from `01-health-data-analyzer`
- **Training samples:** 614 (80%)
- **Testing samples:** 154 (20%)
- **Features:** 8 clinical measurements
- **Target:** Binary diabetes outcome (0=negative, 1=positive)
- **Class distribution:** ~65% healthy, ~35% diabetes (stratified split)

### Input Features

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
02-disease-classifier/
├── README.md
├── requirements.txt
├── data/
│   ├── diabetes_cleaned.csv    # Preprocessed dataset (from project 01)
│   ├── train_data.csv          # Training set (614 samples)
│   └── test_data.csv           # Testing set (154 samples)
├── src/
│   ├── 01-split_data.py        # Train/test split
│   └── 02-train_models.py      # Model training
├── models/
│   ├── logistic_regression.pkl  # Trained LR model
│   ├── decision_tree.pkl        # Trained DT model
│   ├── random_forest.pkl        # Trained RF model
│   └── knn.pkl                  # Trained KNN model
├── results/
│   └── training_results.csv     # Training metrics summary
└── notebooks/
    └── evaluation.ipynb         # Model evaluation (planned)
```

## 🚀 Setup & Usage

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Navigate to project directory
cd PROJECTS/02-disease-classifier

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1️⃣ Split Data into Train/Test Sets
```bash
python src/01-split_data.py
```
**Output:** 
- `data/train_data.csv` (614 samples)
- `data/test_data.csv` (154 samples)

**Details:**
- 80/20 train-test split
- Stratified sampling (maintains class distribution)
- Random state = 42 (reproducible)

---

#### 2️⃣ Train All Models
```bash
python src/02-train_models.py
```
**Output:**
- 4 trained models saved in `models/`
- Training metrics in `results/training_results.csv`

**Models Trained:**
1. **Logistic Regression** - Fast, interpretable baseline
2. **Decision Tree** - Rule-based, easy to visualize
3. **Random Forest** - Ensemble method, high accuracy
4. **K-Nearest Neighbors** - Instance-based learning

---

## 🤖 Models & Results

### Training Performance

| Model | Training Accuracy | Training Time |
|-------|------------------|---------------|
| Logistic Regression | 77.69% | ~0.01s |
| Decision Tree | 84.69% | ~0.01s |
| Random Forest | 81.92% | ~0.15s |
| K-Nearest Neighbors | 82.74% | ~0.00s |

> **Note:** These are training accuracies. Test performance evaluation is planned.

---

## 📈 Model Details

### 1. Logistic Regression
**Type:** Linear classifier  
**Pros:** Fast, interpretable, good baseline  
**Cons:** Assumes linear relationships  
**Hyperparameters:** `max_iter=1000`, `random_state=42`

---

### 2. Decision Tree
**Type:** Tree-based classifier  
**Pros:** Easy to interpret, handles non-linear data  
**Cons:** Can overfit without pruning  
**Hyperparameters:** `max_depth=5`, `random_state=42`

---

### 3. Random Forest
**Type:** Ensemble (multiple trees)  
**Pros:** High accuracy, robust  
**Cons:** Slower training, less interpretable  
**Hyperparameters:** `n_estimators=100`, `max_depth=5`, `random_state=42`

---

### 4. K-Nearest Neighbors
**Type:** Instance-based  
**Pros:** Simple concept, no training  
**Cons:** Slow prediction, sensitive to scale  
**Hyperparameters:** `n_neighbors=5`

---

## 🛠️ Technologies

- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib

## 📝 Next Steps

### Immediate Tasks
- [ ] Evaluate models on test set
- [ ] Generate confusion matrices
- [ ] Calculate precision, recall, F1-score
- [ ] Compare model performance

### Future Enhancements
- [ ] Cross-validation for robust evaluation
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature importance analysis
- [ ] ROC curves and AUC scores
- [ ] Model explainability (SHAP values)
- [ ] Jupyter notebook with visualizations
- [ ] Model deployment pipeline

## 🔗 Related Projects

**Previous:** [01-health-data-analyzer](../01-health-data-analyzer/) - Data cleaning and EDA  
**Next:** Model evaluation and deployment (planned)

## 🤝 Contributing

This is a portfolio project demonstrating machine learning workflow.

## 📄 License

This project is open source and available for educational purposes.

---

**Status:** ✅ Complete - Models Trained  
**Last Updated:** January 2025  
**Next Milestone:** Test Set Evaluation