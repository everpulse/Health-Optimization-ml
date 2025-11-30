# 🤖 Project 2: Disease Classifier

## 🎯 Goal
Learn machine learning basics by building classification models to predict diabetes.

## 📊 Dataset
Using the cleaned diabetes dataset from Project 1 (768 patients, 8 features)

## 🛠️ Task Checklist

### Phase 1: Data Preparation ✅
- [x] Copy cleaned dataset
- [x] Split data into train/test sets (80/20)
- [x] Understand train/test concept

### Phase 2: Model Training ✅
- [x] Logistic Regression (79.32% accuracy)
- [x] Decision Tree (81.16% accuracy)
- [x] Random Forest (84.85% accuracy)
- [x] K-Nearest Neighbors (80.13% accuracy)

### Phase 3: Evaluation 🔄
- [ ] Calculate accuracy, precision, recall
- [ ] Create confusion matrices
- [ ] Compare models
- [ ] Visualize results

### Phase 4: Prediction
- [ ] Make predictions on test data
- [ ] Save best model
- [ ] Test on new samples

## 🏃 How to Run
```bash
# Step 1: Split data
python src/01-split_data.py

# Step 2: Train models ✅
python src/02-train_models.py

# Step 3: Evaluate models (next)
python src/03-evaluate_models.py

# Step 4: Make predictions (coming soon)
python src/04-predict_new.py
```

## 📚 What I'm Learning
- scikit-learn basics ✅
- Train/test split concept ✅
- Classification algorithms ✅
- Model evaluation metrics (next)
- Making predictions

## 🔧 Common Issues
- **ImportError**: Run `pip install -r requirements.txt`
- **FileNotFoundError**: Make sure cleaned data exists in data/
- **Low accuracy**: Normal for first try! We'll improve it.

## 📈 Progress
- **Started:** 2025-11-24
- **Phase 1 Complete:** 2025-11-27 (~4 hours)
- **Phase 2 Complete:** 2025-11-29 (~4 hours)
- **Status:** ✅ Phase 2 - Model Training COMPLETE!
- **Best Model So Far:** Random Forest (84.85%)

## 📂 Project Structure
```
02-disease-classifier/
├── data/
│   ├── diabetes_cleaned.csv
│   ├── train_data.csv          ✅
│   └── test_data.csv           ✅
├── src/
│   ├── 01-split_data.py        ✅
│   ├── 02-train_models.py      ✅
│   ├── 03-evaluate_models.py   (next)
│   └── 04-predict_new.py
├── models/                     ✅ (4 trained models)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── knn.pkl
├── results/                    ✅
│   └── training_results.csv
└── README.md
```

---

**Total time invested:** ~8 hours (Phase 1: 4h, Phase 2: 4h)  
**Next milestone:** Evaluate models on test data (Phase 3)  
**Key achievement:** Successfully trained 4 ML algorithms! 🎉
