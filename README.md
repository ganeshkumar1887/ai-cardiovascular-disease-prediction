# AI-Based Cardiovascular Disease Prediction

An end-to-end machine learning and neural network classification project for predicting cardiovascular disease risk from clinical, anthropometric, and lifestyle attributes. The project performs data cleaning, outlier analysis, feature engineering, stratified train-test evaluation, 5-fold cross-validation, permutation feature importance, and model persistence across four classifiers.

---

## Project Overview

- **Notebook:** [`ai-based-cardiovascular-disease-prediction.ipynb`](./ai-based-cardiovascular-disease-prediction.ipynb)
- **Project Report:** [`reports/Cardiovascular_Disease_Project_Report.pdf`](./reports/Cardiovascular_Disease_Project_Report.pdf)
- **Raw Dataset Size:** `70,000` records, `13` columns (`cardio_train.csv`)
- **Cleaned Dataset Size:** `68,624` records (`1,376` invalid/duplicate records removed)
- **Train / Test Split:** 80% / 20% stratified split (`54,899` training, `13,725` testing)

---

## Data Preprocessing & Feature Engineering

1. **Data Cleaning:**
   - Dropped `id` and converted `age` (in days) to `age_years`.
   - Removed duplicate records and validated `cardio` target values (`0`, `1`).
   - Enforced physiological bounds (`age_years`: 18–100, `height`: 100–250 cm, `weight`: 25–300 kg, `ap_hi`: 60–300, `ap_lo`: 30–200) and ensured systolic BP exceeds diastolic BP (`ap_hi > ap_lo`).
   - Validated categorical attributes (`gender`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`).
2. **Feature Engineering:**
   - `BMI` = $\text{weight} / (\text{height} / 100)^2$
   - `PulsePressure` = $\text{ap\_hi} - \text{ap\_lo}$
3. **Preprocessing Pipeline:**
   - Median imputation + `StandardScaler` for numeric features
   - Mode imputation + `OneHotEncoder` for categorical/nominal features

---

## Models Implemented

| Model | Key Configuration |
| :--- | :--- |
| **Artificial Neural Network (ANN)** | `MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', batch_size=128, max_iter=200, early_stopping=True)` |
| **Decision Tree** | `DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=20)` |
| **K-Nearest Neighbors (KNN)** | `KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')` |
| **Gaussian Naive Bayes** | `GaussianNB()` |

---

## Evaluation Results

### Test Set Performance (`13,725` samples)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN** *(Selected)* | **0.7277** | 0.7459 | 0.6818 | **0.7124** | **0.7937** | 7.70 |
| **Decision Tree** | 0.7238 | 0.7518 | 0.6594 | 0.7026 | 0.7847 | 0.29 |
| **KNN** | 0.7089 | 0.7157 | **0.6828** | 0.6989 | 0.7621 | 0.20 |
| **Naive Bayes** | 0.7081 | **0.7578** | 0.6027 | 0.6714 | 0.7762 | **0.09** |

### 5-Fold Stratified Cross-Validation

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANN** | **0.7341** | 0.7573 | 0.6816 | **0.7173** | **0.8002** |
| **Decision Tree** | 0.7313 | 0.7545 | 0.6777 | 0.7139 | 0.7942 |
| **KNN** | 0.7178 | 0.7246 | **0.6934** | 0.7086 | 0.7751 |
| **Naive Bayes** | 0.7163 | **0.7673** | 0.6124 | 0.6811 | 0.7810 |

### Healthcare Confusion Matrix Metrics (Test Set)

| Model | Sensitivity | Specificity | True Positives | True Negatives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN** | 0.6818 | 0.7726 | 4,630 | 5,357 | 1,577 | 2,161 |
| **Decision Tree** | 0.6594 | 0.7868 | 4,478 | 5,456 | 1,478 | 2,313 |
| **KNN** | 0.6828 | 0.7344 | 4,637 | 5,092 | 1,842 | 2,154 |
| **Naive Bayes** | 0.6027 | 0.8114 | 4,093 | 5,626 | 1,308 | 2,698 |

### Top Predictive Features (Permutation Importance)

1. `ap_hi` (Systolic Blood Pressure) — `0.0349`
2. `cholesterol` — `0.0323`
3. `age_years` — `0.0310`
4. `ap_lo` (Diastolic Blood Pressure) — `0.0308`
5. `PulsePressure` — `0.0193`

---

## Repository Structure

```text
.
├── ai-based-cardiovascular-disease-prediction.ipynb   # Main Jupyter notebook
├── README.md                                          # Project documentation
├── requirements.txt                                   # Python dependencies
├── .gitignore                                         # Git ignore rules
├── reports/
│   └── Cardiovascular_Disease_Project_Report.pdf      # Detailed project report
└── outputs/
    ├── best_cardiovascular_model.pkl                  # Best selected model (ANN)
    ├── ann_model.pkl                                  # Saved ANN pipeline
    ├── decision_tree_model.pkl                        # Saved Decision Tree pipeline
    ├── knn_model.pkl                                  # Saved KNN pipeline
    ├── naive_bayes_model.pkl                          # Saved Naive Bayes pipeline
    ├── cleaned_cardiovascular_data.csv                # Preprocessed dataset (68,624 rows)
    ├── patient_predictions.csv                        # Test predictions
    ├── model_evaluation.csv                           # Test evaluation metrics
    ├── cross_validation_results.csv                   # 5-fold CV results
    ├── healthcare_metrics.csv                         # Sensitivity/specificity metrics
    ├── feature_importance.csv                         # Permutation feature importance
    ├── decision_tree_importance.csv                   # Decision Tree feature importance
    ├── outlier_report.csv                             # IQR outlier summary
    └── *.png                                          # EDA, ROC, confusion matrix & comparison plots
```

---

## Disclaimer

This project is intended for academic and research purposes only and should not be used as a substitute for professional medical diagnosis or clinical decision-making.
