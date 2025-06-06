# 🌳 Custom Decision Tree Classifier from Scratch — Mushroom Classification

This project implements a full-fledged **Decision Tree Classifier from scratch in Python**, without using scikit-learn or any ML libraries. It is built to classify mushroom edibility using a combination of structured tabular datasets. The implementation includes **data preprocessing**, **hyperparameter tuning**, **post-training pruning**, **feature importance analysis**, and **evaluation**.

## 🚀 Key Highlights

- 🔧 **Algorithm from Scratch**: Implemented core decision tree logic, including entropy/gini computation, recursive tree building, prediction, and pruning.
- 🔍 **Hyperparameter Tuning**: Performed exhaustive grid search over max depth, min samples split, criterion, and pruning toggle.
- ✂️ **Post-training Pruning**: Supports optional pruning using a validation set to reduce overfitting and improve generalization.
- 📊 **Feature Importance**: Computed and ranked feature importances based on split gain.
- 📈 **Evaluation Metrics**: Includes accuracy, precision, recall, F1-score.
- 🧪 **Multiple Dataset Handling**: Loads, cleans, and merges primary and secondary mushroom datasets, with categorical encoding and handling of missing data.
- 🧠 **No ML Libraries Used**: Everything (model, metrics, preprocessing, grid search) is custom-built.

---

## 🛠 Technologies & Tools Used

| Category         | Tools & Techniques |
|------------------|--------------------|
| Programming      | Python, NumPy |
| ML Algorithm     | Custom Decision Tree |
| Data Processing  | CSV parsing, categorical encoding, missing value handling |
| Hyperparameter Tuning | Grid search (custom implementation) |
| Model Evaluation | Accuracy, Precision, Recall, F1 |
| Optimization     | Tree pruning (manual post-fit) |
| Experimentation  | Feature importance analysis |

---

## 🧪 Project Workflow

### 1. Data Loading & Preprocessing
- Loaded two separate datasets (primary and secondary mushroom datasets)
- Merged and sampled the data
- Encoded categorical features manually
- Detected feature types (numerical/categorical)

### 2. Decision Tree Implementation
- Built a custom decision tree class:
  - Handles both categorical and numerical features
  - Supports multiple split criteria (e.g., Gini, Entropy)
  - Recursive node creation and prediction traversal
  - Pruning based on validation loss
- `fit()`, `predict()` and pruning methods written from scratch

### 3. Grid Search & Hyperparameter Tuning
- Grid search over:
  - `max_depth`
  - `min_samples_split`
  - `criterion`
  - `prune`
- Selected best model based on validation F1-score

### 4. Model Evaluation
- Trained two versions of the final model:
  - With pruning
  - Without pruning
- Evaluated both on a held-out test set
- Reported metrics and top 5 important features

---
## 📊 Results

- Training Accuracy: ~X%
- Validation Accuracy: ~Y%
- Test Accuracy: ~Z%

