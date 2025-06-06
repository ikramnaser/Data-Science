# 🌲 Tree-Based Classifiers for Multi-Source Data Prediction

This repository implements and evaluates decision tree-based classification models trained on structured datasets. The focus is on building robust classifiers using **Scikit-learn**, exploring **hyperparameter tuning**, and applying **data preprocessing** techniques to integrate **multiple datasets** efficiently.

---

## 🚀 Project Summary

This project demonstrates an end-to-end ML pipeline involving:

- Reading and cleaning multiple structured datasets
- Encoding categorical labels manually for full control
- Training multiple **tree-based classifiers**: `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`
- Performing train/test split and evaluation using accuracy, confusion matrix, and classification report
- Exploring model robustness with partial dataset augmentation
- Experimenting with **parameter tuning** for `max_depth`, `n_estimators`, etc.

---

## 🧰 Tools & Technologies

| Category              | Tools Used                                                                 |
|-----------------------|----------------------------------------------------------------------------|
| **Languages**         | Python                                                                     |
| **Libraries**         | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`                |
| **Modeling**          | `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier` |
| **Evaluation**        | Accuracy Score, Confusion Matrix, Classification Report                    |

---

## 📁 Project Structure

---

## 📊 Dataset & Features

The primary and secondary datasets consist of various structured features and a `class` label. Preprocessing steps ensure compatibility between datasets and include:

- Missing value handling (replaced with "N/A")
- Label encoding for target classes
- Subsampling secondary dataset to control training size

---

## 📈 Models Trained

The notebook supports multiple tree-based classifiers:

- `DecisionTreeClassifier` (with custom depth)
- `RandomForestClassifier` (tunable number of estimators)
- `GradientBoostingClassifier` (focusing on ensemble boosting)

Each model is evaluated using:

- Accuracy Score
- Confusion Matrix Visualization (via Seaborn heatmaps)
- Detailed Classification Report

---

## 🧠 Skills Demonstrated

✅ **Machine Learning Engineering**
- Model selection and performance evaluation  
- Tree ensemble techniques and overfitting control  
- Manual label encoding and dataset integration logic

✅ **Data Science & Analytics**
- Exploratory Data Analysis (EDA)
- Performance metrics beyond accuracy  
- Sampling and class distribution awareness

✅ **Software Engineering Practices**
- Modular coding via reusable functions
- Clear separation of training/evaluation logic
- Readable, reproducible Jupyter workflow

✅ **MLOps & Containerization** *(related deployment experience)*:
- Docker-based containerization for deploying ML APIs  
- Experience deploying models to **GCP** via **Flask** + **App Engine**

---

## 🧪 Example Output

- 📉 Classification Accuracy: ~85–95% (depending on dataset)
- 📊 Confusion matrix plots and precision-recall metrics
- ✅ Demonstrates how tree models handle categorical-rich data

---


