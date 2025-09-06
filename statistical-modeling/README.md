# Personality Traits and Drug Consumption: A Statistical Learning Project

## Project Overview
This project investigates the relationship between human personality—using the Five-Factor Model (FFM)—and consumption of legal and illegal substances. It combines **unsupervised and supervised statistical learning** to uncover patterns in behavior and predict drug use based on personality traits.

**Key Focus:**
- Identify personality profiles via clustering  
- Predict drug consumption from personality and demographics  

---

##  Skills Demonstrated
- **Statistical Analysis & Inference:** Hypothesis testing, multivariate analysis  
- **Unsupervised Learning:** PCA, K-Means clustering  
- **Supervised Learning:** Logistic Regression, LDA, Decision Trees  
- **Model Evaluation:** Accuracy, ROC-AUC metrics  
- **Data Wrangling & Feature Engineering:** Handling demographics, binning, correlation analysis  
- **Interpretability:** Feature importance and trait impact analysis  

---

## Project Structure

### Part 1: Personality Clustering (Unsupervised Learning)
- **Objective:** Cluster 874,366 individuals based on FFM personality scores  
- **Methods:** PCA for dimensionality reduction, K-Means clustering  
- **Key Insight:** Identified distinct personality profiles, revealing trait patterns across populations  

### Part 2: Drug Use Prediction (Supervised Learning)
- **Objective:** Classify users vs. non-users for various substances  
- **Data Source:** [UCI Drug Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)  
- **Methods:** Logistic Regression, LDA, Decision Trees  
- **Key Findings:**  
  - **Sensation Seeking** is the strongest predictor for most drugs  
  - **Openness** and **Low Conscientiousness** are significant for Cannabis, LSD, Ecstasy, Mushrooms  
  - Drug correlations suggest overlapping user profiles (e.g., Mushrooms & LSD: 0.67)  

---

## Key Results

| Substance | AUC | Substance | AUC |
|-----------|-----|-----------|-----|
| Alcohol   | 0.60 | Ecstasy  | 0.78 |
| Benzos    | 0.73 | LSD      | 0.77 |
| Nicotine  | 0.69 | Mushrooms| 0.78 |
| Coke      | 0.70 | Cannabis | 0.83 |

> Note: High accuracy for Alcohol is misleading due to class imbalance; AUC provides a more reliable measure.

**Top Predictors:**
- **Cannabis, LSD, Ecstasy, Mushrooms:** Sensation Seeking, Openness, Low Conscientiousness  
- **Cocaine:** Sensation Seeking, Low Agreeableness  
- **Nicotine:** Sensation Seeking, Low Conscientiousness  

---

