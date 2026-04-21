# Titanic Survival Prediction
## Logistic Regression Classification Model

## Overview
As part of my journey learning AI Engineering and Machine Learning,
I built a Logistic Regression model to predict passenger survival
on the Titanic. This is a binary classification problem —
predicting whether a passenger Survived (1) or Died (0).

## Dataset
- **Source:** Kaggle — Titanic: Machine Learning from Disaster
- **Training Size:** 891 rows, 12 columns
- **Final Size After Cleaning:** 876 rows
- **Target Variable:** Survived (0 = Died, 1 = Survived)
- **Class Distribution:** 62% Died, 38% Survived

## Project Pipeline

| Step | Description |
|------|-------------|
| Data Loading | Load Titanic training dataset |
| EDA | Explore structure, missing values, correlations |
| Data Cleaning | Handle missing Age, Embarked, drop Cabin, remove £0 fares |
| Feature Engineering | Create Family Size, apply log transform on Fare |
| Encoding | Label encode Sex, One Hot encode Embarked |
| Train/Test Split | 80% training, 20% testing |
| Feature Scaling | RobustScaler on Age and Family Size |
| Model Training | Logistic Regression |
| Evaluation | Confusion Matrix, Accuracy Score, Classification Report |

## Data Cleaning Summary

| Column | Issue | Solution |
|--------|-------|---------|
| Age | 177 missing (20%) | Filled with median grouped by Pclass and Sex |
| Embarked | 2 missing | Filled with mode grouped by Pclass |
| Cabin | 687 missing (77%) | Dropped — too much missing data |
| Fare | 15 rows with £0 | Dropped — likely data errors |

## Feature Engineering

| Feature | Description |
|---------|-------------|
| Family Size | SibSp + Parch + 1 — captures family group effect on survival |
| Fare (log) | Log transformation applied to reduce right skew |

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| F1-Score (Died) | 0.84 |
| F1-Score (Survived) | 0.72 |
| Macro F1 | 0.78 |

The model achieves **80% accuracy** — significantly better than
the 62% baseline of always predicting death.

## Key Findings
- **Sex** was the strongest predictor — women survived at much higher rates
- **Pclass** strongly affected survival — 1st class had better lifeboat access
- **Fare** correlated with survival — higher fare = higher survival rate
- **Family Size** mattered — small families survived better than solo travellers or large families
- **Port of Embarkation** was meaningful — Cherbourg passengers had 55% survival vs 34% for Southampton


