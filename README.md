# Customer Churn Prediction & Retention Intelligence

## Business Problem

StreamFlow wants to identify customers at high risk
of churning within the next 30 days.

## Objective

Build an ML system capable of:

• Predicting churn probability
• Ranking customers by risk
• Explaining individual predictions
• Supporting retention decisions
• Monitoring model performance

## Results
Model: XGBOOSTING

ROC-AUC
0.774

PR-AUC
0.613

Precision
0.818

Recall
0.187

F1 Score
0.304

Brier Score
0.169

Recall @ Top 10%
26.3%

## Architecture
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|  +-----------------------------------------------------------------------------+  |
|  |                         Streamlit Web Application                           |  |
|  |                                (app1_4.py)                                  |  |
|  |                                                                             |  |
|  |   [🏠 Overview]      [👤 Customer Risk]    [🔍 What Drives Churn]   [💼 ROI]   |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              OPTIMIZATION & CACHING                               |
|  +-----------------------------------------------------------------------------+  |
|  |                 Streamlit Caching Engine (@st.cache_data / resource)       |  |
|  |                                                                             |  |
|  |  * Vectorized Name Hashing       * Fast Risk Categorization (LOW -> CRITICAL) |  |
|  |  * Model Object Persistence       * Asynchronous Lazy SHAP Calculation     |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+-----------------------------------------+ +---------------------------------------+
|              DATA PIPELINE              | |            MODEL ENGINE               |
|  +-----------------------------------+  | |  +---------------------------------+  |
|  |        Cell2Cell Dataset          |  | |  |       scikit-learn Pipeline     |  |
|  |    (data/raw/cell2celltrain.csv)  |  | |  |    (models/churn_pipeline.joblib)|  |
|  +-----------------------------------+  | |  +---------------------------------+  |
|  | Data Cleaning & Column Normalizer |  | |  | Preprocessor (Imputer/Encoder)  |  |
|  | Vector ID & Risk Mapping Engine   |  | |  | Classifier (Tree-Based Model)   |  |
|  +-----------------------------------+  | |  +---------------------------------+  |
+-----------------------------------------+ +---------------------------------------+
                                                       |
                                                       v
                                            +-----------------------+
                                            |   SHAP EXPLAINABILITY |
                                            |   * TreeExplainer     |
                                            |   * Local Insights    |
                                            |   * Global Feature    |
                                            |     Importance        |
                                            +-----------------------+

## Features

• Data validation
• Feature engineering
• Model training
• Hyperparameter tuning
• SHAP explainability
• FastAPI
• Streamlit
• MLflow
• Docker
• Monitoring

## Demo

[screenshot]

## How to run

docker compose up
