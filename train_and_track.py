import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set up paths matching your ChurnIQ structure
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cell2celltrain.csv"

# 1. Initialize MLflow Experiment
mlflow.set_experiment("ChurnIQ_Customer_Churn")

# 2. Load & Preprocess Data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip().str.lower()

# Basic cleaning for churn baseline
df['churn'] = (df['churn'] == 'Yes').astype(int) if df['churn'].dtype == object else df['churn']
X = df.drop(columns=['churn', 'customerid'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Define Model Configurations
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss")
}
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 4. Train, Track & Log Runs
for model_name, model in candidate_models.items():
    with mlflow.start_run(run_name=model_name):
        if model_name == "XGBoost":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Log Hyperparameters
            mlflow.log_params(model.get_params())
            
            # Log Performance Metrics
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.log_metric("f1", f1_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            # Save Trained Model Artifact to MLflow
            mlflow.xgboost.log_model(model, name="model")
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Log Hyperparameters
            mlflow.log_params(model.get_params())
            
            # Log Performance Metrics
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.log_metric("f1", f1_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            # Save Trained Model Artifact to MLflow
            mlflow.sklearn.log_model(model, name="model")
        print(f"Recorded run for: {model_name}")