import sys
import joblib
import pandas as pd
import numpy as npS
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Suppress sklearn/joblib version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Resolve project root (CCP directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def evaluate_model():
    # 1. Path Resolution
    data_path = Path("E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv")
    model_path = Path("E:/PROJECTS_FILE/CCP/src/models/churn_pipeline.joblib")

    print(f"Loading raw dataset from: {data_path}")
    print(f"Loading trained model binary from: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"❌ Dataset missing at '{data_path}'")
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model binary missing at '{model_path}'. Run training script first.")

    # 2. Load & Clean Dataset
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    target = "churn"
    id_col = "customerid"

    df = df.dropna(subset=[target])
    y = df[target].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    X = df.drop(columns=[target, id_col], errors='ignore')

    # 3. Load Model Pipeline
    pipeline = joblib.load(model_path)

    # 4. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Metrics & Predictions
    train_preds = pipeline.predict(X_train)
    train_acc = (train_preds == y_train).mean()

    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_acc = (test_preds == y_test).mean()

    roc_auc = roc_auc_score(y_test, test_probs)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)

    # 6. Display Metrics
    print("\n========== FINAL TRAINING RESULTS ==========")
    print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%)")

    print("\n========== FINAL TESTING RESULTS ==========")
    print(f"Testing Accuracy  : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn (0)", "Churn (1)"])
    disp.plot(cmap="coolwarm")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, test_preds, target_names=["No Churn", "Churn"], zero_division=0))
    import numpy as np


    # 7. Dynamic SHAP Explanations
    print("\nGenerating SHAP Explanations...")

    # 1. Extract Transformers & Features
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    num_cols = list(preprocessor.transformers_[0][2])
    cat_pipeline = preprocessor.transformers_[1][1]
    onehot_encoder = cat_pipeline.named_steps['encoder']
    cat_cols_input = list(preprocessor.transformers_[1][2])

    cat_feature_names = list(onehot_encoder.get_feature_names_out(cat_cols_input))
    all_feature_names = num_cols + cat_feature_names

    # 2. Sample & Preprocess Test Data
    shap_sample_raw = X_test.sample(n=min(200, len(X_test)), random_state=42)
    shap_sample_transformed = preprocessor.transform(shap_sample_raw)

    if hasattr(shap_sample_transformed, "toarray"):
        shap_sample_transformed = shap_sample_transformed.toarray()

    # 3. Initialize DataFrame BEFORE any try/except blocks
    shap_df = pd.DataFrame(
        np.asarray(shap_sample_transformed, dtype=np.float64),
        columns=all_feature_names
    )

    # 4. Compute SHAP Values (Handles XGBoost array base_score bug)
    try:
        # Pass underlying Booster model directly to bypass scikit-learn wrapper parsing
        booster = classifier.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(shap_df)
    except Exception:
        # Fallback to model-agnostic KernelExplainer if TreeExplainer fails
        explainer = shap.KernelExplainer(classifier.predict_proba, shap_df.iloc[:10])
        shap_values = explainer.shap_values(shap_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # 5. Plot & Save Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_df, show=False)
    plt.title("SHAP Feature Importance (Top Churn Drivers)")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()

    print("✓ SHAP summary plot saved as 'shap_summary_plot.png'")
if __name__ == "__main__":
    evaluate_model()








    '''# SHAP Explanations
    print("\nGenerating SHAP Explanations...")
    try:
        # Ensure pipeline is defined
        if 'pipeline' not in locals():
            pipeline = joblib.load(config["model_path"])
    
        # Retrieve feature names
        # Ensure pipeline is defined
        if 'pipeline' not in locals():
            pipeline = joblib.load(config["model_path"])
        preprocessor = pipeline.named_steps['preprocessor']
        cat_cols = config["categorical_columns"]
        num_cols = config["numerical_columns"]
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_feature_names

        # Generate SHAP values
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test[:500])  # Sampled for speed

        # SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:500], feature_names=all_feature_names, show=False)
        plt.title("SHAP Feature Importance (Top Churn Drivers)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"SHAP plot skipped due to: {e}")'''