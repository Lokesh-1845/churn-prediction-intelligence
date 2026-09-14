import sys
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# Add the project root directory to the Python path
sys.path.append("E:/PROJECTS_FILE/CCP")

def analyze_saved_model(model_path="E:/PROJECTS_FILE/CCP/models/churn_pipeline.joblib", test_data_path="data/processed/cleaned_churn_data.csv"):
    try:
        # Load the trained pipeline
        pipeline = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at: {model_path}. Please ensure the model exists.")
        return
    except ModuleNotFoundError as e:
        print(f"Error: {e}. Ensure the 'src' module is in the Python path.")
        return

    # Load the test dataset
    df = pd.read_csv(test_data_path)
    print(f"Test data loaded successfully from: {test_data_path}")

    # Prepare features and target
    X = df.drop(columns=['churn', 'customerid'], errors='ignore')
    y = df['churn'].map({'Yes': 1, 'No': 0})

    # Make predictions
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    # Print evaluation metrics
    print("=== Evaluation Summary ===")
    print(classification_report(y, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y, probs):.4f}")

if __name__ == "__main__":
    analyze_saved_model()