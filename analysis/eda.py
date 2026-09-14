import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

def run_eda(dataset_path="E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv"):
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(df)

    # Exploring the dataset
    print("=== Dataset Overview ===")
    print("Shape:", df.shape)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nTotal missing values:", df.isna().sum().sum())
    print("\nColumns with missing values:")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print("\nData types:")
    print(df.dtypes.value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe(include="all").T.head(10))
    print("Columns:", df.columns.tolist())
    print("Number of columns:", df.shape[1])
    print("\nTarget Class Distribution:")
    print(df['Churn'].value_counts(normalize=True))

    # Ensure the directory exists
    os.makedirs("analysis_phase", exist_ok=True)

    # Plotting Customer Churn Distribution
    plt.figure(figsize=(6, 4))
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    sns.countplot(data=df, x='Churn', hue='Churn', palette='coolwarm', legend=False)
    plt.savefig("analysis_phase/churn_distribution.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_eda(dataset_path="E:/PROJECTS_FILE/CCP/data/raw/cell2celltrain.csv")
