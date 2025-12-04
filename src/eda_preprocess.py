"""
Exploratory Data Analysis and Preprocessing for HR Attrition Prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from utils import ensure_directories, load_dataset

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load raw data from CSV."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw_data.csv')
    print(f"Loading data from: {data_path}")
    df = load_dataset(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def perform_eda(df):
    """Perform Exploratory Data Analysis."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Basic information
    print("\n1. DATASET INFORMATION")
    print("-" * 80)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print(f"Rows: {df.shape[0]}")
    
    # Data types
    print("\n2. DATA TYPES")
    print("-" * 80)
    print(df.dtypes)
    
    # Missing values
    print("\n3. MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    # Target variable distribution
    print("\n4. TARGET VARIABLE DISTRIBUTION (Attrition)")
    print("-" * 80)
    attrition_counts = df['Attrition'].value_counts()
    attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
    print(attrition_counts)
    print(f"\nPercentage:\n{attrition_pct}")
    
    # Numerical statistics
    print("\n5. NUMERICAL FEATURES SUMMARY")
    print("-" * 80)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numerical_cols].describe())
    
    # Categorical features
    print("\n6. CATEGORICAL FEATURES")
    print("-" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Save EDA plots
    save_eda_plots(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETED")
    print("="*80 + "\n")


def save_eda_plots(df):
    """Generate and save EDA visualization plots."""
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eda_plots')
    ensure_directories(plot_dir)
    
    # 1. Attrition distribution
    plt.figure(figsize=(8, 6))
    df['Attrition'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
    plt.title('Attrition Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Attrition', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'attrition_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(plot_dir, 'attrition_distribution.png')}")
    
    # 2. Correlation heatmap (numerical features)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        plt.figure(figsize=(14, 12))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(plot_dir, 'correlation_heatmap.png')}")
    
    # 3. Attrition by key categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    key_cats = ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'BusinessTravel']
    key_cats = [col for col in key_cats if col in categorical_cols]
    
    for col in key_cats[:4]:  # Limit to 4 plots
        plt.figure(figsize=(10, 6))
        pd.crosstab(df[col], df['Attrition']).plot(kind='bar', stacked=False, 
                                                    color=['#2ecc71', '#e74c3c'])
        plt.title(f'Attrition by {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Attrition', labels=['No', 'Yes'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'attrition_by_{col.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(plot_dir, f'attrition_by_{col.lower()}.png')}")


def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # 1. Convert Attrition Yes/No to 1/0
    print("\n1. Converting Attrition: Yes/No -> 1/0")
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    print(f"   Attrition distribution after conversion:")
    print(f"   {df_processed['Attrition'].value_counts().to_dict()}")
    
    # 2. Remove constant columns (if any)
    print("\n2. Removing constant columns")
    constant_cols = [col for col in df_processed.columns if df_processed[col].nunique() <= 1]
    if constant_cols:
        print(f"   Removing: {constant_cols}")
        df_processed = df_processed.drop(columns=constant_cols)
    else:
        print("   No constant columns found")
    
    # 3. Remove ID columns (EmployeeNumber, EmployeeCount, etc.)
    print("\n3. Removing ID columns")
    id_cols = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    id_cols = [col for col in id_cols if col in df_processed.columns]
    if id_cols:
        print(f"   Removing: {id_cols}")
        df_processed = df_processed.drop(columns=id_cols)
    
    # 4. Separate features and target
    target = 'Attrition'
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    # 5. Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n4. Feature types:")
    print(f"   Categorical: {len(categorical_cols)} features")
    print(f"   Numerical: {len(numerical_cols)} features")
    
    # 6. One-Hot Encoding for categorical features
    print("\n5. Applying One-Hot Encoding to categorical features")
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, prefix_sep='_')
        print(f"   Encoded features: {X_encoded.shape[1]} (from {X.shape[1]})")
    else:
        X_encoded = X.copy()
        print("   No categorical features to encode")
    
    # 7. StandardScaler for numerical features
    print("\n6. Applying StandardScaler to numerical features")
    scaler = StandardScaler()
    if numerical_cols:
        # Get the numerical column names after encoding
        numerical_cols_encoded = [col for col in numerical_cols if col in X_encoded.columns]
        X_encoded[numerical_cols_encoded] = scaler.fit_transform(X_encoded[numerical_cols_encoded])
        print(f"   Scaled {len(numerical_cols_encoded)} numerical features")
    
    # 8. Combine with target
    df_final = X_encoded.copy()
    df_final[target] = y
    
    print(f"\n7. Final processed dataset shape: {df_final.shape}")
    print(f"   Features: {df_final.shape[1] - 1}")
    print(f"   Target: 1 (Attrition)")
    print(f"   Samples: {df_final.shape[0]}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED")
    print("="*80 + "\n")
    
    return df_final


def save_processed_data(df_processed):
    """Save processed data to CSV."""
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("HR ATTRITION PREDICTION - EDA & PREPROCESSING")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Save processed data
    save_processed_data(df_processed)
    
    print("\n✅ EDA and Preprocessing completed successfully!\n")


if __name__ == "__main__":
    main()
