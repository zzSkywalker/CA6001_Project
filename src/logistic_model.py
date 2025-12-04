"""
Logistic Regression Model for HR Attrition Prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import (
    load_dataset, save_json, load_json, print_metrics, 
    plot_roc_curve, ensure_directories
)


def load_processed_data():
    """Load processed data."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')
    print(f"Loading processed data from: {data_path}")
    df = load_dataset(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def train_logistic_regression(df):
    """Train Logistic Regression model."""
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*80)
    
    # Separate features and target
    target = 'Attrition'
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target distribution:\n{y.value_counts().to_dict()}")
    
    # Split data
    print("\nSplitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining Logistic Regression with class_weight='balanced'...")
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = print_metrics(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    ensure_directories(model_dir)
    model_path = os.path.join(model_dir, 'logistic_regression.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save ROC curve
    roc_path = os.path.join(model_dir, 'logistic_regression_roc.png')
    plot_roc_curve(y_test, y_pred_proba, "Logistic Regression", roc_path)
    
    # Update model summary
    update_model_summary('logistic_regression', metrics)
    
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION TRAINING COMPLETED")
    print("="*80 + "\n")
    
    return model, metrics


def update_model_summary(model_name, metrics):
    """Update or create model summary JSON."""
    summary_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'model', 
        'model_summary.json'
    )
    
    # Load existing summary or create new
    if os.path.exists(summary_path):
        summary = load_json(summary_path)
    else:
        summary = {}
    
    # Update with new model metrics
    summary[model_name] = metrics
    
    # Save updated summary
    save_json(summary, summary_path)
    print(f"Model summary updated: {summary_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("HR ATTRITION PREDICTION - LOGISTIC REGRESSION")
    print("="*80)
    
    # Load processed data
    df = load_processed_data()
    
    # Train model
    model, metrics = train_logistic_regression(df)
    
    print("\n✅ Logistic Regression training completed successfully!\n")


if __name__ == "__main__":
    main()
