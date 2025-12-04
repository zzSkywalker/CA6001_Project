"""
XGBoost Model for HR Attrition Prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


def calculate_scale_pos_weight(y):
    """Calculate scale_pos_weight for imbalanced dataset."""
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    scale_pos_weight = negative_count / positive_count
    return scale_pos_weight


def train_xgboost(df):
    """Train XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
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
    
    # Calculate scale_pos_weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    # Hyperparameter tuning with RandomizedSearchCV
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")
    print("="*80)
    
    # Define parameter search space
    param_grid = {
        'n_estimators': [300, 500, 800],
        'learning_rate': [0.02, 0.03, 0.05],
        'max_depth': [2, 3, 4],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.6, 0.8],
        'min_child_weight': [1, 2, 3]
    }
    
    # Base model with fixed parameters
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    # RandomizedSearchCV with 5-fold CV, optimizing for ROC-AUC
    print("\nStarting RandomizedSearchCV with 5-fold cross-validation...")
    print("Optimizing for ROC-AUC score...")
    print(f"Parameter search space: {len(param_grid)} parameters")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nHyperparameter tuning completed!")
    print(f"Best ROC-AUC score (CV): {random_search.best_score_:.4f}")
    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Retrain model with best parameters
    print("\n" + "="*80)
    print("RETRAINING XGBOOST WITH BEST PARAMETERS")
    print("="*80)
    print("\nTraining XGBoost with tuned hyperparameters...")
    
    model = xgb.XGBClassifier(
        **random_search.best_params_,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = print_metrics(y_test, y_pred, y_pred_proba, "XGBoost (Tuned)")
    
    # Add best parameters to metrics (convert numpy types to native Python types for JSON serialization)
    best_params_dict = {}
    for k, v in random_search.best_params_.items():
        if isinstance(v, (np.integer, int)):
            best_params_dict[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            best_params_dict[k] = float(v)
        else:
            best_params_dict[k] = v
    metrics['best_params'] = best_params_dict
    metrics['best_cv_score'] = float(random_search.best_score_)
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    ensure_directories(model_dir)
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save ROC curve
    roc_path = os.path.join(model_dir, 'xgboost_roc.png')
    plot_roc_curve(y_test, y_pred_proba, "XGBoost (Tuned)", roc_path)
    
    # Update model summary with "xgboost_tuned"
    update_model_summary('xgboost_tuned', metrics)
    
    print("\n" + "="*80)
    print("XGBOOST TRAINING COMPLETED (WITH HYPERPARAMETER TUNING)")
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
    print("HR ATTRITION PREDICTION - XGBOOST")
    print("="*80)
    
    # Load processed data
    df = load_processed_data()
    
    # Train model
    model, metrics = train_xgboost(df)
    
    print("\n✅ XGBoost training completed successfully!\n")


if __name__ == "__main__":
    main()
