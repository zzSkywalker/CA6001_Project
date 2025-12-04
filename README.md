# HR Employee Attrition Prediction

A machine learning project to predict employee attrition using various classification algorithms.

## 📋 Project Description

This project analyzes employee data to predict attrition (employee turnover) using three different machine learning models:
- **Logistic Regression**: Baseline model with balanced class weights
- **Random Forest**: Ensemble method with balanced class weights
- **XGBoost**: Gradient boosting with scale_pos_weight for handling class imbalance

The project includes comprehensive exploratory data analysis (EDA), data preprocessing, model training, and evaluation metrics.

## 📁 Directory Structure

```
HR/
├── data/
│   ├── raw_data.csv              # Original dataset (unchanged)
│   ├── processed_data.csv        # Preprocessed dataset
│   └── eda_plots/                # EDA visualization plots
│
├── model/
│   ├── logistic_regression.pkl   # Trained Logistic Regression model
│   ├── random_forest.pkl         # Trained Random Forest model
│   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── model_summary.json        # Model performance metrics
│   ├── logistic_regression_roc.png
│   ├── random_forest_roc.png
│   └── xgboost_roc.png
│
├── src/
│   ├── eda_preprocess.py         # EDA and data preprocessing
│   ├── logistic_model.py         # Logistic Regression training
│   ├── random_forest_model.py    # Random Forest training
│   ├── xgboost_model.py          # XGBoost training
│   └── utils.py                  # Utility functions
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd HR
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Step 1: Exploratory Data Analysis and Preprocessing

Run the EDA and preprocessing script to analyze the data and create the processed dataset:

```bash
python src/eda_preprocess.py
```

**What it does:**
- Loads `data/raw_data.csv`
- Performs comprehensive EDA (prints statistics and generates plots)
- Converts Attrition from Yes/No to 1/0
- Applies One-Hot Encoding to categorical features
- Applies StandardScaler to numerical features
- Saves processed data to `data/processed_data.csv`
- Saves EDA plots to `data/eda_plots/`

**Output:**
- `data/processed_data.csv` - Preprocessed dataset ready for modeling
- `data/eda_plots/` - Visualization plots (attrition distribution, correlation heatmap, etc.)

### Step 2: Train Models

Train each model individually:

#### Logistic Regression
```bash
python src/logistic_model.py
```

#### Random Forest
```bash
python src/random_forest_model.py
```

#### XGBoost
```bash
python src/xgboost_model.py
```

**What each script does:**
- Loads `data/processed_data.csv`
- Splits data into train/test sets (80/20)
- Trains the model with appropriate class imbalance handling
- Evaluates performance on test set
- Saves the trained model as `.pkl` file
- Generates ROC curve plot
- Updates `model/model_summary.json` with performance metrics

**Output for each model:**
- `model/{model_name}.pkl` - Trained model file
- `model/{model_name}_roc.png` - ROC curve visualization
- `model/model_summary.json` - Updated with model metrics

### Running All Models Sequentially

You can run all models in sequence:

```bash
python src/logistic_model.py
python src/random_forest_model.py
python src/xgboost_model.py
```

## 📈 Model Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (measures model's ability to distinguish between classes)

All metrics are displayed in the console and saved to `model/model_summary.json`.

## 📄 Output Interpretation

### Console Output

Each script provides:
- Dataset information (shape, feature counts)
- Target variable distribution
- Training progress
- Performance metrics on test set
- File save locations

### model_summary.json

Contains performance metrics for all trained models in JSON format:
```json
{
    "logistic_regression": {
        "accuracy": 0.85,
        "precision": 0.72,
        "recall": 0.68,
        "f1_score": 0.70,
        "roc_auc": 0.82
    },
    "random_forest": { ... },
    "xgboost": { ... }
}
```

### ROC Curves

ROC curve plots show:
- Model's true positive rate vs false positive rate
- Area Under Curve (AUC) score
- Comparison with random classifier (diagonal line)

Higher AUC indicates better model performance.

## 🔧 Technical Details

### Data Preprocessing

1. **Target Conversion**: Attrition (Yes/No) → (1/0)
2. **Feature Encoding**: One-Hot Encoding for categorical variables
3. **Feature Scaling**: StandardScaler for numerical features
4. **ID Removal**: Removes non-predictive columns (EmployeeNumber, EmployeeCount, etc.)

### Model Configurations

- **Logistic Regression**: `class_weight='balanced'`, max_iter=1000
- **Random Forest**: n_estimators=100, max_depth=10, `class_weight='balanced'`
- **XGBoost**: n_estimators=100, max_depth=6, `scale_pos_weight` calculated from data

### Class Imbalance Handling

All models use techniques to handle the imbalanced nature of attrition data:
- Logistic Regression & Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` calculated as (negative_count / positive_count)

## 📝 Notes

- The `raw_data.csv` file should remain unchanged
- All scripts are designed to be run independently
- Models are saved using pickle format
- EDA plots are automatically generated during preprocessing
- The project uses a fixed random seed (42) for reproducibility

## 🤝 Contributing

This is a standalone project. For modifications:
1. Ensure `raw_data.csv` remains unchanged
2. Follow the existing code structure
3. Update `model_summary.json` when adding new models

## 📧 Support

For issues or questions, refer to the code comments in each script file.

---

**Last Updated**: 2024
