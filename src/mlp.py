"""
Deep Learning Model (MLP) for HR Attrition Prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from utils import (
    load_dataset, save_json, load_json, print_metrics,
    plot_roc_curve, ensure_directories
)

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_processed_data():
    """Load the processed data saved by eda_preprocess.py."""
    # 路径：项目根目录/data/processed_data.csv
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run eda_preprocess.py first.")

    print(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)

    target = 'Attrition'

    # 强制转换为 float32，防止 Keras 报错
    X = df.drop(columns=[target]).values.astype('float32')
    y = df[target].values.astype('float32')

    print(f"Data loaded. Shape: {df.shape}")
    return X, y

def build_mlp_model(input_dim):
    """
    Build a Multi-Layer Perceptron (MLP) model.
    Structure: Input -> Dense -> BatchNorm -> ReLU -> Dropout -> Output
    """
    model = Sequential([
        # 显式定义输入层
        Input(shape=(input_dim,)),

        # 第一层隐藏层
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # 第二层隐藏层
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # 第三层隐藏层
        Dense(16, activation='relu'),

        # 输出层：二分类使用 Sigmoid
        Dense(1, activation='sigmoid')
    ])

    # 编译模型
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

def update_model_summary(model_name, metrics):
    """Update or create model summary JSON."""
    # 路径：项目根目录/model/model_summary.json
    summary_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'model',
        'model_summary.json'
    )

    # Ensure directory exists
    ensure_directories(os.path.dirname(summary_path))

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

def train_model():
    print("\n" + "="*80)
    print("DEEP LEARNING MODEL TRAINING (MLP)")
    print("="*80)

    # 1. 准备数据
    X, y = load_processed_data()

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 再从训练集中划分出验证集 (用于监控模型训练过程)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    print(f"\nData Split:")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

    # 2. 处理类别不平衡
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass Weights computed: {class_weight_dict}")

    # 3. 构建模型
    model = build_mlp_model(input_dim=X_train.shape[1])
    model.summary()

    # 4. 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    # 5. 开始训练
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # 6. 模型评估
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # 预测
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算并打印指标
    metrics = print_metrics(y_test, y_pred, y_pred_proba, model_name="Deep Learning MLP")

    # 7. 保存结果
    # 统一保存到 model/ 目录下
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    ensure_directories(model_dir)

    # 保存 ROC 曲线
    roc_path = os.path.join(model_dir, 'mlp_roc.png')
    plot_roc_curve(y_test, y_pred_proba, "Deep Learning MLP", roc_path)

    # 保存模型文件 (.h5)
    model_path = os.path.join(model_dir, 'mlp_model.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # 更新 Model Summary JSON
    update_model_summary('mlp', metrics)

    print("\n" + "="*80)
    print("MLP TRAINING COMPLETED")
    print("="*80 + "\n")

    return history

if __name__ == "__main__":
    train_model()