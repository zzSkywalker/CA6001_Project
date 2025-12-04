"""
Hyperparameter Tuning for MLP using Keras Tuner.
(Read-Only Mode: This script outputs best params but saves NOTHING to disk)
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# 保持随机性一致
tf.random.set_seed(42)
np.random.seed(42)

def load_data_for_tuning():
    """Load data (Read-Only)."""
    # 既然不引用utils了，我们直接用相对路径读取
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    target = 'Attrition'

    # 强制类型转换
    X = df.drop(columns=[target]).values.astype('float32')
    y = df[target].values.astype('float32')
    return X, y

def build_hypermodel(hp):
    """
    Builds a model with hyperparameters to tune.
    """
    model = Sequential()

    # 注意：Keras Tuner 会自动处理 Input shape，或者我们可以让第一层 Dense 自动推断

    # 1. 调优：隐藏层的数量 (1 到 3 层)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            # 2. 调优：每一层的神经元数量
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))

        # 3. 调优：是否使用 BatchNormalization
        if hp.Boolean('batch_norm'):
            model.add(BatchNormalization())

        # 4. 调优：Dropout 率
        model.add(Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 5. 调优：学习率
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def run_tuning():
    print("\n" + "=" * 80)
    print("STARTING HYPERPARAMETER TUNING (READ-ONLY MODE)")
    print("=" * 80)

    # 1. 加载数据
    X, y = load_data_for_tuning()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 2. 初始化 Tuner
    # 注意：directory='kt_dir' 是为了存放搜索过程中的临时文件，不会覆盖你的模型文件
    tuner = kt.Hyperband(
        build_hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=50,
        factor=3,
        directory='kt_dir',
        project_name='hr_attrition_tuning'
    )

    # 3. 开始搜索
    print("\nSearching for best hyperparameters...")
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[stop_early],
        class_weight=class_weight_dict,
        verbose=1
    )

    # 4. 获取并打印最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n" + "=" * 80)
    print("🏆 BEST HYPERPARAMETERS FOUND 🏆")
    print("-" * 80)
    print(f"Number of layers: {best_hps.get('num_layers')}")
    print(f"Learning rate:    {best_hps.get('learning_rate')}")
    print(f"Batch Norm:       {best_hps.get('batch_norm')}")
    print(f"Dropout Rate:     {best_hps.get('dropout')}")
    print("-" * 40)

    for i in range(best_hps.get('num_layers')):
        print(f"Layer {i} Units:      {best_hps.get(f'units_{i}')}")

    print("=" * 80 + "\n")
    print("✅ Tuning finished! You can now manually copy these parameters to mlp.py.")

if __name__ == "__main__":
    run_tuning()