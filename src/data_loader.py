import pandas as pd
import os

def load_data():
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上退一级到根目录，再进入 data 文件夹
    data_path = os.path.join(current_dir, '../data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # 读取数据
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"成功读取数据！包含 {df.shape[0]} 行, {df.shape[1]} 列")
        return df
    else:
        print(f"错误：找不到文件，请检查路径: {data_path}")
        return None


if __name__ == "__main__":
    load_data()