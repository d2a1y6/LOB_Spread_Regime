import pandas as pd
import os

def read_and_show_parquet(file_path, n_rows=5):
    """
    读取parquet文件并展示前n行数据
    
    参数:
    file_path: parquet文件的路径
    n_rows: 要展示的行数，默认5行
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"⚠️  文件 {file_path} 不存在，请检查路径是否正确！")
        return None
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        print(f"\n===== {file_path} 的前 {n_rows} 行数据 =====")
        # 展示前n行
        print(df.head(n_rows))
        # 额外展示数据的基本信息
        print(f"\n📊 {file_path} 的数据基本信息：")
        print(f"数据形状（行×列）：{df.shape}")
        print(f"列名：{list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ 读取 {file_path} 时出错：{str(e)}")
        return None

# 主执行逻辑
if __name__ == "__main__":
    train_file = "ready_train.parquet"
    test_file = "ready_test.parquet"
    
    # 读取并展示训练集和测试集的前5行（可修改n_rows参数调整行数）
    train_df = read_and_show_parquet(train_file, n_rows=5)
    test_df = read_and_show_parquet(test_file, n_rows=5)