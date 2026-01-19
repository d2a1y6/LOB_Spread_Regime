"""
功能：命令行工具，用于快速检查 Parquet 或 CSV 文件的基本信息。
输入：文件路径 (通过命令行参数传入)。
输出：终端打印文件元数据（形状、内存、列类型）及前5行预览。
逻辑：识别文件后缀 -> 仅读取文件头部或元数据 -> 格式化输出信息。
"""

import pandas as pd
import argparse
import os

def inspect_file(file_path, n_rows=5):
    """
    执行文件检查并打印摘要。
    输入：
        file_path (str): 目标文件路径
        n_rows (int): 预览行数
    输出：无（直接打印到控制台）
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    print(f"正在检查: {os.path.basename(file_path)}")
    print("-" * 60)

    try:
        ext = os.path.splitext(file_path)[-1].lower()
        
        if ext == '.parquet':
            df = pd.read_parquet(file_path)
            # 对于极大的 Parquet，通常应只读 Schema，此处简化处理直接读入
            total_shape = df.shape
            df_preview = df.head(n_rows)
        elif ext in ['.csv', '.txt']:
            # 使用 iterator 避免读取大文件全量
            with pd.read_csv(file_path, iterator=True) as reader:
                df_preview = reader.get_chunk(n_rows)
                total_shape = ("Unknown", df_preview.shape[1])
        else:
            print(f"不支持的格式: {ext}")
            return

        # 输出信息
        print(f"形状: {total_shape}")
        print(f"列表: {', '.join(df_preview.columns)}")
        
        print("\n数据预览:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_preview)
        print("-" * 60)

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据文件快速预览工具")
    parser.add_argument('file_path', type=str, help="文件路径")
    parser.add_argument('--lines', type=int, default=5, help="预览行数")
    args = parser.parse_args()
    
    inspect_file(args.file_path, args.lines)