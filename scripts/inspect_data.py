# scripts/inspect_data.py
# 使用方法：终端输入 python scripts/inspect_data.py data/processed/ready_train.parquet

import pandas as pd
import argparse
import os
import sys

def inspect_file(file_path, n_rows=5):
    """
    读取文件的前 n_rows 行并打印元数据信息。
    支持 .parquet, .csv, .txt
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    file_ext = os.path.splitext(file_path)[-1].lower()
    df = None
    
    print(f"\n🔍 正在检查文件: {file_path}")
    print("=" * 60)

    try:
        # 1. 根据后缀读取数据 (只读少量行以提速)
        if file_ext == '.parquet':
            # Parquet 即使读取全量通常也很快，但为了保险，我们尝试只读 schema
            # 如果文件巨大，建议使用 pyarrow.parquet.ParquetFile
            try:
                df = pd.read_parquet(file_path) # Parquet列式存储，读取head并不需要全读，但这取决于引擎
                # 为了不让输出刷屏，我们在内存中只保留前 n 行
                total_shape = df.shape
                df = df.head(n_rows)
            except Exception as e:
                print(f"读取 Parquet 失败: {e}")
                return
                
        elif file_ext in ['.csv', '.txt']:
            # CSV 必须使用 nrows，否则会加载整个文件
            df_iter = pd.read_csv(file_path, iterator=True)
            df = df_iter.get_chunk(n_rows)
            # CSV获取总行数比较慢，这里暂时只显示列数
            total_shape = ("Unknown", df.shape[1])
            
        else:
            print(f"❌ 不支持的文件格式: {file_ext}")
            return

        # 2. 打印概览信息
        print(f"📊 数据形状 (Rows, Cols): {total_shape} (仅加载预览)")
        print(f"💾 内存占用 (预览): {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # 3. 打印列名和类型
        print("\n📋 列名清单 (Columns & Types):")
        print("-" * 60)
        # 格式化输出，每行显示 3 个列名，节省空间
        cols = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        for i in range(0, len(cols), 3):
            print(" | ".join(f"{c:<35}" for c in cols[i:i+3]))

        # 4. 打印数据示例
        print("\n👀 数据预览 (Head 5):")
        print("-" * 60)
        pd.set_option('display.max_columns', None)  # 强制显示所有列
        pd.set_option('display.width', 1000)        # 防止换行
        print(df.head(n_rows))
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ 读取过程中发生未知错误: {e}")

if __name__ == "__main__":
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="快速查看 Parquet/CSV 文件结构的工具")
    parser.add_argument('file_path', type=str, help="数据文件的路径")
    parser.add_argument('--lines', type=int, default=5, help="显示的行数 (默认: 5)")
    
    args = parser.parse_args()
    
    inspect_file(args.file_path, args.lines)