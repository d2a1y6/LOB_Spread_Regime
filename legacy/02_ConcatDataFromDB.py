"""
程序名称：多文件行情数据合并与格式转换 (CSV to Parquet Merger)

1. 程序概述：
   本程序负责将从数据库（如 DolphinDB）导出的分散 CSV 行情文件（沪市/深市）
   合并为一个统一的 Parquet 文件。它是数据预处理流水线的第一步，
   旨在解决海量 CSV 文件的读取效率低、占用空间大及排序混乱问题。

2. 输入数据：
   - 来源文件：当前目录下匹配 'sh_*.csv' (沪市) 和 'sz_*.csv' (深市) 的所有文件。
   - 数据格式：原始 CSV 格式，包含 Stock, Time 及微观结构特征列。

3. 核心逻辑流程：
   1) 文件扫描：自动检索并汇总所有待处理的 CSV 文件路径。
   2) 批量读取：逐个读取文件，强制解析 'Time' 列为 datetime 对象（关键步骤）。
   3) 数据合并：将分散的 DataFrame 拼接为单一大表。
   4) 全局排序：【核心】优先按 'Stock' 排序，再按 'Time' 排序。
      * 目的：保证同一股票的数据在物理存储上连续，大幅优化后续 Rolling/Shift 操作及压缩率。
   5) 格式转换：输出为 Snappy 压缩的 Parquet 文件，提升 I/O 速度 10 倍以上。

4. 输出内容：
   - 文件：'market_data.parquet' (合并、排序、压缩后的最终数据文件)。
"""

import pandas as pd
import glob
import os
import time

def process_and_merge_data():
    # ================= 1. 配置与初始化 =================
    input_folder = '.'
    output_file = 'market_data.parquet'
    
    # 匹配沪市(sh)和深市(sz)的分片数据文件
    file_patterns = ['sh_*.csv', 'sz_*.csv']

    all_files = []
    for pattern in file_patterns:
        full_pattern = os.path.join(input_folder, pattern)
        all_files.extend(glob.glob(full_pattern))
    
    if not all_files:
        print(f"错误: 在 {input_folder} 下未找到任何匹配 sh/sz 的 CSV 文件。")
        return

    print(f">>> 发现 {len(all_files)} 个数据文件，准备处理...")
    for f in all_files:
        print(f"    - {os.path.basename(f)}")
    
    start_time = time.time()

    # ================= 2. 读取与合并 =================
    print("\n>>> 正在读取并合并 CSV 文件 (这可能需要几分钟)...")
    
    try:
        df_list = []
        for f in all_files:
            # parse_dates: 读取时直接解析时间，避免后续转换开销
            # low_memory=False: 防止大文件读取时的混合类型警告
            temp_df = pd.read_csv(f, parse_dates=['Time'], low_memory=False)
            df_list.append(temp_df)
            print(f"    已加载: {os.path.basename(f)} | 行数: {len(temp_df)}")
            
        full_data = pd.concat(df_list, ignore_index=True)
        print(f">>> 合并完成。总数据量: {len(full_data)} 行")
        
        # 主动释放内存
        del df_list
        
    except Exception as e:
        print(f"读取或合并过程中出错: {e}")
        return

    # ================= 3. 全局排序 (核心优化) =================
    # 策略：Stock (主键1) -> Time (主键2)
    # 理由：Parquet 列式存储对连续数据压缩率极高，且后续特征工程依赖于股票内部的时间连续性
    print("\n>>> 正在进行全局排序 (按 Stock 和 Time)...")
    full_data.sort_values(by=['Stock', 'Time'], inplace=True)

    # ================= 4. 存储为 Parquet =================
    print(f"\n>>> 正在写入 Parquet 文件: {output_file} ...")
    try:
        # engine='pyarrow': 业界标准引擎
        # compression='snappy': 速度与压缩率的最佳平衡点，适合频繁读取
        full_data.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        
        end_time = time.time()
        print(f"====== 任务成功 ======")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"文件已保存至: {os.path.abspath(output_file)}")
        
    except ImportError:
        print("错误: 缺少 pyarrow 库。请运行: pip install pyarrow")
    except Exception as e:
        print(f"写入文件失败: {e}")

if __name__ == "__main__":
    process_and_merge_data()