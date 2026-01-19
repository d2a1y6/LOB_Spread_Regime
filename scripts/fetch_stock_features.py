"""
功能：获取沪深300成分股的截面异质性特征（价格、波动、换手、PE、市值）。
输入：AkShare API 接口数据。
输出：data/processed/stock_heterogeneity_features.csv
逻辑：
    1. 获取 HS300 成分股列表。
    2. 获取全市场实时快照（提取 PE 和 流通市值）。
    3. 循环获取指定时间段的日线数据，计算均价、平均振幅、平均换手。
    4. 合并数据并导出 CSV。
"""

import akshare as ak
import pandas as pd
import numpy as np
import os

# 配置区域
DATA_START_DATE = "20250630"
DATA_END_DATE   = "20250725"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', "stock_heterogeneity_features.csv")

def generate_full_features():
    """
    主流程：拉取数据、计算统计量并保存。
    输入：无（依赖全局配置和网络接口）
    输出：无（生成 CSV 文件）
    """
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    print("正在获取成分股列表及市场快照...")
    try:
        stock_list = ak.index_stock_cons(symbol="000300")['品种代码'].tolist()
        spot_df = ak.stock_zh_a_spot_em()
        pe_map = dict(zip(spot_df['代码'], spot_df['市盈率-动态']))
        float_mv_map = dict(zip(spot_df['代码'], spot_df['流通市值']))
    except Exception as e:
        print(f"初始化数据获取失败: {e}")
        return

    print(f"开始处理 {len(stock_list)} 只股票的历史数据 ({DATA_START_DATE}-{DATA_END_DATE})...")
    features = []
    
    for i, code in enumerate(stock_list):
        if i % 50 == 0 and i > 0: 
            print(f"进度: {i}/{len(stock_list)}")
            
        try:
            df_hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=DATA_START_DATE, end_date=DATA_END_DATE, adjust="qfq")
            if df_hist.empty: continue
            
            features.append({
                'Stock': code,
                'Daily_Price': df_hist['收盘'].mean(),
                'Daily_Amplitude': df_hist['振幅'].mean(),
                'Daily_Turnover': df_hist['换手率'].mean(),
                'Daily_PE': pe_map.get(code, np.nan),
                'Daily_FloatMV': float_mv_map.get(code, np.nan)
            })
        except:
            continue # 跳过获取失败的个股

    if not features: 
        print("错误：未能生成有效特征。")
        return

    df_res = pd.DataFrame(features).dropna()
    df_res['Stock'] = df_res['Stock'].astype(str).str.zfill(6)
    df_res.to_csv(OUTPUT_FILE, index=False)
    
    print(f"处理完成。已保存 {len(df_res)} 只股票特征至: {os.path.basename(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_full_features()