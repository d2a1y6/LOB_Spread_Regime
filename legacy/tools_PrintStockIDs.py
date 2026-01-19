import pandas as pd

def extract_hs300_stocks():
    """
    读取HS300成分股Excel文件，按交易所分类提取股票代码
    """
    # 读取Excel文件（确保文件在当前目录下）
    try:
        # 读取xls文件，header=0表示第一行是表头
        df = pd.read_excel('000300cons.xls', header=0)
    except FileNotFoundError:
        print("错误：未找到000300cons.xls文件，请确认文件在当前目录下！")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return
    
    # 清理列名（去除可能的空格）
    df.columns = [col.strip() for col in df.columns]
    
    # 定义列名（适配你的表头）
    code_col = '成份券代码Constituent Code'
    exchange_col = '交易所Exchange'
    
    # 检查必要列是否存在
    if code_col not in df.columns or exchange_col not in df.columns:
        print("错误：文件中缺少必要的列，请检查表头是否正确！")
        print(f"当前文件列名：{df.columns.tolist()}")
        return
    
    # 去重（避免重复的股票代码）
    df = df.drop_duplicates(subset=[code_col])
    
    # 按交易所筛选
    # 深市：深圳证券交易所
    sz_df = df[df[exchange_col] == '深圳证券交易所']
    sz_stocks = sz_df[code_col].astype(str).tolist()
    # 确保代码是字符串格式，补齐6位（防止数字格式导致前导0丢失）
    sz_stocks = [stock.zfill(6) for stock in sz_stocks]
    # 排序（可选，让列表更规整）
    sz_stocks.sort()
    
    # 沪市：上海证券交易所
    sh_df = df[df[exchange_col] == '上海证券交易所']
    sh_stocks = sh_df[code_col].astype(str).tolist()
    sh_stocks = [stock.zfill(6) for stock in sh_stocks]
    sh_stocks.sort()
    
    # 定义格式化输出列表的函数（每10个元素换行）
    def print_stock_list(var_name, stock_list):
        print(f"{var_name} = [", end="")
        for i, code in enumerate(stock_list):
            # 每10个元素换行，第一个元素不换行
            if i != 0 and i % 10 == 0:
                print("\n    ", end="")
            # 最后一个元素不加逗号，其余加逗号
            if i == len(stock_list) - 1:
                print(f"'{code}'", end="")
            else:
                print(f"'{code}', ", end="")
        print("]")
    
    # 按新格式输出股票列表
    print_stock_list("sz_targetStock", sz_stocks)
    print("\n=== 沪市targetStock列表 ===")
    print_stock_list("sh_targetStock", sh_stocks)
    
    # 统计并输出数量信息
    sz_count = len(sz_stocks)
    sh_count = len(sh_stocks)
    total_count = sz_count + sh_count
    
    print("\n=== 股票数量统计 ===")
    print(f"深市成分股数量：{sz_count} 只")
    print(f"沪市成分股数量：{sh_count} 只")
    print(f"沪深两市成分股总数：{total_count} 只")

if __name__ == "__main__":
    extract_hs300_stocks()