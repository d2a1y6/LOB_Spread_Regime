import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# 0. 全局配置
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

MODEL_PATH = 'models/LightGBM.pkl'
OUTPUT_DIR = 'analysis_results/09_TreeDiagram'
TREE_INDEX = 0        
MAX_DEPTH = 4

FEATURE_COLS = [
    'Accum_Vol_Diff', 'VolumeMax', 'VolumeAll', 'Immediacy', 'Depth_Change', 
    'LobImbalance', 'DeepLobImbalance', 'Relative_Spread', 'Micro_Mid_Spread', 
    'PastReturn', 'Lambda', 'Volatility', 'AutoCov'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 1. 颜色配置 (高反差)
# ==============================================================================
def get_node_style(feature_name, is_leaf=False, leaf_value=0):
    """返回 (背景色, 边框色)"""
    if is_leaf:
        # 涨红跌绿
        return ('#ffcccc', '#cc0000') if leaf_value > 0 else ('#ccffcc', '#006600')
    
    # 特征颜色映射
    color_map = {
        'Accum_Vol_Diff':   '#FFB3BA', 
        'Relative_Spread':  '#BAE1FF',
        'Immediacy':        '#FFFFBA', 
        'LobImbalance':     '#BAFFC9',
        'Micro_Mid_Spread': '#E6B3FF', 
        'VolumeMax':        '#FFDFBA',
        'VolumeAll':        '#FFCCF9',
        'Depth_Change':     '#BAFFEF', 
        'DeepLobImbalance': '#E6FF80',
        'PastReturn':       '#D4C4FB', 
        'Lambda':           '#F2E2C4', 
        'Volatility':       '#D3D3D3', 
        'AutoCov':          '#FFC2C2', 
    }
    bg = color_map.get(feature_name, '#FFFFFF')
    return bg, '#333333' # 背景色，黑色边框

# ==============================================================================
# 2. 核心算法：布局计算
# ==============================================================================
class TreeLayout:
    def __init__(self, tree_structure, max_depth):
        self.nodes = {} 
        self.leaf_counter = 0
        self.max_depth_limit = max_depth
        
        # 计算坐标
        self._traverse_assign_x(tree_structure, current_depth=0)
        
    def _traverse_assign_x(self, node_info, current_depth):
        node_id = id(node_info)
        
        # 超过深度或遇到叶子
        is_leaf_node = 'leaf_value' in node_info
        is_max_depth = current_depth >= self.max_depth_limit
        
        if is_leaf_node or is_max_depth:
            self.nodes[node_id] = {
                'x': self.leaf_counter,
                'y': -current_depth, 
                'raw': node_info,
                'type': 'leaf' if is_leaf_node else 'truncated'
            }
            self.leaf_counter += 1
            return self.nodes[node_id]['x']
        
        # 内部节点
        left_x = self._traverse_assign_x(node_info['left_child'], current_depth + 1)
        right_x = self._traverse_assign_x(node_info['right_child'], current_depth + 1)
        
        current_x = (left_x + right_x) / 2
        
        self.nodes[node_id] = {
            'x': current_x,
            'y': -current_depth,
            'raw': node_info,
            'type': 'split',
            'children_ids': [id(node_info['left_child']), id(node_info['right_child'])]
        }
        return current_x

# ==============================================================================
# 3. 绘图引擎 (Matplotlib)
# ==============================================================================
def draw_tree_matplotlib(model, tree_index=0, max_depth=3, filename='tree_mpl_final'):
    print(f"\n[Step 1] 解析模型结构 (Tree {tree_index})...")
    try:
        model_json = model.dump_model()
        tree_info = model_json['tree_info'][tree_index]['tree_structure']
    except AttributeError:
        model_json = model.booster_.dump_model()
        tree_info = model_json['tree_info'][tree_index]['tree_structure']

    # 计算布局
    print("[Step 2] 计算节点坐标...")
    layout = TreeLayout(tree_info, max_depth)
    
    # [关键] 动态调整画布大小
    # 6层树可能非常宽，增加每个叶子的预留宽度
    width = max(12, layout.leaf_counter * 1.8) 
    height = max(10, max_depth * 2.5)             
    
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off() 
    
    ax.set_xlim(-0.5, layout.leaf_counter - 0.5)
    ax.set_ylim(-max_depth - 0.5, 0.5)
    
    print("[Step 3] 绘制节点与连线...")
    
    for n_id, data in layout.nodes.items():
        x, y = data['x'], data['y']
        node = data['raw']
        ntype = data['type']
        
        if ntype == 'split':
            feat_idx = node['split_feature']
            feat_name = FEATURE_COLS[feat_idx] if feat_idx < len(FEATURE_COLS) else f"Column_{feat_idx}"
            threshold = node['threshold']
            gain = node.get('split_gain', 0)
            
            text_str = f"{feat_name}\n< {threshold:.4f}\nGain: {gain:.0f}"
            bg_color, border_color = get_node_style(feat_name, is_leaf=False)
            
            # 画连线
            children = data['children_ids']
            lx, ly = layout.nodes[children[0]]['x'], layout.nodes[children[0]]['y']
            ax.plot([x, lx], [y, ly], color='blue', linestyle='-', linewidth=1, zorder=1)
            
            rx, ry = layout.nodes[children[1]]['x'], layout.nodes[children[1]]['y']
            ax.plot([x, rx], [y, ry], color='red', linestyle='-', linewidth=1, zorder=1)
            
            # [修改 2] 增大 Yes/No 字号，并设为粗体
            ax.text((x+lx)/2, (y+ly)/2, "Yes", color='blue', 
                    fontsize=14, fontweight='bold', # 字号 14 + 粗体
                    ha='center', va='center', zorder=5, 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0))
            
            ax.text((x+rx)/2, (y+ry)/2, "No", color='red', 
                    fontsize=14, fontweight='bold', # 字号 14 + 粗体
                    ha='center', va='center', zorder=5, 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0))

        else: # Leaf or Truncated
            val = node.get('leaf_value', 0)
            count = node.get('leaf_count', 0)
            if ntype == 'truncated':
                text_str = "...\n(Pruned)"
                bg_color, border_color = '#eeeeee', '#999999'
            else:
                text_str = f"Val: {val:.3f}\nN: {count}"
                bg_color, border_color = get_node_style('leaf', is_leaf=True, leaf_value=val)
        
        # 绘制方框
        ax.text(x, y, text_str, 
                ha='center', va='center', 
                fontsize=11, fontweight='normal', fontfamily='Verdana',
                bbox=dict(boxstyle="round,pad=0.4", fc=bg_color, ec=border_color, lw=1.5),
                zorder=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[完成] 树结构图已保存: {save_path}")

# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == '__main__':
    print(f"正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("错误：找不到模型文件。")
        exit()
        
    model = joblib.load(MODEL_PATH)
    
    # 运行绘图
    draw_tree_matplotlib(model, tree_index=0, max_depth=MAX_DEPTH, filename='tree_0_matplotlib')