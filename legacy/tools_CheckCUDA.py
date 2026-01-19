import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

# 1. 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 2. 检查 PyTorch 编译时的 CUDA 版本
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# 3. 尝试获取显卡信息
if cuda_available:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
else:
    print(">> 错误原因推断：")
    if torch.version.cuda is None:
        print("   [!] 你安装的是 CPU 版本的 PyTorch。")
    else:
        print("   [!] PyTorch 安装了 CUDA 版，但无法连接显卡。可能是驱动过旧。")