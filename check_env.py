import onnxruntime as ort
import torch

# 1. 检测 ONNX Runtime 的可用提供者
print("ONNX Runtime 可用的提供者:")
print(ort.get_available_providers())

print("-" * 30)

# 2. 检测 PyTorch 的 CUDA 可用性
print(f"PyTorch 是否能使用 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"当前 CUDA 设备名称: {torch.cuda.get_device_name(0)}")
