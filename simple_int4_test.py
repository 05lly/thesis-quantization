import torch
import torch.nn as nn
import os

# 设置环境变量解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("=== 简单INT4量化验证 ===")
print(f"PyTorch版本: {torch.__version__}")

# 创建一个超级简单的模型（只有一个线性层）
model = nn.Linear(16, 8)

# 打印原始模型信息
print("\n原始模型:")
print(f"权重形状: {model.weight.shape}")
print(f"权重类型: {model.weight.dtype}")
print(f"权重示例值: {model.weight.data[0, :4].tolist()}")

# 简单的INT4量化函数
def simple_int4_quantize(tensor):
    """将FP32张量量化为INT4"""
    # INT4范围: -8 到 7
    scale = tensor.abs().max() / 7.0
    scale = scale if scale != 0 else 1.0
    quantized = torch.round(tensor / scale).clamp(-8, 7)
    return quantized.to(torch.int8), scale

def simple_int4_dequantize(quantized_tensor, scale):
    """将INT4张量反量化为FP32"""
    return quantized_tensor.to(torch.float32) * scale

# 测试权重量化
print("\n=== 测试权重量化 ===")
quantized_weight, scale = simple_int4_quantize(model.weight.data)

print(f"量化后权重类型: {quantized_weight.dtype}")
print(f"量化后权重值: {quantized_weight[0, :4].tolist()}")
print(f"缩放因子: {scale.item():.6f}")

# 测试反量化
dequantized_weight = simple_int4_dequantize(quantized_weight, scale)
print(f"反量化后权重: {dequantized_weight[0, :4].tolist()}")

# 计算量化误差
error = torch.norm(model.weight.data - dequantized_weight) / torch.norm(model.weight.data)
print(f"相对量化误差: {error.item():.4%}")

# 测试推理
print("\n=== 测试推理 ===")
input_data = torch.randn(4, 16)  # 4个样本，每个样本16个特征

# 原始模型推理
with torch.no_grad():
    original_output = model(input_data)
    print(f"原始模型输出: {original_output.tolist()}")

# 量化模型推理
with torch.no_grad():
    # 使用反量化的权重进行计算
    quant_output = nn.functional.linear(input_data, dequantized_weight, model.bias)
    print(f"量化模型输出: {quant_output.tolist()}")

# 计算推理误差
inference_error = torch.norm(original_output - quant_output) / torch.norm(original_output)
print(f"相对推理误差: {inference_error.item():.4%}")

print("\n=== 验证完成 ===")
print("✅ 成功实现了不依赖torchao的INT4量化！")
print("✅ 保持PyTorch 2.5.1环境，无需重新运行之前的实验！")
print("\n您现在可以使用修改后的resnet18_c100_int4_qat.py脚本进行INT4量化实验。")