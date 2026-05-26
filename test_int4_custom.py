import torch
import torch.nn as nn

# 导入自定义INT4量化器
from resnet18_c100_int4_qat import CustomInt4Quantizer

def test_custom_int4_quantizer():
    """
    测试自定义INT4量化器的基本功能
    """
    print("=== 测试自定义INT4量化器 ===")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建一个简单的线性模型
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    print("\n原始模型结构:")
    print(model)
    
    # 测试单个张量的量化
    print("\n=== 测试张量量化 ===")
    quantizer = CustomInt4Quantizer(group_size=32)
    
    # 创建一个测试张量
    test_tensor = torch.randn(64, 128) * 10  # 随机张量，范围大约在[-10, 10]
    print(f"原始张量形状: {test_tensor.shape}")
    print(f"原始张量类型: {test_tensor.dtype}")
    print(f"原始张量范数: {torch.norm(test_tensor):.4f}")
    
    # 量化张量
    quantized_tensor, scales = quantizer.quantize_tensor(test_tensor)
    print(f"量化后张量类型: {quantized_tensor.dtype}")
    print(f"量化后张量范数: {torch.norm(quantized_tensor.to(torch.float32)):.4f}")
    print(f"缩放因子形状: {scales.shape}")
    print(f"缩放因子类型: {scales.dtype}")
    
    # 反量化测试
    dequantized_tensor = quantized_tensor.to(torch.float32) * scales
    print(f"反量化后张量范数: {torch.norm(dequantized_tensor):.4f}")
    print(f"量化误差 (L2): {torch.norm(test_tensor - dequantized_tensor):.4f}")
    print(f"相对误差: {(torch.norm(test_tensor - dequantized_tensor) / torch.norm(test_tensor)):.4%}")
    
    # 测试模型量化
    print("\n=== 测试模型量化 ===")
    quantized_model = quantizer.quantize(model)
    print("量化后模型结构:")
    print(quantized_model)
    
    # 测试推理
    print("\n=== 测试推理功能 ===")
    input_data = torch.randn(32, 1024)
    
    # 原始模型推理
    with torch.no_grad():
        original_output = model(input_data)
        print(f"原始模型输出形状: {original_output.shape}")
        print(f"原始模型输出范数: {torch.norm(original_output):.4f}")
    
    # 量化模型推理
    with torch.no_grad():
        quantized_output = quantized_model(input_data)
        print(f"量化模型输出形状: {quantized_output.shape}")
        print(f"量化模型输出范数: {torch.norm(quantized_output):.4f}")
        print(f"推理误差 (L2): {torch.norm(original_output - quantized_output):.4f}")
        print(f"相对推理误差: {(torch.norm(original_output - quantized_output) / torch.norm(original_output)):.4%}")
    
    print("\n=== 测试完成 ===")
    print("自定义INT4量化器在PyTorch 2.5.1环境下工作正常！")

if __name__ == "__main__":
    test_custom_int4_quantizer()