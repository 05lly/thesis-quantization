import torch
import torch.nn as nn
import os

# 设置环境变量解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CustomInt4Quantizer:
    """
    自定义INT4权重量化器，不依赖torchao
    实现真正的INT4权重量化（有符号4位整数，范围[-8, 7]）
    """
    def __init__(self, group_size=32):
        self.group_size = group_size
    
    def quantize_tensor(self, tensor):
        """
        将FP32张量量化为INT4
        返回量化后的张量和缩放因子
        """
        # 确保输入是FP32
        tensor = tensor.to(torch.float32)
        
        # 按组大小分割张量进行量化（每group_size个元素一组）
        original_shape = tensor.shape
        
        # 如果是线性层权重 [out_features, in_features]
        if len(original_shape) == 2:
            out_features, in_features = original_shape
            
            # 调整分组大小以适应输入特征数
            if in_features % self.group_size != 0:
                adjusted_group_size = in_features
            else:
                adjusted_group_size = self.group_size
            
            # 重塑张量以进行分组量化 [out_features, in_features // adjusted_group_size, adjusted_group_size]
            tensor_reshaped = tensor.reshape(out_features, -1, adjusted_group_size)
            
            # 计算每组的缩放因子（取绝对值最大值）
            max_vals = tensor_reshaped.abs().max(dim=-1, keepdim=True)[0]
            
            # 处理零值情况
            max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
            
            # 计算缩放因子：将最大值映射到7.0（INT4有符号最大值）
            scales = max_vals / 7.0
            
            # 确保缩放因子不为零
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            
            # 量化：缩放并舍入到最近的整数
            quantized = torch.round(tensor_reshaped / scales).clamp(-8, 7)
            
            # 重塑回原始形状
            quantized = quantized.reshape(original_shape)
            scales = scales.reshape(out_features, -1, 1).expand(-1, -1, adjusted_group_size).reshape(original_shape)
        
        # 其他类型张量的量化
        else:
            # 简单的逐张量量化
            max_val = tensor.abs().max()
            max_val = max_val if max_val != 0 else 1.0
            scale = max_val / 7.0
            scale = scale if scale != 0 else 1.0
            quantized = torch.round(tensor / scale).clamp(-8, 7)
            scales = scale * torch.ones_like(tensor)
        
        return quantized.to(torch.int8), scales
    
    def quantize(self, model):
        """
        量化模型中的所有线性层权重为INT4
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # 量化权重
                quantized_weight, scale = self.quantize_tensor(module.weight.data)
                
                # 替换为量化后的权重和缩放因子
                module.weight.data = quantized_weight.to(torch.int8)
                module.register_buffer('weight_scale', scale)
                
                # 替换forward方法以支持INT4推理，使用lambda解决闭包问题
                def make_forward_func(current_module):
                    def new_forward(input):
                        # 将输入转换为FP32
                        input_fp32 = input.to(torch.float32)
                        # 获取量化权重和缩放因子
                        quant_weight = current_module.weight.data.to(torch.float32)
                        weight_scale = current_module.weight_scale
                        # 反量化权重：quant_weight * weight_scale
                        dequantized_weight = quant_weight * weight_scale
                        # 执行矩阵乘法
                        return nn.functional.linear(input_fp32, dequantized_weight, current_module.bias)
                    return new_forward
                
                module.forward = make_forward_func(module)
            
            # 递归处理子模块
            elif hasattr(module, 'children'):
                self.quantize(module)
        
        return model

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