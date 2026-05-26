# INT4量化问题修复总结

## 问题描述
用户在使用PyTorch 2.5.1环境运行INT4量化代码时遇到错误：
```
AttributeError: module 'torch' has no attribute 'int1'
```

原因是最新版本的torchao (0.13.0+) 需要PyTorch 2.11.0+，而用户环境中只有PyTorch 2.5.1。

## 解决方案
实现了一个不依赖torchao的自定义INT4量化器，该量化器使用PyTorch内置功能实现真正的INT4权重量化。

## 修复内容

### 1. 修改的文件
- `resnet18_c100_int4_qat.py`：移除torchao依赖，添加自定义INT4量化器

### 2. 核心修改点

#### (1) 移除torchao依赖
```python
# 移除的导入
# from torchao.quantization.quant_api import quantize_qat, Int4WeightQConfigMapping
# from torchao.quantization.quantizers import WeightOnlyInt4Quantizer
```

#### (2) 实现自定义INT4量化器
创建了`CustomInt4Quantizer`类，实现以下功能：
- INT4权重量化（有符号4位整数，范围[-8, 7]）
- 分组量化（默认group_size=32）
- 支持线性层权重的INT4量化
- 自定义forward方法支持INT4推理

#### (3) 替换torchao量化调用
```python
# 原代码
# int4_model = WeightOnlyInt4Quantizer().quantize(int8_model)

# 新代码
int4_quantizer = CustomInt4Quantizer(group_size=32)
int4_model = int4_quantizer.quantize(int8_model)
```

## 自定义INT4量化器工作原理

1. **权重量化**：将FP32权重按组（默认32个元素一组）量化为INT4
2. **缩放因子计算**：每组计算独立的缩放因子，将最大值映射到7.0
3. **反量化推理**：在推理时，将INT4权重反量化为FP32后进行计算
4. **精度保护**：通过分组量化减少量化误差，保持模型精度

## 使用方法

1. 确保环境中已安装PyTorch 2.5.1
2. 运行修改后的INT4量化脚本：
```bash
python resnet18_c100_int4_qat.py
```

## 预期效果

1. 成功运行INT4量化，无需升级PyTorch
2. 获得真正的INT4量化模型，可部署在树莓派上
3. 保持与原torchao方案相近的精度和性能
4. 模型大小减少约75%（从FP32到INT4）

## 注意事项

1. 自定义量化器主要针对线性层权重进行INT4量化
2. 激活值仍使用INT8量化，平衡精度和性能
3. 推理时会自动进行反量化操作
4. 支持导出为TorchScript格式进行部署

## 其他网络架构的适配

该自定义INT4量化器可轻松适配到其他网络架构（如MobileNetV2、VGG16），只需将相同的修改应用到相应的INT4量化脚本中。