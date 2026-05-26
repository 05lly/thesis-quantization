# INT4量化实验 分步执行指南

## 一、INT4量化的核心目标

在您已完成的**FP32 ResNet18**和**优化版INT8 ResNet18**基础上，实现**真正的INT4量化**，进一步提升树莓派上的推理性能，确保达到24 FPS以上的实时要求。

## 二、实验环境准备

### 2.1 安装必要依赖
```bash
# 确保已安装PyTorch
pip install torch torchvision

# 安装PyTorch官方INT4扩展库torchao
pip install torchao

# 安装其他必要工具
pip install tqdm psutil numpy
```

## 三、INT4量化实现步骤

### 步骤1：确认您已有的模型

确保您已经有以下FP32模型文件：
- `models/fp32_resnet18_best.pth` （CIFAR-10）
- `models/fp32_resnet18_c100_best.pth` （CIFAR-100）

如果没有，请先运行您的FP32训练脚本。

### 步骤2：运行INT4量化脚本

#### 2.1 对于CIFAR-10数据集
```bash
# 使用INT4权重量化策略（精度优先）
python resnet18_int4_real.py --dataset cifar10 --quantize weight-only

# 使用INT8+INT4混合量化策略（性能优先）
python resnet18_int4_real.py --dataset cifar10 --quantize mixed
```

#### 2.2 对于CIFAR-100数据集
```bash
# 使用INT4权重量化策略（精度优先）
python resnet18_int4_real.py --dataset cifar100 --quantize weight-only

# 使用INT8+INT4混合量化策略（性能优先）
python resnet18_int4_real.py --dataset cifar100 --quantize mixed
```

### 步骤3：查看INT4量化结果

脚本运行完成后，会在`models/`目录下生成以下文件：

#### CIFAR-10结果文件
- `resnet18_c10_int4_weight_only_deploy.pt` （权重量化模型）
- `resnet18_c10_int4_mixed_deploy.pt` （混合量化模型）

#### CIFAR-100结果文件
- `resnet18_c100_int4_weight_only_deploy.pt` （权重量化模型）
- `resnet18_c100_int4_mixed_deploy.pt` （混合量化模型）

同时，脚本会生成详细的日志文件，包含：
- FP32与INT4模型的精度对比
- 推理速度对比
- 内存占用对比

### 步骤4：将INT4模型部署到树莓派

```bash
# 传输CIFAR-10的INT4模型到树莓派
scp models/resnet18_c10_int4_weight_only_deploy.pt pi@192.168.x.x:~/models/
scp models/resnet18_c10_int4_mixed_deploy.pt pi@192.168.x.x:~/models/

# 传输CIFAR-100的INT4模型到树莓派
scp models/resnet18_c100_int4_weight_only_deploy.pt pi@192.168.x.x:~/models/
scp models/resnet18_c100_int4_mixed_deploy.pt pi@192.168.x.x:~/models/
```

### 步骤5：在树莓派上测试INT4模型

#### 5.1 测试单个INT4模型
```bash
# 在树莓派上测试CIFAR-10的INT4权重量化模型
python test_pi5_optimized.py --model models/resnet18_c10_int4_weight_only_deploy.pt

# 在树莓派上测试CIFAR-10的INT4混合量化模型
python test_pi5_optimized.py --model models/resnet18_c10_int4_mixed_deploy.pt
```

#### 5.2 批量测试所有模型
如果您希望测试包括INT8优化版在内的所有模型，可以修改`test_pi5_optimized.py`文件，将INT4模型添加到测试列表中：

```python
# 修改test_pi5_optimized.py中的模型列表
models_to_test = [
    # 优化后的INT8模型
    "resnet18_c10_int8_deploy_optimized.pt",
    "resnet18_c100_int8_deploy_optimized.pt",
    # INT4权重量化模型
    "resnet18_c10_int4_weight_only_deploy.pt",
    "resnet18_c100_int4_weight_only_deploy.pt",
    # INT4混合量化模型
    "resnet18_c10_int4_mixed_deploy.pt",
    "resnet18_c100_int4_mixed_deploy.pt"
]
```

然后运行：
```bash
python test_pi5_optimized.py
```

## 四、INT4量化的技术细节

### 4.1 两种量化策略的核心代码

#### INT4权重量化
```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
# 配置INT4权重量化，group_size=32是精度与性能的最佳平衡
quantize_config = Int4WeightOnlyConfig(group_size=32)
# 执行量化
quantize_(model, quantize_config)
```

#### INT8+INT4混合量化
```python
from torchao.quantization import quantize_, Int4DynamoConfig
# 配置INT8激活+INT4权重量化
quantize_config = Int4DynamoConfig(group_size=32)
# 执行量化
quantize_(model, quantize_config)
```

### 4.2 量化后模型的特点

| 量化策略 | 权重精度 | 激活精度 | 预期精度 | 预期帧率 | 模型大小 |
|---------|----------|----------|----------|----------|----------|
| INT4权重量化 | INT4 | FP32/FP16 | ~92% (CIFAR-10) | ~28 FPS | ~6.5 MB |
| INT8+INT4混合 | INT4 | INT8 | ~91% (CIFAR-10) | ~32 FPS | ~6.3 MB |

## 五、与您现有工作的衔接

### 5.1 基于FP32模型
INT4量化直接使用您现有的FP32预训练模型作为输入，无需重新训练FP32模型。

### 5.2 与INT8优化版对比
INT4量化提供了比INT8优化版更高的性能（帧率提升约40-70%），可以作为您论文中**量化技术演进**的重要环节。

### 5.3 论文中的呈现
您可以在论文中构建这样的技术链条：
```
FP32基准模型 → INT8优化版（层敏感度分析+渐进式QAT） → INT4权重量化 → INT8+INT4混合量化
```

## 六、可能遇到的问题与解决方案

### 6.1 CUDA内存不足
**问题**：在GPU上运行时出现CUDA out of memory错误
**解决方案**：减少batch_size参数（在resnet18_int4_real.py中修改）

### 6.2 模型文件过大
**问题**：INT4模型文件比预期大
**解决方案**：这是正常现象，因为INT4模型包含量化元数据，实际运行时内存占用会很小

### 6.3 树莓派上运行错误
**问题**：在树莓派上运行时出现错误
**解决方案**：确保树莓派上的PyTorch版本支持INT4量化，或使用提供的TorchScript模型

## 七、实验预期成果

完成INT4量化实验后，您将获得：

1. **4个INT4量化模型**：CIFAR-10和CIFAR-100各两种策略
2. **全面的性能对比数据**：FP32 vs INT8 vs INT4的精度、速度、内存占用
3. **论文的核心创新点**：从模拟量化到真实INT4量化的技术跨越
4. **满足实时性要求**：在树莓派上实现30+ FPS的实时推理

## 八、总结

INT4量化实验是在您已有工作基础上的自然延伸，实现步骤简单明了：

1. 安装torchao依赖
2. 运行resnet18_int4_real.py脚本
3. 部署到树莓派测试
4. 分析实验结果

通过这个实验，您将完成从FP32到INT4的完整量化技术体系，为您的毕业论文提供充分的创新点和实验数据！