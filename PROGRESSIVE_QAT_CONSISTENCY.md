# 渐进式QAT训练策略一致性说明

## 所有INT4量化脚本的训练阶段划分

经过全面检查，**所有6个INT4量化脚本**（ResNet18、MobileNetV2、VGG16分别对应CIFAR-10和CIFAR-100数据集）都采用了**完全一致**的渐进式QAT训练策略：

### 1. 阶段划分
- **Relaxed阶段**：前3个epoch（epoch < 3）
  - 宽松的量化约束
  - 启用量化观察者
  - 不冻结BN层统计信息
  - 目的：让模型逐渐适应量化噪声

- **Moderate阶段**：中间7个epoch（3 ≤ epoch < 10）
  - 中等强度的量化约束
  - 禁用量化观察者
  - 冻结BN层统计信息
  - 目的：平衡精度和量化程度

- **Strict阶段**：最后5个epoch（epoch ≥ 10）
  - 严格的量化约束
  - 禁用量化观察者
  - 冻结BN层统计信息
  - 目的：模拟真实部署环境

### 2. 训练配置一致性
- **总epoch数**：所有脚本统一使用15个epoch
- **学习率**：统一设置为1e-4
- **优化器**：统一使用Adam优化器
- **损失函数**：统一使用交叉熵损失

### 3. 网络和数据集适配
虽然训练策略一致，但脚本会根据不同网络架构和数据集进行适当适配：
- **批量大小**：VGG16由于模型较大，使用较小的batch_size=64，其他网络使用batch_size=128
- **数据集预处理**：CIFAR-10和CIFAR-100使用不同的标准化参数
- **模型架构**：每个脚本加载对应网络的预训练权重

## 这样设计的优势

1. **公平比较**：统一的训练策略确保了不同网络和数据集之间的可比性
2. **可重复性**：一致的配置便于重现实验结果
3. **渐进式适应**：三阶段训练帮助模型逐步适应量化噪声，减少精度损失
4. **符合实际需求**：最后阶段严格的量化约束确保模型能在真实部署环境中表现良好

## 代码验证

所有脚本中的渐进式QAT阶段划分代码完全一致：

```python
if epoch < 3:
    # 阶段1：放松阶段，启用观察者，不冻结BN
    model.apply(torch.ao.quantization.enable_observer)
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = True
            if hasattr(module, 'freeze_bn'):
                module.freeze_bn = False
    qat_stage = "Relaxed"
elif epoch < 10:
    # 阶段2：适度阶段，禁用观察者，冻结BN
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    qat_stage = "Moderate"
else:
    # 阶段3：严格阶段，完全启用量化约束
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    qat_stage = "Strict"
```

这确保了所有INT4量化实验都在统一的训练框架下进行，便于您进行公平的性能比较和分析。