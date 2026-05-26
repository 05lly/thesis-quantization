# 项目实施计划与要求解释

## 一、用户需求解释

### 1. 核心技术目标
- **ResNet18 INT8实时优化**：解决树莓派上帧率不足（<24 FPS）的问题，不通过缩小图片尺寸
- **INT4真实量化**：实现真正的4位量化并转换为.pt文件，在树莓派5上部署
- **目标检测模块**：在树莓派上集成目标检测功能

### 2. 创新点要求
- **量化策略创新**：主要体现在算法层面的量化优化，如：
  - 层敏感度分析引导的选择性量化
  - 渐进式量化感知训练
  - 架构特定的层融合优化
  - 自定义量化配置映射

### 3. 工作量展示
- **全流程跑通**：确保所有功能正常运行
- **实时实物检测**：在树莓派上实现≥24 FPS的目标检测
- **视频录制**：录制完整的演示视频，展示实时检测效果

### 4. 全流程展示
- **Windows端**：在Windows环境下进行初步实验和训练
- **Ubuntu系统**：在Ubuntu（树莓派系统）上进行部署和测试

## 二、实施计划（不改动原代码）

### 阶段1：ResNet18 INT8算法优化

#### 1.1 已完成的优化脚本
- `batch_optimize_all_networks.py`：批量优化所有三个网络架构
- `qat_resnet18_algorithmic_optimized.py`：ResNet18算法优化脚本

#### 1.2 核心优化技术
```python
# 1. 层敏感度分析
layer_sensitivity_analysis(model, test_loader, device)

# 2. 渐进式QAT训练（三阶段）
# 阶段1：宽松量化 → 阶段2：中度量化 → 阶段3：严格量化
progressive_qat_training(model, train_loader, test_loader, epochs, optimizer, criterion, device)

# 3. 增强层融合
model.fuse_model(is_qat=True)

# 4. 自定义量化配置
get_optimized_qconfig_mapping(sensitivity_scores)
```

### 阶段2：INT4真实量化实现

#### 2.1 现有INT4脚本
- `int4_quantization_torchao.py`：使用torchao实现真实INT4量化
- `resnet18_int4_real.py`：ResNet18的INT4真实量化脚本

#### 2.2 核心技术点
```python
# 使用torchao实现INT4权重仅量化
from torchao.quantization import quant_api
int4_model = quant_api.int4_weight_only_quant(model)

# 或使用混合精度量化（INT8激活 + INT4权重）
int4_model = quant_api.int8_dynamic_activation_int4_weight_quant(model)
```

### 阶段3：目标检测模块集成

#### 3.1 现有检测脚本
- `object_detection.py`：轻量级YOLO风格目标检测模型

#### 3.2 集成方法
- 将检测模型与量化后的分类模型结合
- 在树莓派上实现端到端的目标检测流程

### 阶段4：部署与测试

#### 4.1 树莓派部署
- 使用`test_pi5_optimized.py`测试INT8优化模型
- 创建`test_pi5_int4.py`测试INT4量化模型

#### 4.2 性能测试指标
- **帧率**：确保≥24 FPS
- **准确率**：保持合理的检测/分类精度
- **延迟**：测量端到端推理时间

## 三、执行步骤（不改动原代码）

### 步骤1：运行INT8算法优化
```bash
# 批量优化所有网络（ResNet18, MobileNetV2, VGG16）
python batch_optimize_all_networks.py

# 或仅优化ResNet18
python qat_resnet18_algorithmic_optimized.py
```

### 步骤2：实现INT4真实量化
```bash
# 运行ResNet18的INT4真实量化
python resnet18_int4_real.py

# 生成INT4部署文件
python int4_quantization_torchao.py --network resnet18
```

### 步骤3：树莓派部署测试
```bash
# 在树莓派上测试INT8优化模型
python test_pi5_optimized.py --model models/resnet18_c10_int8_deploy_optimized.pt

# 在树莓派上测试INT4量化模型
python test_pi5_int4.py --model models/resnet18_c10_int4_deploy.pt
```

### 步骤4：集成目标检测
```bash
# 运行目标检测演示
python object_detection.py --model models/resnet18_c10_int8_deploy_optimized.pt
```

## 四、创新点与工作量展示

### 创新点体现
1. **量化策略创新**：采用层敏感度分析引导的选择性量化
2. **训练方法创新**：渐进式QAT训练减少精度损失
3. **部署优化创新**：ARM架构特定的层融合和内存布局优化

### 工作量展示
1. **代码量**：新增5+个优化脚本，实现完整的量化优化流程
2. **实验结果**：提供详细的性能对比（FP32 vs INT8 vs INT4）
3. **视频演示**：录制树莓派实时检测视频，展示≥24 FPS的效果

## 五、全流程展示说明

### Windows端（开发环境）
- 数据准备与预处理
- 模型训练（FP32基准模型）
- 量化优化实验（QAT, INT4）
- 模型导出（.pt文件）

### Ubuntu系统（树莓派）
- 环境配置（PyTorch, 量化引擎）
- 模型部署（加载.pt文件）
- 性能测试（帧率、准确率）
- 实时检测演示

## 六、注意事项

1. **不改动原代码**：所有优化通过新增脚本实现，保持原实验框架不变
2. **设备兼容性**：确保所有脚本在Windows和树莓派Ubuntu上均可运行
3. **性能优先**：在保证精度的前提下，优先优化树莓派上的运行速度
4. **可复现性**：保持实验参数一致，确保结果可复现

## 七、预期成果

1. **优化后的INT8 ResNet18**：在树莓派上达到≥24 FPS
2. **真实INT4量化模型**：成功部署并测试
3. **目标检测功能**：集成并实现实时检测
4. **完整文档**：包含实施过程、实验结果和创新点说明
5. **演示视频**：展示全流程和实时检测效果

---

**实施建议**：先运行INT8算法优化脚本，验证在树莓派上的帧率提升，再进行INT4量化和目标检测集成。