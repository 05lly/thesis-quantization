# 文件夹清理建议

## 一、当前文件状态

### 1. 核心脚本文件（保留）
✅ **INT8 QAT脚本**
- `qat_resnet18.py` - 基础版ResNet18 INT8 QAT (CIFAR-10)
- `qat_resnet18_c100.py` - 基础版ResNet18 INT8 QAT (CIFAR-100)
- `qat_resnet18_optimized.py` - 优化版ResNet18 INT8 QAT (CIFAR-10)
- `qat_resnet18_c100_optimized.py` - 优化版ResNet18 INT8 QAT (CIFAR-100)

✅ **INT4 QAT脚本（新生成）**
- `resnet18_c10_int4_qat.py` - ResNet18 INT4 QAT (CIFAR-10)
- `resnet18_c100_int4_qat.py` - ResNet18 INT4 QAT (CIFAR-100)
- `mobilenetv2_c10_int4_qat.py` - MobileNetV2 INT4 QAT (CIFAR-10)
- `mobilenetv2_c100_int4_qat.py` - MobileNetV2 INT4 QAT (CIFAR-100)
- `vgg16_c10_int4_qat.py` - VGG16 INT4 QAT (CIFAR-10)
- `vgg16_c100_int4_qat.py` - VGG16 INT4 QAT (CIFAR-100)

✅ **测试脚本**
- `test_pi5_optimized.py` - 树莓派测试脚本

### 2. 文档文件（可选择性删除）

#### 建议删除的文档（临时或重复内容）
❌ **INT4相关临时指南**
- `INT4_OPTION2_ONLY_GUIDE.md`
- `INT4_STEP_BY_STEP_GUIDE.md`
- `INT4_EXPERIMENT_GUIDE.md`
- `INT4_IMPLEMENTATION_DETAILS.md`
- `INT4_INNOVATION_EXPLANATION.md`
- `INT4_FULL_QUANTIZATION_EXPLANATION.md`
- `INT4_SIMPLE_EXPLANATION.md`
- `INT4_IMPLEMENTATION_PLAN.md`
- `INT4_RESEARCH_FRAMEWORK.md`
- `INT4_VS_INT8_STRATEGY.md`
- `INT4_QUANTIZATION_DETAILS.md`

❌ **比较和解释文档**
- `QAT_COMPARISON.md`
- `QAT_IMPROVEMENTS_COMPARISON.md`
- `QAT_SCRIPT_COMPARISON.md`
- `QAT_RESULTS_ANALYSIS.md`
- `OPTIMIZATION_DETAILS_EXPLAINED.md`
- `FRAMERATE_OPTIMIZATIONS_EXPLANATION.md`
- `TEACHER_REQUIREMENTS_EXPLANATION.md`

❌ **项目计划和总结**
- `PROJECT_FINAL_SUMMARY.md`
- `IMPLEMENTATION_PLAN.md`
- `GRADUATION_IMPROVEMENT_PLAN.md`
- `IMPROVEMENTS.md`

#### 建议保留的文档（核心必要）
✅ **项目说明**
- `README.md` - 项目整体说明

✅ **脚本映射**
- `SCRIPT_DATASET_MAPPING.md` - 脚本与数据集对应关系

✅ **创新策略**
- `QUANTIZATION_STRATEGY_INNOVATIONS.md` - 量化策略创新说明

✅ **日志说明**
- `README_LOGGING.md` - 日志功能说明

✅ **脚本审查报告**
- `SCRIPT_REVIEW_REPORT.md` - 脚本功能与测试配置审查

## 二、清理操作

### 1. 删除单个文件
```bash
# 删除特定INT4临时指南
rm INT4_OPTION2_ONLY_GUIDE.md INT4_STEP_BY_STEP_GUIDE.md
```

### 2. 批量删除
```bash
# 批量删除所有INT4相关临时指南
rm INT4_*.md

# 批量删除比较和解释文档
rm *COMPARISON.md *EXPLAINED.md *EXPLANATION.md

# 批量删除项目计划和总结
rm *PLAN.md *SUMMARY.md IMPROVEMENTS.md
```

## 三、清理后文件夹结构

清理后，您的文件夹将只包含：
- **核心脚本文件**：INT8和INT4 QAT脚本，以及测试脚本
- **必要文档**：项目说明、脚本映射、创新策略等核心文档
- **模型文件夹**：`models/` - 存放训练好的模型
- **日志文件夹**：`logs/` - 存放训练和测试日志
- **数据文件夹**：`data/` - 存放数据集

这样可以保持文件夹整洁，便于您专注于核心实验工作！