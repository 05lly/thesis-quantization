# 脚本与数据集对应关系说明

## 一、现有脚本数据集对应关系

| 脚本文件名 | 对应的数据集 | 功能说明 |
|------------|--------------|----------|
| `resnet18_int8_optimized.py` | **CIFAR-10** | 基础版INT8量化优化脚本 |
| `qat_resnet18_optimized.py` | **CIFAR-10** | 优化版QAT量化脚本（带层敏感度分析、渐进式QAT等） |
| `qat_resnet18_c100_optimized.py` | **CIFAR-100** | 优化版QAT量化脚本（用于CIFAR-100数据集） |
| `resnet18_int4_real.py` | **支持CIFAR-10/CIFAR-100** | INT4真实量化脚本（通过参数选择数据集） |
| `test_pi5_optimized.py` | **自动匹配** | 树莓派性能测试脚本（根据模型名自动匹配数据集） |

## 二、如何识别脚本对应的数据集

### 2.1 通过脚本内容识别

#### CIFAR-10脚本的特征：
- 使用`datasets.CIFAR10()`加载数据
- 归一化参数：`transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))`
- 分类数：10个类别
- 通常不包含"c100"字样

#### CIFAR-100脚本的特征：
- 使用`datasets.CIFAR100()`加载数据
- 归一化参数：`transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))`
- 分类数：100个类别
- 通常包含"c100"字样

### 2.2 具体脚本分析

#### 1. `resnet18_int8_optimized.py` (CIFAR-10)
```python
# 数据加载
# CIFAR-10 标准均值标准差
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
testset = datasets.CIFAR10(...)  # 使用CIFAR10数据集
```

#### 2. `qat_resnet18_optimized.py` (CIFAR-10)
```python
# 数据加载
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10归一化参数
])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_qat),  # 使用CIFAR10数据集
    ...
)
```

#### 3. `qat_resnet18_c100_optimized.py` (CIFAR-100)
```python
# 日志明确标识
log_message(f"Environment: {device} | Task: CIFAR-100 QAT | ...")

# 数据加载
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # CIFAR100归一化参数
])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_qat),  # 使用CIFAR100数据集
    ...
)
```

## 三、关于您之前运行的脚本

### 3.1 确认您之前运行的脚本
根据您提供的运行结果日志，最后一行显示：
```
[2026-05-26 16:10:04] ResNet18 CIFAR-10 Optimized QAT Results
```

这表明您之前运行的是**`qat_resnet18_optimized.py`**脚本，对应**CIFAR-10**数据集。

### 3.2 日志与脚本对应关系
- 该脚本生成的日志文件名格式：`qat_resnet18_optimized_YYYYMMDD_HHMMSS.log`
- 您之前运行生成的日志已经保存在`logs/`目录下

## 四、脚本命名保证

我**不会修改**任何已经生成日志的脚本文件名，包括：
- `qat_resnet18_optimized.py`（您刚刚运行过的）
- `qat_resnet18_c100_optimized.py`
- `resnet18_int8_optimized.py`

所有新创建的脚本会使用明确的命名规则，避免混淆。

## 五、使用建议

1. **运行CIFAR-10实验**：使用`qat_resnet18_optimized.py`
2. **运行CIFAR-100实验**：使用`qat_resnet18_c100_optimized.py`
3. **运行INT4实验**：使用`resnet18_int4_real.py --dataset cifar10`或`resnet18_int4_real.py --dataset cifar100`
4. **树莓派测试**：使用`test_pi5_optimized.py --model <模型文件路径>`

## 六、总结

| 任务 | 推荐脚本 | 命令示例 |
|------|----------|----------|
| CIFAR-10 QAT优化 | qat_resnet18_optimized.py | `python qat_resnet18_optimized.py` |
| CIFAR-100 QAT优化 | qat_resnet18_c100_optimized.py | `python qat_resnet18_c100_optimized.py` |
| CIFAR-10 INT4量化 | resnet18_int4_real.py | `python resnet18_int4_real.py --dataset cifar10` |
| CIFAR-100 INT4量化 | resnet18_int4_real.py | `python resnet18_int4_real.py --dataset cifar100` |
| 树莓派性能测试 | test_pi5_optimized.py | `python test_pi5_optimized.py --model models/resnet18_c10_int8_deploy_optimized.pt` |