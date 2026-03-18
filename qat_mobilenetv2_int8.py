import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os
import time
import logging

# --- 1. 环境与自适应路径初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 树莓派 5 核心加速引擎：qnnpack
torch.backends.quantized.engine = 'qnnpack'

# 自动定位路径：优先 AutoDL 备份区，次之本地 models
if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"

log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 日志记录：记录量化过程中的精度恢复曲线
log_path = os.path.join(log_dir, f"qat_mobilenetv2_{time.strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 2. 数据处理 (对齐 FP32 的 224 分辨率) ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_qat),
    batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform_qat),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# --- 3. 模型准备与权重加载 ---
# 必须使用 quantization 分支的模型，以便插入伪量化算子
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
if not os.path.exists(fp32_path):
    logger.error(f" 错误：找不到基准权重 {fp32_path}")
    exit()

# 加载刚才跑出的 96.52% 的权重
checkpoint = torch.load(fp32_path, map_location='cpu', weights_only=True)
model.load_state_dict({k: v for k, v in checkpoint.items() if k in model.state_dict()}, strict=False)
model.to(device)
logger.info(f"成功加载 FP32 基准权重: {fp32_path}")

# --- 4. QAT 核心配置 ---
model.eval()               # 先进入 eval 模式
model.fuse_model(is_qat=True) # 融合 Conv+BN+ReLU

model.train()              # 切换回 train 模式以开启 prepare_qat
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# QAT 微调：使用极低学习率，防止破坏已有的特征表达
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 训练循环 ---
logger.info(" 开始 MobileNetV2 QAT 精度恢复训练...")
best_sim_acc = 0.0

for epoch in range(15):
    model.train()
    # 冻结：第 4 个 Epoch 后固定 BN 统计量和观察器，稳定量化参数
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    # 评估：此时运行的是“模拟量化（Fake Quantization）”
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    sim_acc = 100 * correct / total
    logger.info(f"Epoch [{epoch+1}/15] Sim Acc: {sim_acc:.2f}%")
    
    if sim_acc > best_sim_acc:
        best_sim_acc = sim_acc
        save_path = os.path.join(model_dir, "mobilenetv2_qat_best.pth")
        torch.save(model.state_dict(), save_path)

# --- 6. 物理量化转换与序列化 (部署优化核心) ---
logger.info("\n--- 正在生成最终 INT8 物理部署模型 ---")
# 加载 QAT 训练出的最优参数
model.load_state_dict(torch.load(os.path.join(model_dir, "mobilenetv2_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()

# 1. 物理转换：将权重正式转为 int8 格式
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 2. 导出 TorchScript 格式 (包含结构信息，树莓派 5 专用)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)

# 3. 保存两个文件：用于对比体积和实际部署
weights_path = os.path.join(model_dir, "mobilenetv2_int8_final.pth")
deploy_path = os.path.join(model_dir, "mobilenetv2_int8_deploy.pt")

torch.save(int8_model.state_dict(), weights_path)
torch.jit.save(traced_model, deploy_path)

# --- 7. 实验报表输出 ---
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
int8_size = os.path.getsize(deploy_path) / (1024 * 1024)

logger.info("\n" + "="*45)
logger.info(f"MobileNetV2 实验总结:")
logger.info(f"FP32 原始大小: {fp32_size:.2f} MB")
logger.info(f"INT8 部署大小: {int8_size:.2f} MB")
logger.info(f"模型压缩比例: {fp32_size/int8_size:.2f}x")
logger.info(f"最高模拟精度: {best_sim_acc:.2f}%")
logger.info(f"部署文件路径: {deploy_path}")
logger.info("="*45)
logger.info(f"实验数据已记录至: {log_path}")