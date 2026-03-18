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
torch.backends.quantized.engine = 'qnnpack'

# 自动定位路径
if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 日志记录
log_path = os.path.join(log_dir, f"qat_mobilenetv2_{time.strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 2. 数据处理 (对齐 224 分辨率) ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform_qat),
    batch_size=128, shuffle=False, num_workers=4)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, transform=transform_qat),
    batch_size=128, shuffle=True, num_workers=4)

# --- 3. 加载 FP32 权重 ---
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
if not os.path.exists(fp32_path):
    logger.error(f"找不到权重文件: {fp32_path}，请先运行 FP32 脚本！")
    exit()

checkpoint = torch.load(fp32_path, map_location='cpu')
# 过滤权重，确保能对齐
model.load_state_dict({k: v for k, v in checkpoint.items() if k in model.state_dict()}, strict=False)
model.to(device)
logger.info(f"成功加载 FP32 基准权重: {fp32_path}")

# --- 4. QAT 配置 ---
model.eval()
model.fuse_model(is_qat=True)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 微调循环 ---
best_sim_acc = 0.0
for epoch in range(15): # QAT 不需要太久，15个 Epoch 足够
    model.train()
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    # 评估
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
        torch.save(model.state_dict(), os.path.join(model_dir, "mobilenetv2_qat_best.pth"))

# --- 6. 最终物理转换 ---
logger.info("\n--- 执行物理量化转换 (FP32 -> INT8) ---")
model.load_state_dict(torch.load(os.path.join(model_dir, "mobilenetv2_qat_best.pth")))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 保存部署模型
deploy_path = os.path.join(model_dir, "mobilenetv2_int8_deploy.pt")
torch.save(int8_model.state_dict(), deploy_path)

# 计算体积
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
int8_size = os.path.getsize(deploy_path) / (1024 * 1024)

logger.info(f"FP32 体积: {fp32_size:.2f} MB | INT8 体积: {int8_size:.2f} MB")
logger.info(f"模型压缩比: {fp32_size/int8_size:.2f}x")
logger.info(f"最终 INT8 部署精度预计在 {best_sim_acc:.2f}% 左右")