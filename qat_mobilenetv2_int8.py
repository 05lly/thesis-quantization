-import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os
import time
import logging

# --- 1. 日志与环境初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 针对树莓派5（ARM 架构），量化后端必须用 qnnpack
torch.backends.quantized.engine = 'qnnpack'

model_dir, log_dir = "models", "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 毕设建议：把实验过程存成 log，方便写论文时翻数据
log_path = os.path.join(log_dir, f"qat_log_{time.strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 2. 数据加载 (对齐 FP32 的 224 分辨率) ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 简单起见，测试集不用随机翻转
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainloader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, transform=transform_qat), 
                                          batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_val), 
                                         batch_size=128, shuffle=False, num_workers=4)

# --- 3. 模型准备与权重对齐 ---
# 使用 quantization 专门的模型结构，因为多了量化观察器桩点
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2.pth")
if not os.path.exists(fp32_path):
    logger.error("错误：找不到 FP32 权重文件，请先跑 train_fp32 脚本！")
    exit()

# 严谨加载：通过映射确保 FP32 权重精准填入量化模型的对应层
checkpoint = torch.load(fp32_path, map_location='cpu')
model.load_state_dict({k: v for k, v in checkpoint.items() if k in model.state_dict()}, strict=False)
model.to(device)

# --- 4. QAT 核心配置 ---
model.eval()
model.fuse_model(is_qat=True) # 融合 Conv+BN+ReLU，减少推理开销
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# QAT 建议学习率小一点，1e-4 是个比较稳妥的值
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. 正式训练循环 ---
logger.info("开始 MobileNetV2 QAT 性能恢复微调...")
best_sim_acc = 0.0

for epoch in range(20):
    model.train()
    # 冻结 BN：第 4 个 Epoch 后固定 BN 的均值和方差，让量化范围更稳定
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device) # 修正：确保 labels 是 Tensor
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 评估 QAT 模拟量化精度
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    sim_acc = 100 * correct / total
    logger.info(f"Epoch [{epoch+1}/20] Sim Acc: {sim_acc:.2f}%")
    
    if sim_acc > best_sim_acc:
        best_sim_acc = sim_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "mobilenetv2_qat_best.pth"))

# --- 6. 物理模型转换与体积/精度评估 ---
logger.info("\n--- 正在生成最终 INT8 部署模型 ---")
model.load_state_dict(torch.load(os.path.join(model_dir, "mobilenetv2_qat_best.pth")))
model.to('cpu').eval()
# 这里是真正的物理转换，参数从浮点转成 8bit 整数
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 保存最终要在树莓派上用的文件
deploy_path = os.path.join(model_dir, "mobilenetv2_int8_deploy.pt")
torch.save(int8_model.state_dict(), deploy_path)

# 真实精度测试
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = int8_model(inputs) # 此时运行的是物理量化算子
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
final_acc = 100 * correct / total

# 体积计算
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
int8_size = os.path.getsize(deploy_path) / (1024 * 1024)

# --- 7. 输出论文报表 ---
logger.info("\n" + "="*35)
logger.info(f"FP32 原始体积: {fp32_size:.2f} MB")
logger.info(f"INT8 压缩体积: {int8_size:.2f} MB")
logger.info(f"最终物理精度: {final_acc:.2f}%")
logger.info(f"模型压缩比: {fp32_size/int8_size:.2f}x")
logger.info("="*35)
logger.info(f"实验日志已保存至: {log_path}")