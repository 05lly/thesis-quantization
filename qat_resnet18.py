import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18 as qat_resnet18
from torch.utils.data import DataLoader
import os
import time
import datetime
from tqdm import tqdm

# -----------------------------
# 1. 参数配置
# -----------------------------
data_root = "./data"
batch_size = 128
epochs = 10         
lr = 1e-5           # QAT阶段采用极小学习率以微调量化参数
momentum = 0.9
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"train_qat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message("QAT Training session started.")
log_message(f"Config: Epochs={epochs}, LR={lr}, Device={device}")

# -----------------------------
# 2. 数据处理流 (保持与FP32 Baseline一致)
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# 3. 构建 QAT 模型并加载 FP32 权重
# -----------------------------
# 加载支持量化算子的模型结构
model = qat_resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 10)

# 载入预训练好的 FP32 基准权重 (确保接力逻辑)
fp32_weight_path = os.path.join(model_dir, "fp32_resnet18.pth")
if os.path.exists(fp32_weight_path):
    model.load_state_dict(torch.load(fp32_weight_path, map_location='cpu')，)
    log_message(f"Checkpoint loaded: {fp32_weight_path}")
else:
    log_message("Warning: Baseline weights not found. Check path.")

# 准备 QAT 流程
model.train()
model.fuse_model()  # 算子融合 (Conv+BN+ReLU)
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)
model.to(device)

# -----------------------------
# 4. 优化器与损失函数
# -----------------------------
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# 5. 训练微调
# -----------------------------
best_acc = 0.0
start_time = time.time()

log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"QAT Epoch {epoch+1}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 验证模拟量化精度
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * test_correct / len(test_set)
    train_acc = 100. * correct / total
    epoch_loss = running_loss / len(train_set)

    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{epoch_loss:<15.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "resnet18_qat_best.pth"))
        log_message(f"--- Best QAT Accuracy Updated: {best_acc:.2f}% ---")

# -----------------------------
# 6. 转换并序列化模型 (部署优化)
# -----------------------------
log_message("Converting QAT model to deployed INT8 format...")
model.load_state_dict(torch.load(os.path.join(model_dir, "resnet18_qat_best.pth"), map_location=device))

log_message("Converting QAT model to deployed INT8 format...")
model.eval()
model.to('cpu')

# 1. 转换为量化模型 (INT8)
int8_model = quant.convert(model, inplace=False)

# 2. 导出为 TorchScript 格式 (关键：用于树莓派部署)
# 使用 Trace 方法记录计算图，这样模型就包含了结构信息
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)

# 3. 保存两种格式：
# 纯权重格式（用于继续在 PC/PyTorch 环境下分析调试）
weights_path = os.path.join(model_dir, "resnet18_int8_final.pth")
torch.save(int8_model.state_dict(), weights_path)

# TorchScript格式（用于树莓派脱机部署）
deploy_path = os.path.join(model_dir, "resnet18_int8_deploy.pt")
torch.jit.save(traced_model, deploy_path)

# -----------------------------
# 7. 性能对比与文件体积记录
# -----------------------------
def get_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

fp32_size = get_size_mb(fp32_weight_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 50)
log_message(f"QAT Process Finished.")
log_message(f"Best Simulation Accuracy: {best_acc:.2f}%")
log_message(f"Deployment Model Saved: {deploy_path}")
if fp32_size > 0:
    log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
    log_message(f"INT8 Deploy Size: {int8_size:.2f} MB")
    log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message(f"Total Training Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 50)
log_message("Experiment Complete. Ready for Raspberry Pi 5. ")