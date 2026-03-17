import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torch.utils.data import DataLoader
import os
import time
import datetime
from tqdm import tqdm

# -----------------------------
# 1. 基础配置
# -----------------------------
data_root = "./data"
batch_size = 128
epochs = 10          # 微调 10 个 epoch
lr = 0.0001
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir, f"qat_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

def write_log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(line + "\n")

write_log("="*80)
write_log("ResNet18 QAT CIFAR-10 实验 - 日志记录开始")
write_log(f"设备: {device}, Batch Size: {batch_size}, Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
write_log("="*80)

# -----------------------------
# 2. 数据集
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

write_log(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

# -----------------------------
# 3. QAT 模型
# -----------------------------
write_log("初始化 ResNet18 QAT 模型...")

model = resnet18(pretrained=True, quantize=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,10)
model.train()
model.fuse_model()
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)
model = model.to(device)
write_log("QAT 模型准备完成")

# -----------------------------
# 4. 损失函数 & 优化器
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# 5. 训练循环
# -----------------------------
best_acc = 0.0
total_start_time = time.time()
write_log(f"{'Epoch':<6}{'TrainAcc':<12}{'TestAcc':<12}{'QuantSimAcc':<15}{'Loss':<12}{'LR':<10}")

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = 100*train_correct/train_total

    # -----------------------------
    # 测试
    # -----------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100*correct/total

    # QAT 模拟量化精度
    quant_sim_acc = test_acc

    lr_current = optimizer.param_groups[0]['lr']

    write_log(f"{epoch+1:<6}{train_acc:<12.2f}{test_acc:<12.2f}{quant_sim_acc:<15.2f}{train_loss:<12.4f}{lr_current:<10.6f}")

    # 保存最佳 QAT 模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), os.path.join(model_dir,"resnet18_qat_best.pth"))
        write_log(f"Epoch {epoch+1} 保存最佳模型 | TestAcc: {test_acc:.2f}%")

    scheduler.step()
    write_log(f"Epoch {epoch+1} 完成 | 耗时: {time.time()-epoch_start:.1f}s")

write_log(f"训练完成 | 总耗时: {time.time()-total_start_time:.1f}s | 最佳精度: {best_acc:.2f}%")

# -----------------------------
# 6. 转换 INT8
# -----------------------------
write_log("开始 QAT -> INT8 转换...")
model.eval()
model_cpu = model.to("cpu")
model_int8 = quant.convert(model_cpu, inplace=False)
torch.save(model_int8.state_dict(), os.path.join(model_dir,"resnet18_int8.pth"))
write_log("INT8 模型已保存")

# -----------------------------
# 7. INT8 测试
# -----------------------------
correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model_int8(images)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()

int8_acc = 100*correct/total
infer_time = time.time()-start_time

write_log(f"INT8 测试精度: {int8_acc:.2f}% | 推理总耗时: {infer_time:.2f}s | 单批次推理时间: {infer_time/len(test_loader)*1000:.2f} ms")
write_log("="*80)
write_log("实验完成 ✅")