import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import datetime
from tqdm import tqdm

# -----------------------------
# 1. 环境与参数配置
# -----------------------------
data_root = "./data"
batch_size = 128
num_epochs = 30
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def logger(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

logger(f"Environment: {device} | Batch Size: {batch_size} | Epochs: {num_epochs}")

# -----------------------------
# 2. 数据处理流
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

train_data = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# 3. 网络构建
# -----------------------------
# 使用预训练权重并适配 CIFAR-10 类别数
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# -----------------------------
# 4. 优化器与策略
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

# -----------------------------
# 5. 主训练循环
# -----------------------------
best_accuracy = 0.0
start_wall_time = time.time()

logger(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}{'LR':<10}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # 训练阶段
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total_samples += targets.size(0)
        correct_preds += preds.eq(targets).sum().item()

    avg_train_loss = running_loss / len(train_data)
    train_acc = 100. * correct_preds / total_samples

    # 评估阶段
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            test_total += targets.size(0)
            test_correct += preds.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    curr_lr = optimizer.param_groups[0]['lr']

    logger(f"{epoch+1:<10}{train_acc:<15.2f}{test_acc:<15.2f}{avg_train_loss:<15.4f}{curr_lr:<10.6f}")

    # 保存最优模型
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        
        # 保存完整 checkpoint (包含优化器状态)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'accuracy': best_accuracy
        }
        torch.save(checkpoint, os.path.join(model_dir, "checkpoint_best.pth"))
        
        # 按要求保存纯权重文件
        torch.save(model.state_dict(), os.path.join(model_dir, "fp32_resnet18.pth"))
        logger(f"--- Saved best model: {best_accuracy:.2f}% ---")

    scheduler.step()

# -----------------------------
# 6. 结果汇总
# -----------------------------
total_duration = (time.time() - start_wall_time) / 60
model_file_path = os.path.join(model_dir, "fp32_resnet18.pth")
final_size = os.path.getsize(model_file_path) / (1024 * 1024)

logger("=" * 50)
logger(f"Training Complete.")
logger(f"Best Test Accuracy: {best_accuracy:.2f}%")
logger(f"Model Saved As: {model_file_path}")
logger(f"Model Size: {final_size:.2f} MB")
logger(f"Total Time: {total_duration:.2f} mins")
logger("=" * 50)