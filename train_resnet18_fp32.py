import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import datetime

# -----------------------------
# 1. 基本配置
# -----------------------------
batch_size = 128
num_epochs = 20
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建模型保存目录
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 日志文件名
log_file = f"logs/resnet18_fp32_gpu_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# -----------------------------
# 2. CIFAR-10 数据预处理
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# 3. ResNet18 模型
# -----------------------------
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # CIFAR-10 10 类
model = model.to(device)

# -----------------------------
# 4. 损失函数 & 优化器
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# 5. 训练 & 日志
# -----------------------------
best_acc = 0.0
with open(log_file, "w") as f_log:
    f_log.write("epoch,train_loss,train_acc,test_acc\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        f_log.write(f"\nEpoch [{epoch+1}/{num_epochs}]\n")

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        f_log.write(f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%\n")

        # -----------------------------
        # 测试
        # -----------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        print(f"Test Acc: {test_acc:.2f}%")
        f_log.write(f"test_acc={test_acc:.2f}%\n")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/resnet18_fp32_gpu.pth")
            print("✔ Best ResNet18 FP32 model saved")

        # 保存每个epoch的数据到csv风格文件，方便画图
        with open(log_file, "a") as f_csv:
            f_csv.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{test_acc:.2f}\n")

print("\nTraining Finished")
print(f"Best Test Accuracy: {best_acc:.2f}%")