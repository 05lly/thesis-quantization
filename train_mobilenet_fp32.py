import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# --- 1. 基础配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128  # 如果显存溢出(OOM)，请改为 64
epochs = 30
# 重要：保存到 Git 删不到的备份目录
model_dir = "/root/autodl-tmp/my_backup"
os.makedirs(model_dir, exist_ok=True)

# --- 2. 数据预处理 ---
# MobileNetV2 建议输入 224x224，CIFAR-10 的 32x32 太小会导致特征坍缩
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 载入数据
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 3. 模型定义 (迁移学习) ---
print("==> 正在加载 ImageNet 预训练的 MobileNetV2...")
# 使用最新的 weights 参数写法
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# 修改分类头：MobileNetV2 的 classifier 是一个 Sequential
# [0] 是 Dropout, [1] 是 Linear
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = model.to(device)

# --- 4. 优化器与损失函数 ---
criterion = nn.CrossEntropyLoss()
# 使用 SGD + Momentum 是刷高基准精度的标配
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# 余弦退火学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 5. 训练与验证循环 ---
best_acc = 0.0
print(f"==> 开始训练，设备: {device}")

for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    scheduler.step()
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100. * correct / total
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
    
    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        save_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  --> 精度提升！已保存至: {save_path}")

print(f"训练结束！最高精度: {best_acc:.2f}%")
print(f"请检查备份目录: {model_dir}")