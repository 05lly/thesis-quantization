import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# --- 实验预处理 (为了严谨，基准和量化必须完全一套预处理) ---
# 注意：MobileNetV2 在 32x32 下由于下采样太多，特征会坍缩，这里必须拉伸到 224
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 载入 CIFAR-10
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- 模型定义：迁移学习方案 ---
print("[开始加载模型] 使用 ImageNet 预训练的 MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# 修改最后的分类层，改成我们 CIFAR-10 的 10 类
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = model.to(device)

# --- 优化策略 ---
criterion = nn.CrossEntropyLoss()
# 选 SGD 加余弦退火，这是为了让基准精度尽可能跑高，方便后续量化对比
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 训练循环 ---
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    scheduler.step()
    
    # 每个 epoch 测一次精度
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")
    
    # 记录表现最好的 FP32 权重，给 QAT 当初始化参数
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(model_dir, "fp32_mobilenetv2.pth"))
        print(f"--> 已保存当前最优 FP32 权重，准确率: {best_acc:.2f}%")

print("FP32 基准实验训练完成。")