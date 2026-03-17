import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torch.utils.data import DataLoader
import os
import time

# =========================
# 1 基础配置
# =========================
batch_size = 128
epochs = 5
lr = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("models", exist_ok=True)

# =========================
# 2 CIFAR10 数据
# =========================
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False)

# =========================
# 3 构建QAT模型
# =========================
print("Initializing QAT ResNet18...")

model = resnet18(pretrained=True, quantize=False)

# 修改分类层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,10)

# 融合层
model.train()
model.fuse_model()

# QAT配置
model.qconfig = quant.get_default_qat_qconfig("fbgemm")

# 插入FakeQuant
quant.prepare_qat(model, inplace=True)

model = model.to(device)

print("QAT model ready")

# =========================
# 4 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5)

# =========================
# 5 训练
# =========================
best_acc = 0

for epoch in range(epochs):

    model.train()

    total = 0
    correct = 0
    loss_sum = 0

    start = time.time()

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        _,pred = outputs.max(1)

        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100*correct/total

    # =================
    # 测试
    # =================
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for images,labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,pred = outputs.max(1)

            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    test_acc = 100*correct/total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Test Acc {test_acc:.2f}% | "
        f"Time {time.time()-start:.1f}s"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                   "models/resnet18_qat.pth")

    scheduler.step()

print("\nBest QAT Accuracy:",best_acc)

# =========================
# 6 INT8转换
# =========================
print("\nConverting to INT8...")

model.eval()

model_cpu = model.to("cpu")

model_int8 = quant.convert(model_cpu, inplace=False)

torch.save(model_int8.state_dict(),
           "models/resnet18_int8.pth")

print("INT8 model saved")

# =========================
# 7 INT8测试
# =========================
correct = 0
total = 0

start = time.time()

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to("cpu")
        labels = labels.to("cpu")

        outputs = model_int8(images)

        _,pred = outputs.max(1)

        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

int8_acc = 100*correct/total

print("INT8 Accuracy:",int8_acc)
print("Inference time:",time.time()-start)