import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os
import time

# =========================
# 1 基础配置
# =========================
batch_size = 128
epochs = 5  # 依旧保持较小 epoch 用于测试
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)

# 数据增强保持与 FP32 一致，确保输入尺寸为 224
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_loader = DataLoader(
    datasets.CIFAR10(root="./data", train=True, download=True, 
                     transform=transforms.Compose([
                         transforms.Resize(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                     ])),
    batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test),
    batch_size=batch_size, shuffle=False)

# =========================
# 2 构建 QAT 模型
# =========================
print("Initializing QAT ResNet18...")
# 使用更现代的 weights 参数避免警告
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 10)

model.train()
model.fuse_model(is_qat=True) # 明确指定为 QAT 融合

# 配置量化参数 (fbgemm 适用于 x86 CPU)
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# =========================
# 3 训练与微调
# =========================
print("Start QAT Training...")
for epoch in range(epochs):
    model.train()
    
    # 关键优化：在最后阶段冻结 BN 统计量和量化观察器
    if epoch >= epochs - 2:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        print(f"--- Epoch {epoch+1}: Freezing BN and Observers ---")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # 每个 Epoch 验证一次
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs} | Test Acc: {100*correct/total:.2f}%")

# =========================
# 4 转换 INT8 并做最终评估
# =========================
print("\nConverting to INT8...")
model.eval()
model_cpu = model.to("cpu")
model_int8 = quant.convert(model_cpu, inplace=False)

# 保存模型用于体积对比
torch.save(model.state_dict(), "models/temp_qat_float.pth")
torch.save(model_int8.state_dict(), "models/resnet18_int8_final.pth")

# 评估数据统计
def get_size(file):
    return os.path.getsize(file) / (1024 * 1024)

def benchmark(model, loader, name):
    model.eval()
    start = time.time()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    end = time.time()
    acc = 100 * correct / total
    latency = (end - start) / len(loader.dataset) * 1000
    print(f"{name} -> Acc: {acc:.2f}%, Latency: {latency:.4f}ms/img")
    return acc, latency

print("\n" + "="*40)
print("论文实验数据支撑 (Testing on CPU)")
print("="*40)
# 这里的对比是转换前的伪量化模型 vs 转换后的真实 INT8 模型
benchmark(model_cpu, test_loader, "QAT-Float(Pseudo)")
acc_8, lat_8 = benchmark(model_int8, test_loader, "INT8-Quantized")

size_f = get_size("models/temp_qat_float.pth")
size_8 = get_size("models/resnet18_int8_final.pth")
print(f"Model Compression: {size_f:.2f}MB -> {size_8:.2f}MB ({(1-size_8/size_f)*100:.1f}% reduced)")
print("="*40)