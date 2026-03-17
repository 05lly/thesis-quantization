import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torchvision.models import ResNet18_Weights  # 修复导入路径
from torch.utils.data import DataLoader
import os
import time

# =========================
# 1 基础配置
# =========================
batch_size = 128
epochs = 5  # 依旧保持较小 epoch 用于快速测试流程
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs("models", exist_ok=True)

# 论文点：Resize(224) 使模型能够充分利用 ImageNet 预训练特征，是高精度的关键
transform_common = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, padding=4)] + transform_common)
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True,
    transform=transforms.Compose(transform_common)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# =========================
# 2 构建 QAT 模型
# =========================
print("Initializing QAT ResNet18...")
# 使用 quantization 版本的 ResNet18，它带有 fuse_model 方法
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, quantize=False)

# 修改分类头
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model.train()
# 算子融合：Conv+BN+ReLU 合并，减少量化误差
model.fuse_model(is_qat=True)

# 配置 QAT
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)

model = model.to(device)
print("QAT model ready.")

# =========================
# 3 训练/微调 (Fine-tuning)
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Starting QAT training...")
best_acc = 0

for epoch in range(epochs):
    model.train()
    
    # 优化点：最后 2 个 Epoch 冻结观察者和 BN 统计量，使量化参数稳定
    if epoch >= epochs - 2:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        print(f"Epoch {epoch+1}: Observers and BN stats frozen.")

    start_time = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% | Time: {time.time()-start_time:.1f}s")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/resnet18_qat_temp.pth")

print(f"\nBest QAT Accuracy: {best_acc:.2f}%")

# =========================
# 4 转换 INT8
# =========================
print("\nConverting to INT8...")
model.eval()
model_cpu = model.to("cpu")
# 将 FakeQuant 节点转换为真正的定点运算
model_int8 = quant.convert(model_cpu, inplace=False)

torch.save(model_int8.state_dict(), "models/resnet18_int8_final.pth")
print("INT8 model saved.")

# =========================
# 5 论文数据分析 (对比评估)
# =========================
def evaluate_final(model, name):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    end = time.time()
    acc = 100. * correct / total
    latency = (end - start) / len(test_loader.dataset) * 1000 # 单张延迟 ms
    print(f"[{name}] Acc: {acc:.2f}% | Latency: {latency:.4f} ms/img")
    return acc

print("\n" + "="*40)
print("FINAL RESULTS FOR THESIS")
print("="*40)

# 测试 QAT 浮点模型 (伪量化)
evaluate_final(model_cpu, "QAT-Float (Pseudo)")
# 测试 转换后的 INT8 模型 (真实量化)
evaluate_final(model_int8, "INT8-Quantized")

# 文件体积对比
size_float = os.path.getsize("models/resnet18_qat_temp.pth") / (1024*1024)
size_int8 = os.path.getsize("models/resnet18_int8_final.pth") / (1024*1024)
print(f"FP32 Weight Size: {size_float:.2f} MB")
print(f"INT8 Weight Size: {size_int8:.2f} MB")
print(f"Compression Ratio: {size_float/size_int8:.2f}x")
print("="*40)