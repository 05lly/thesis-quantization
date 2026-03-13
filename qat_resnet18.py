import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# ==================== 基础配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("QAT Training with GPU Support")
print("="*60)
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print(f"Quantized engines: {torch.backends.quantized.supported_engines}")

batch_size = 64
num_epochs = 15
learning_rate = 0.001
os.makedirs("models", exist_ok=True)

# ==================== CIFAR-10 数据 ====================
print("\n=== Loading CIFAR-10 Dataset ===")
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True if device.type=='cuda' else False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True if device.type=='cuda' else False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")

# ==================== 加载 FP32 Checkpoint ====================
print("\n=== Loading FP32 baseline model ===")
model_qat = models.resnet18(pretrained=False)
model_qat.fc = nn.Linear(model_qat.fc.in_features, 10)

fp32_path = "models/resnet18_fp32.pth"
if not os.path.exists(fp32_path):
    print(f"❌ FP32 checkpoint not found: {fp32_path}")
    print("Please train FP32 model first using train_fp32.py")
    exit(1)

checkpoint = torch.load(fp32_path, map_location='cpu')
model_qat.load_state_dict(checkpoint, strict=False)
print(f"✓ Loaded FP32 checkpoint: {fp32_path}")

# ==================== 模型融合函数 ====================
def fuse_resnet18(model):
    fuse_list = [
        ['conv1', 'bn1', 'relu'],
        ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],
        ['layer1.0.conv2', 'layer1.0.bn2'],
        ['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu'],
        ['layer1.1.conv2', 'layer1.1.bn2'],
        ['layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu'],
        ['layer2.0.conv2', 'layer2.0.bn2'],
        ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
        ['layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu'],
        ['layer2.1.conv2', 'layer2.1.bn2'],
        ['layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu'],
        ['layer3.0.conv2', 'layer3.0.bn2'],
        ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
        ['layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu'],
        ['layer3.1.conv2', 'layer3.1.bn2'],
        ['layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu'],
        ['layer4.0.conv2', 'layer4.0.bn2'],
        ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
        ['layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu'],
        ['layer4.1.conv2', 'layer4.1.bn2']
    ]
    fused_count = 0
    for fuse_set in fuse_list:
        try:
            tq.fuse_modules(model, fuse_set, inplace=True)
            fused_count += 1
        except Exception as e:
            print(f"  ⚠ Fusion warning for {fuse_set}: {str(e)[:50]}")
    print(f"✓ Fused {fused_count}/{len(fuse_list)} module groups")
    return model

# ==================== 设置量化后端 ====================
if 'fbgemm' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'fbgemm'
    print(f"✓ Using quantized engine: fbgemm")
else:
    torch.backends.quantized.engine = 'qnnpack'
    print(f"✓ Using quantized engine: qnnpack")

# ==================== QAT 准备 ====================
model_qat.train()  # ⚠ 必须在训练模式
model_qat = fuse_resnet18(model_qat)
model_qat.qconfig = tq.get_default_qat_qconfig(torch.backends.quantized.engine)

try:
    model_prepared = tq.prepare_qat(model_qat, inplace=False).to(device)
    print("✓ QAT preparation successful")
except Exception as e:
    print(f"❌ QAT preparation failed: {e}")
    exit(1)

# ==================== QAT 训练 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_prepared.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_qat_acc = 0.0
training_start = time.time()
log_file = open("qat_training_log.txt", "w")

for epoch in range(num_epochs):
    epoch_start = time.time()
    model_prepared.train()
    running_loss = 0.0
    correct, total = 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_prepared(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_prepared.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            log_file.write(f"{time.time()} Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}\n")

    train_acc = 100.*correct/total

    # 测试
    model_prepared.eval()
    correct_test, total_test = 0,0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_prepared(images)
            _, pred = outputs.max(1)
            correct_test += pred.eq(labels).sum().item()
            total_test += labels.size(0)
    test_acc = 100.*correct_test/total_test

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} | Loss {running_loss/len(train_loader):.4f} | Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}% | Time {epoch_time:.1f}s | LR {scheduler.get_last_lr()[0]:.6f}")
    log_file.write(f"Epoch {epoch+1} | Train Acc {train_acc:.2f} | Test Acc {test_acc:.2f}\n")

    if test_acc > best_qat_acc:
        best_qat_acc = test_acc
        torch.save(model_prepared.state_dict(), "models/resnet18_qat_best.pth")
        print(f"✔ Saved best QAT checkpoint (Acc: {test_acc:.2f}%)")

log_file.close()
total_time = time.time()-training_start
print(f"\n✓ QAT Training completed in {total_time/60:.1f} minutes")
print(f"✓ Best QAT Test Acc: {best_qat_acc:.2f}%")

# ==================== INT8 转换 ====================
print("\n=== Converting to INT8 ===")
model_prepared.eval()
model_prepared.to('cpu')

try:
    model_int8 = tq.convert(model_prepared, inplace=False)
    print("✓ INT8 conversion successful")
except Exception as e:
    print(f"❌ INT8 conversion failed: {e}")
    exit(1)

# ==================== INT8 测试 ====================
correct_int8, total_int8 = 0,0
inference_start = time.time()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to('cpu'), labels.to('cpu')
        outputs = model_int8(images)
        _, pred = outputs.max(1)
        correct_int8 += pred.eq(labels).sum().item()
        total_int8 += labels.size(0)
int8_acc = 100.*correct_int8/total_int8
inference_time = time.time() - inference_start
print(f"✓ INT8 Test Acc: {int8_acc:.2f}% | Inference time: {inference_time:.2f}s | Avg {inference_time/total_int8*1000:.2f}ms/img")

torch.save(model_int8.state_dict(), "models/resnet18_int8_final.pth")
print("✓ INT8 model saved to models/resnet18_int8_final.pth")

# ==================== 模型大小对比 ====================
fp32_size = os.path.getsize("models/resnet18_fp32.pth")/(1024*1024)
qat_size = os.path.getsize("models/resnet18_qat_best.pth")/(1024*1024)
int8_size = os.path.getsize("models/resnet18_int8_final.pth")/(1024*1024)
print("\n" + "="*60)
print("MODEL SIZE COMPARISON")
print("="*60)
print(f"{'Model':<20} {'Size(MB)':<15} {'Ratio':<10}")
print(f"{'FP32':<20} {fp32_size:<15.2f} {'1.00x':<10}")
print(f"{'QAT (FP32)':<20} {qat_size:<15.2f} {fp32_size/qat_size:<5.2f}x")
print(f"{'INT8':<20} {int8_size:<15.2f} {fp32_size/int8_size:<5.2f}x")
print(f"\nAccuracy Summary:")
print(f"  FP32 Baseline: (请从训练日志获取)")
print(f"  QAT Best: {best_qat_acc:.2f}%")
print(f"  INT8 Final: {int8_acc:.2f}%")
print(f"  Accuracy drop: {best_qat_acc-int8_acc:.2f}%")
print("="*60)
print("Training Completed Successfully!")