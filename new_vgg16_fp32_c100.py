import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import datetime
from tqdm import tqdm

# --- 1. 核心参数配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 120          # 增加到 120 轮，确保充分收敛
lr = 0.01             # 初始学习率
weight_decay = 5e-4    # 增大权重衰减，抑制 VGG16 过拟合

# 路径配置
model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志系统 ---
log_filename = os.path.join(log_dir, f"fp32_vgg16_enhanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 3. 可量化 VGG16 结构 ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(QuantizableVGG16, self).__init__()
        # 加载 ImageNet 预训练权重
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        # 修改分类层为 CIFAR-100 的 100 类
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        # 量化占位符
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# --- 4. 高强度数据增强 ---
norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),             # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.ColorJitter(0.2, 0.2, 0.2),  # 颜色抖动，防止过拟合
    transforms.ToTensor(),
    norm
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    norm
])

trainloader = torch.utils.data.DataLoader(
    datasets.CIFAR100('/root/autodl-tmp/data', train=True, download=True, transform=transform_train),
    batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR100('/root/autodl-tmp/data', train=False, download=True, transform=transform_test),
    batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# --- 5. 初始化模型与优化器 ---
model = QuantizableVGG16(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

# 使用余弦退火， eta_min 设为极小值，让学习率平滑下降
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# --- 6. 训练与验证逻辑 ---
best_acc = 0.0
start_time = time.time()
log_message("Starting Enhanced FP32 Training for VGG16...")

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    scheduler.step()
    
    # 验证
    model.eval()
    t_corr, t_tot = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            t_tot += labels.size(0)
            t_corr += (pred == labels).sum().item()
    
    val_acc = 100. * t_corr / t_tot
    train_acc = 100. * correct / total
    current_lr = scheduler.get_last_lr()[0]
    
    log_message(f"Epoch {epoch+1:03d} | TrainAcc: {train_acc:.2f}% | TestAcc: {val_acc:.2f}% | LR: {current_lr:.6f}")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join(model_dir, "new_fp32_vgg16_c100_best.pth")
        torch.save(model.state_dict(), save_path)
        log_message(f"*** New Best: {best_acc:.2f}% (Saved) ***")

# --- 7. 总结 ---
total_mins = (time.time() - start_time) / 60
log_message("=" * 60)
log_message(f"Enhanced Training Finished!")
log_message(f"Best Accuracy: {best_acc:.2f}%")
log_message(f"Total Time: {total_mins:.2f} mins")
log_message("=" * 60)