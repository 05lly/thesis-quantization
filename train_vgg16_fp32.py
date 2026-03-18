import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 参数配置 (严格对齐) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30 
lr = 0.005  # VGG16 建议起步稍微稳一点

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

# --- 2. 统一日志函数 ---
log_filename = os.path.join(log_dir, f"fp32_vgg16_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"; print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

log_message(f"Environment: {device} | Batch Size: {batch_size} | Epochs: {epochs} | Mode: VGG16-FP32")

# --- 3. 数据处理 (统一 224 分辨率) ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainloader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), 
                                          batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=transform), 
                                         batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 模型构建 (迁移学习) ---
log_message("Loading VGG16 with ImageNet weights...")
# 使用 quantization 版本的 VGG16 定义以确保后续 QAT 结构兼容
model = models.quantization.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, quantize=False)
model.classifier[6] = nn.Linear(4096, 10) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 5. 训练循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'LR':<15}")

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for inputs, labels in tqdm(trainloader, desc=f"VGG16 FP32 Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        _, pred = torch.max(outputs, 1); total += labels.size(0); correct += (pred == labels).sum().item()
    
    scheduler.step()
    model.eval(); test_correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1); test_correct += (pred == labels).sum().item()
    
    val_acc = 100. * test_correct / 10000
    log_message(f"{epoch+1:<10}{100.*correct/total:<15.2f}{val_acc:<15.2f}{scheduler.get_last_lr()[0]:<15.6f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "fp32_vgg16_best.pth"))
        log_message(f"New Best Accuracy: {best_acc:.2f}%")

log_message(f"VGG16 FP32 Finished. Best: {best_acc:.2f}% | Time: {(time.time()-start_time)/60:.2f} mins")