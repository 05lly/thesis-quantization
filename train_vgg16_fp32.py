import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 环境与日志配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, epochs, lr = 128, 30, 0.01
save_dir = "/root/autodl-tmp/my_backup"
log_dir = "logs"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 论文级别日志系统
log_path = os.path.join(log_dir, f"fp32_vgg16_{datetime.datetime.now().strftime('%m%d_%H%M')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_path, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

# --- 2. 自定义量化兼容模型 (对齐论文 2.2-3) ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizableVGG16, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)
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

# --- 3. 数据处理 (224 分辨率) ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=batch_size, shuffle=False, num_workers=4)

# --- 4. 训练初始化 ---
log_message("Starting VGG16 FP32 Training for Thesis Baseline...")
model = QuantizableVGG16(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# --- 5. 训练循环 ---
best_acc = 0.0
log_message(f"{'Epoch':<10}{'TrainAcc%':<15}{'TestAcc%':<15}{'LR':<15}")

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"VGG16 FP32 Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        _, pred = torch.max(outputs, 1); total += labels.size(0); correct += (pred == labels).sum().item()
    
    scheduler.step()
    model.eval(); t_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device); outputs = model(inputs)
            _, pred = torch.max(outputs, 1); t_correct += (pred == labels).sum().item()
    
    acc = 100. * t_correct / 10000
    log_message(f"{epoch+1:<10}{100.*correct/total:<15.2f}{acc:<15.2f}{scheduler.get_last_lr()[0]:.6f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(save_dir, "fp32_vgg16_best.pth"))
        log_message(f"--> New Best Accuracy: {best_acc:.2f}% Saved.")

log_message(f"VGG16 FP32 Training Finished. Global Best: {best_acc}%")