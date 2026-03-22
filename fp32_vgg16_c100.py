import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import datetime
from tqdm import tqdm

# --- 1. 参数配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
lr = 0.01

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 统一日志函数 ---
log_filename = os.path.join(log_dir, f"fp32_vgg16_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Env: {device} | Dataset: CIFAR-100 | Mode: FP32-Baseline VGG16")

# --- 3. 模型定义 ---
class PureVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(PureVGG16, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- 4. 数据处理 ---

data_dir = '/root/autodl-tmp/data' 

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

trainloader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform_train),
    batch_size=batch_size, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform_test),
    batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. 训练 ---
model = PureVGG16(num_classes=100).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'LR':<15}")

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for inputs, labels in tqdm(trainloader, desc=f"FP32 VGG Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    scheduler.step()
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (pred == labels).sum().item()
    
    val_acc = 100. * test_correct / test_total
    log_message(f"{epoch+1:<10}{100.*correct/total:<15.2f}{val_acc:<15.2f}{scheduler.get_last_lr()[0]:<15.6f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "fp32_vgg16_c100_best.pth"))

# --- 6. 总结 ---
model.eval()
dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    for _ in range(50): _ = model(dummy)
    st = time.time()
    for _ in range(100): _ = model(dummy)
    lat = (time.time() - st) / 100 * 1000

log_message("=" * 55)
log_message("FP32 Baseline Training Finished")
log_message(f"Best CIFAR-100 Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Inference Latency: {lat:.2f} ms/image")
log_message(f"Total Training Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)