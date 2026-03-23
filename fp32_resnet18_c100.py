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

# --- 2. 日志函数 ---
log_filename = os.path.join(log_dir, f"fp32_resnet18_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Env: {device} | Dataset: CIFAR-100 | Mode: Pure FP32 Baseline")

# --- 3. 数据处理  ---
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


# --- 4. 模型加载  ---
log_message("Loading Pure FP32 ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 100)
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
    for inputs, labels in tqdm(trainloader, desc=f"FP32 Epoch {epoch+1}", leave=False):
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
    train_acc = 100. * correct / total
    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{scheduler.get_last_lr()[0]:<15.6f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "fp32_resnet18_c100_best.pth"))

# --- 6. 实验总结 ---
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    for _ in range(50): _ = model(dummy_input) # 预热
    start_lat = time.time()
    for _ in range(100): _ = model(dummy_input)
    avg_latency = (time.time() - start_lat) / 100 * 1000 # ms

log_message("=" * 55)
log_message("FP32 Baseline Training Finished")
log_message(f"Best CIFAR-100 Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Inference Latency: {avg_latency:.2f} ms/image")
log_message(f"Total Training Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)