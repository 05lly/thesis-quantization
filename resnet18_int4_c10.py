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
epochs = 15
lr = 1e-4

if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"

log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 自定义 INT4 QConfig (核心：模拟 4-bit 截断) ---
def get_int4_qat_qconfig():
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=15, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )

log_filename = os.path.join(log_dir, f"qat_int4_resnet18_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 3. 数据处理 ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 模型加载与 QAT 准备 ---
log_message("Loading ResNet18 for INT4 QAT...")
model = models.quantization.resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 10)

fp32_path = os.path.join(model_dir, "fp32_resnet18_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"❌ Error: {fp32_path} not found!")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu'))
model.to(device)

# 算子融合与 QAT 插入
model.eval()
model.fuse_model(is_qat=True)
model.qconfig = get_int4_qat_qconfig()
torch.ao.quantization.prepare_qat(model, inplace=True)
model.train()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. 训练循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    if epoch > 3: # 冻持 BN 和观察器
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(trainloader, desc=f"INT4 QAT ResNet18 Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0); correct += (pred == labels).sum().item()

    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            t_total += labels.size(0); t_correct += (pred == labels).sum().item()
    
    val_acc = 100. * t_correct / t_total
    log_message(f"{epoch+1:<10}{100.*correct/total:<15.2f}{val_acc:<15.2f}{running_loss/total:<15.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "resnet18_int4_best.pth"))

# --- 6. 实验报表 ---
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
theory_int4_size = fp32_size / 8

log_message("=" * 55)
log_message(f"ResNet18 INT4 QAT Report")
log_message(f"Best Simulated Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"Theory INT4 Size: {theory_int4_size:.2f} MB")
log_message(f"Compression Ratio: 8.00x")
log_message(f"Execution Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)