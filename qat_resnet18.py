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
torch.backends.quantized.engine = 'qnnpack' 
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

# --- 2. 日志函数  ---
log_filename = os.path.join(log_dir, f"qat_resnet18_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Environment: {device} | Batch Size: {batch_size} | Epochs: {epochs} | Engine: qnnpack")

# --- 3. 数据处理 (统一 224 分辨率与归一化参数) ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 模型加载与 QAT 准备 ---
# 必须使用 quantization 专用模型定义以确保结构兼容
model = models.quantization.resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 10)

# 自动读取之前的 FP32 权重
fp32_path = os.path.join(model_dir, "fp32_resnet18_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found.")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
log_message(f"FP32 Checkpoint Loaded: {fp32_path}")

# 融合算子
model.eval()
model.fuse_model(is_qat=True)
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 训练循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    #第 5 轮 (epoch index 4) 冻结
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"QAT Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    # 验证模拟量化精度
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()
    
    val_acc = 100. * test_correct / len(test_loader.dataset)
    train_acc = 100. * correct / total
    epoch_loss = running_loss / len(train_loader.dataset)
    
    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{epoch_loss:<15.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "resnet18_qat_best.pth"))
        log_message(f"New Best Accuracy: {best_acc:.2f}%")

# --- 6. 最终转换与部署导出 ---
log_message("Converting to INT8 Trace Format...")
model.load_state_dict(torch.load(os.path.join(model_dir, "resnet18_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 导出部署包 
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)

weights_path = os.path.join(model_dir, "resnet18_int8_final.pth")
deploy_path = os.path.join(model_dir, "resnet18_int8_deploy.pt")

torch.save(int8_model.state_dict(), weights_path)
torch.jit.save(traced_model, deploy_path)

# --- 7. 报表 ---
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 55)
log_message("QAT Summary Report")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT8 Model Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message(f"Execution Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)
