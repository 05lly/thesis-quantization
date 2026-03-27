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
lr = 5e-5  # 针对 CIFAR-100 微调略微降低

model_dir = "/root/autodl-tmp/my_backup"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志函数 ---
log_filename = os.path.join(log_dir, f"qat_resnet18_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Environment: {device} | Task: CIFAR-100 QAT | Batch Size: {batch_size}")

# --- 3. 数据处理 ---
data_dir = '/root/autodl-tmp/data' 
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 模型加载与 QAT 准备 ---
model = models.quantization.resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 100) 

fp32_path = os.path.join(model_dir, "fp32_resnet18_c100_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found.")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu'))
model.to(device)
log_message(f"FP32 Checkpoint Loaded: {fp32_path}")

model.eval()
model.fuse_model(is_qat=True)
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 训练循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()exi
    if epoch > 3: # 第 5 轮起冻结
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

# --- 6. 转换与导出 ---
log_message("Converting to INT8 Trace Format...")
model.load_state_dict(torch.load(os.path.join(model_dir, "resnet18_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

deploy_path = os.path.join(model_dir, "resnet18_int8_deploy.pt")
traced_model = torch.jit.trace(int8_model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, deploy_path)
# --- 6.5 真实 INT8 精度校验 ---
log_message("Starting Real INT8 Model Evaluation on CPU...")
int8_model.eval()
test_correct_int8 = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing REAL INT8", leave=False):
        # INT8 模型在 CPU 上运行效率最高
        inputs, labels = inputs.to('cpu'), labels.to('cpu') 
        outputs = int8_model(inputs)
        _, pred = torch.max(outputs, 1)
        test_correct_int8 += (pred == labels).sum().item()

real_int8_acc = 100. * test_correct_int8 / len(test_loader.dataset)
log_message(f"Real INT8 Accuracy after conversion: {real_int8_acc:.2f}%")

# --- 7. 总结 ---
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

log_message("=" * 55)
log_message("QAT Summary Report")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"REAL INT8 Accuracy (CPU): {real_int8_acc:.2f}%")  
log_message(f"Accuracy Drop: {best_acc - real_int8_acc:.2f}%") 
log_message(f"INT8 Model Size: {get_size_mb(deploy_path):.2f} MB")
log_message(f"Execution Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)