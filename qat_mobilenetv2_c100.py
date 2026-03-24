import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os
import time
import datetime
from tqdm import tqdm

# --- 1. 参数配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = 'qnnpack'  # 适配树莓派 5 
batch_size = 128  # 保持与 ResNet18 一致
epochs = 15
lr = 5e-5         # 针对 CIFAR-100 调低学习率

if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"

log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志函数 ---
log_filename = os.path.join(log_dir, f"qat_mobilenetv2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Environment: {device} | Task: CIFAR-100 QAT | Batch Size: {batch_size} | LR: {lr}")

# --- 3. 数据处理  ---
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

# --- 4. 模型准备与权重加载 ---
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 100) # 类别修改为 100

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_c100_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found! Please run CIFAR-100 FP32 training first.")
    exit()

# 加载 FP32 权重
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
log_message(f"Checkpoint loaded: {fp32_path}")

# QAT 准备：融合与插入伪量化节点
model.eval()
model.fuse_model(is_qat=True)
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# 使用 SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 微调循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    # 第 5 轮开始冻结 BN 和观察器以稳定量化参数
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"MNv2 QAT Epoch {epoch+1}", leave=False):
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

    # 验证精度
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
        torch.save(model.state_dict(), os.path.join(model_dir, "mobilenetv2_qat_best.pth"))
        log_message(f"--- Saved best model: {best_acc:.2f}% ---")

# --- 6. 最终转换与模型保存 ---
log_message("Converting QAT model to deployed INT8 format...")
model.load_state_dict(torch.load(os.path.join(model_dir, "mobilenetv2_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()

# 物理转换 (FP32 -> INT8)
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 导出 TorchScript
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)

deploy_path = os.path.join(model_dir, "mobilenetv2_int8_deploy.pt")
torch.jit.save(traced_model, deploy_path)
# --- 6.5 真实 INT8 精度校验  ---
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

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 55)
log_message("QAT Summary Report (CIFAR-100)")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"REAL INT8 Accuracy (CPU): {real_int8_acc:.2f}%")  # 引用 6.5 得到的值
log_message(f"Accuracy Drop: {best_acc - real_int8_acc:.2f}%") # 查看掉点情况
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT8 Deploy Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message(f"Total Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)