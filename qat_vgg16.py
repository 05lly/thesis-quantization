import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 参数配置 (对齐 ResNet18 QAT 逻辑) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = 'qnnpack'
batch_size, epochs, lr = 128, 15, 1e-4

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

# --- 2. 统一日志函数 ---
log_filename = os.path.join(log_dir, f"qat_vgg16_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"; print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

# --- 3. 数据处理 (224 保持一致) ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, transform=transform), batch_size=batch_size, shuffle=True)

# --- 4. 模型加载与 QAT 准备 ---
model = models.quantization.vgg16(weights=None, quantize=False)
model.classifier[6] = nn.Linear(4096, 10)

fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
log_message(f"VGG16 FP32 Checkpoint Loaded: {fp32_path}")

# 融合算子与 QAT 初始化
model.eval()
model.fuse_model() # VGG16 的融合逻辑
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# --- 5. QAT 训练循环 (15 Epochs) ---
best_acc = 0.0
start_time = time.time()

for epoch in range(epochs):
    model.train()
    if epoch > 3: # 第 5 轮开始冻结 BN 和 Observer
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    correct, total = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"VGG16 QAT Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        _, pred = torch.max(outputs, 1); total += labels.size(0); correct += (pred == labels).sum().item()

    # 验证模拟量化精度
    model.eval(); test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device); outputs = model(inputs)
            _, pred = torch.max(outputs, 1); test_correct += (pred == labels).sum().item()
    
    val_acc = 100. * test_correct / 10000
    log_message(f"QAT Epoch {epoch+1:<8}{100.*correct/total:<15.2f}{val_acc:<15.2f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "vgg16_qat_best.pth"))

# --- 6. 最终转换与部署导出 ---
log_message("Exporting VGG16 INT8 for Raspberry Pi 5...")
model.load_state_dict(torch.load(os.path.join(model_dir, "vgg16_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

traced_model = torch.jit.trace(int8_model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, os.path.join(model_dir, "vgg16_int8_deploy.pt"))

# --- 7. 实验总结报表 ---
def get_size(path): return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
fp32_s, int8_s = get_size(fp32_path), get_size(os.path.join(model_dir, "vgg16_int8_deploy.pt"))

log_message("=" * 55)
log_message(f"VGG16 QAT Summary")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Size: {fp32_s:.2f} MB | INT8 Size: {int8_s:.2f} MB")
log_message(f"Compression: {fp32_s/int8_s:.2f}x")
log_message("=" * 55)