import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import datetime
from tqdm import tqdm

# 1. 参数与环境配置 (严格锁定 QNNPACK)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局强制使用 qnnpack (ARM设备专属)
torch.backends.quantized.engine = 'qnnpack'

batch_size = 128  
epochs = 15
lr = 1e-4  

# 目录配置 
model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 2. 日志记录
log_filename = os.path.join(log_dir, f"VGG16_CIFAR10_QAT_Training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message("=" * 60)
log_message("Starting VGG16 QAT (Quantization Aware Training) for CIFAR-10")
log_message(f"Device: {device} | Quant Engine: {torch.backends.quantized.engine}")
log_message("=" * 60)

# 3. 定义可量化的 VGG16 结构
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizableVGG16, self).__init__()
        # 加载空壳 VGG16 (稍后会加载训练好的 FP32 权重)
        vgg = models.vgg16(weights=None) 
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        
        # 量化/反量化桩
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

    def fuse_model(self):
        """融合相邻的层以提升量化推理性能"""
        log_message("Fusing Conv2d+ReLU and Linear+ReLU layers...")
        for m in self.modules():
            if type(m) == nn.Sequential:
                for i in range(len(m)):
                    # 融合 Conv2d + ReLU
                    if i + 1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)
                    # 融合 Linear + ReLU (VGG的分类器中也有ReLU)
                    elif i + 1 < len(m) and type(m[i]) == nn.Linear and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

# 4. 数据管道 (对齐 FP32 的处理)

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

# 5. 加载 FP32 模型并配置 QAT
model = QuantizableVGG16(num_classes=10)

# 确保路径指向你训练出的最优 FP32 模型
fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"CRITICAL ERROR: FP32 model not found at {fp32_path}")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
log_message(f"Successfully loaded FP32 checkpoint: {fp32_path}")

# --- QAT 准备步骤 ---
model.eval()
model.fuse_model() # 1. 融合

model.train() # 2. 必须切换回 train 模式插入 FakeQuant
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True) # 3. 准备QAT
log_message("QAT model prepared with FakeQuant nodes.")

model.to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 6. QAT 训练循环
best_acc = 0.0
start_time = time.time()
log_message("-" * 60)
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    
    # 在最后几个 epoch 冻结量化参数（稳定 Scale 和 Zero Point）
    if epoch > epochs - 5: 
        model.apply(torch.ao.quantization.disable_observer)
    if epoch > epochs - 4:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{epochs}", leave=False):
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

    # 验证模拟量化精度 (FakeQuant 仍在工作)
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()
    
    val_acc = 100. * test_correct / 10000
    train_acc = 100. * correct / total
    epoch_loss = running_loss / 50000
    
    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{epoch_loss:<15.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_qat_path = os.path.join(model_dir, "vgg16_qat_best_weights.pth")
        torch.save(model.state_dict(), best_qat_path)
        log_message(f"   -> New Best QAT Accuracy: {best_acc:.2f}% (Saved weights)")

# 7. 安全导出 INT8 部署模型 (The Critical Fix)
log_message("-" * 60)
log_message("Starting INT8 Conversion for Raspberry Pi (qnnpack)...")

# 重新加载最好的空壳模型
export_model = QuantizableVGG16(num_classes=10)
export_model.eval()
export_model.fuse_model()
export_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(export_model, inplace=True)

# 加载训练好的 FakeQuant 权重
export_model.load_state_dict(torch.load(best_qat_path, map_location='cpu'))

# 必须在 CPU 环境下执行 Convert
export_model.to('cpu').eval()

# 执行真实的 INT8 转换 (剥离 FakeQuant，变成真正的 int8 算子)
int8_model = torch.ao.quantization.convert(export_model, inplace=False)

# 保存 JIT 追踪模型 (给树莓派调用的最终文件)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)
deploy_path = os.path.join(model_dir, "vgg16_cifar10_int8_deploy.pt")
torch.jit.save(traced_model, deploy_path)

log_message(f"Deployable TorchScript model saved to: {deploy_path}")

# 8. 总结报告
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 60)
log_message("VGG16 CIFAR-10 QAT Process Completed")
log_message(f"Final Best QAT Accuracy : {best_acc:.2f}%")
log_message(f"FP32 Checkpoint Size  : {fp32_size:.2f} MB")
log_message(f"INT8 Deploy Model Size: {int8_size:.2f} MB")
if int8_size > 0:
    log_message(f"Compression Ratio       : {fp32_size/int8_size:.2f}x")
log_message("=" * 60)