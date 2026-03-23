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
lr = 5e-5        # 针对 CIFAR-100 

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志函数 ---
log_filename = os.path.join(log_dir, f"qat_vgg16_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 3. 模型结构 (支持量化) ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(QuantizableVGG16, self).__init__()
        # 加载基础 VGG16 模型
        vgg = models.vgg16(weights=None) 
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        # 插入量化/反量化节点
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
        # 融合 Conv + ReLU 以提高量化精度和速度
        for m in self.modules():
            if type(m) == nn.Sequential:
                for i in range(len(m)):
                    if i + 1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

# --- 4. 数据处理 (CIFAR-100) ---
transform_qat = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=False, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 5. 模型加载与 QAT 准备 ---
log_message(f"Initializing Quantizable VGG16 for CIFAR-100 (Batch: {batch_size}, LR: {lr})...")
model = QuantizableVGG16(num_classes=100)

fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found.")
    exit()

# 加载 FP32 权重
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
log_message(f"FP32 Checkpoint Loaded: {fp32_path}")

# 融合与 QAT 准备
model.eval()
model.fuse_model() 
model.train() 
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# --- 6. 训练循环 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    # 冻结量化观察器（第 5 轮开始）
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
    # 冻结 BN 层参数
    if epoch > 2:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"VGG16 QAT Epoch {epoch+1}", leave=False):
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
        torch.save(model.state_dict(), os.path.join(model_dir, "vgg16_qat_best.pth"))
        log_message(f"--- Saved new best QAT Accuracy: {best_acc:.2f}% ---")

# --- 7. 导出最终模型 ---
log_message("Converting to INT8 and Exporting for Deployment...")
model.load_state_dict(torch.load(os.path.join(model_dir, "vgg16_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 保存权重文件
torch.save(int8_model.state_dict(), os.path.join(model_dir, "vgg16_int8_final.pth"))

# 保存部署用的脚本文件
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)
deploy_path = os.path.join(model_dir, "vgg16_int8_deploy.pt")
torch.jit.save(traced_model, deploy_path)

# --- 8. 总结 ---
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 55)
log_message(f"VGG16 QAT Report (CIFAR-100)")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT8 Deploy Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message(f"Execution Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)