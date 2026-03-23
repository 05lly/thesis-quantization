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
epochs = 20         
lr = 5e-5             

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志函数 ---
log_filename = os.path.join(log_dir, f"qat_vgg16_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 3. 模型结构 ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(QuantizableVGG16, self).__init__()
        vgg = models.vgg16(weights=None) 
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        # 量化桩
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
        log_message("Merging Conv and ReLU layers...")
        for m in self.modules():
            if type(m) == nn.Sequential:
                for i in range(len(m)):
                    if i + 1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

# --- 4. 数据处理 (CIFAR-100) ---
data_dir = '/root/autodl-tmp/data' 
norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
transform_qat = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),           
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    norm
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    norm
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('/root/autodl-tmp/data', train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('/root/autodl-tmp/data', train=False, download=True, transform=transform_test),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 5. 模型加载与 QAT 准备 ---
log_message(f"Initializing QAT Mode... Target Device: {device}")
model = QuantizableVGG16(num_classes=100)

# 加载之前跑完的优化版 FP32 权重 
fp32_path = os.path.join(model_dir, "new_fp32_vgg16_c100_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found. Ensure FP32 training is finished.")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu'))
model.to(device)
log_message(f"Successfully Loaded FP32 Checkpoint: {fp32_path}")

# QAT 准备流程
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
best_qat_ckpt = os.path.join(model_dir, "new_vgg16_qat_optimized_best.pth")

log_message(f"Starting QAT Training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    if epoch > 5:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    if epoch > 15:
        model.apply(torch.ao.quantization.disable_observer)
    
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

    # 验证环节
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
    log_message(f"Epoch {epoch+1:02d} | TrainAcc: {train_acc:.2f}% | TestAcc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_qat_ckpt)
        log_message(f"--- Saved New Best QAT: {best_acc:.2f}% ---")

# --- 7. 导出最终量化模型  ---
log_message("Final Step: Converting FakeQuant to Real INT8 with JIT Freeze...")
model.load_state_dict(torch.load(best_qat_ckpt, map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 使用 JIT Trace 和 Freeze 优化推理
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)
frozen_model = torch.jit.freeze(traced_model)

deploy_path = os.path.join(model_dir, "new_vgg16_int8_final_deploy.pt")
torch.jit.save(frozen_model, deploy_path)

# --- 8. 总结 ---
total_mins = (time.time() - start_time) / 60

def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

log_message("=" * 55)
log_message(f"VGG16 QAT Final Report")
log_message(f"Best Accuracy: {best_acc:.2f}%")
log_message(f"Total Time: {total_mins:.2f} mins")
log_message(f"FP32 Size: {fp32_size:.2f} MB | INT8 Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message(f"Deployment Model: {deploy_path}")
log_message("=" * 55)