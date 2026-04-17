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

# --- 2. 日志系统 ---
log_filename = os.path.join(log_dir, f"qat_vgg16_c10_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{t}] {msg}"
    print(msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log_message(f"Environment: {device} | Batch Size: {batch_size} | Epochs: {epochs} | Engine: qnnpack")

# --- 3. 模型定义 ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)

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
        # 针对 Features 部分：自动循环融合 Conv+ReLU
        for i in range(len(self.features) - 1):
            if isinstance(self.features[i], nn.Conv2d) and isinstance(self.features[i+1], nn.ReLU):
                torch.ao.quantization.fuse_modules(self.features, [str(i), str(i+1)], inplace=True)
        
        # 针对 Classifier 部分：手动精确融合 Linear+ReLU (避开 Dropout 干扰)
        # VGG16 Classifier 结构: 0:Linear, 1:ReLU, 2:Dropout, 3:Linear, 4:ReLU, 5:Dropout, 6:Linear
        torch.ao.quantization.fuse_modules(self.classifier, ['0', '1'], inplace=True)
        torch.ao.quantization.fuse_modules(self.classifier, ['3', '4'], inplace=True)
        log_message("VGG16 Fusion: Features (auto) & Classifier (manual) complete.")

# --- 4. 数据处理 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. QAT 准备 ---
model = QuantizableVGG16(num_classes=10)
fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")

if not os.path.exists(fp32_path):
    log_message(f"Error: {fp32_path} not found.")
    exit()

model.load_state_dict(torch.load(fp32_path, map_location='cpu'))
model.to(device)
log_message(f"FP32 Checkpoint Loaded: {fp32_path}")

model.eval()
model.fuse_model()

model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# --- 6. QAT 训练 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}")

for epoch in range(epochs):
    model.train()
    if epoch > 3:
        model.apply(torch.ao.quantization.disable_observer)
        # VGG16无BN层，故不使用freeze_bn_stats

    correct, total, loss_sum = 0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        #inputs, labels = inputs.to(device), labels.size(0)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_acc = 100. * correct / total
    epoch_loss = loss_sum / len(train_loader.dataset)

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()

    val_acc = 100. * test_correct / len(test_loader.dataset)
    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{epoch_loss:<15.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_qat_path = os.path.join(model_dir, "vgg16_c10_qat_best.pth")
        torch.save(model.state_dict(), best_qat_path)
        log_message(f"New Best Accuracy: {best_acc:.2f}%")

#INT8 转换与导出 
log_message("Converting QAT model to INT8 and validating...")
model.load_state_dict(torch.load(best_qat_path, map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)

# 在导出前做一次CPU验证 证明 INT8 转换成功且精度达标
test_correct_int8 = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # INT8模型在CPU上跑
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')
        outputs = int8_model(inputs)
        _, pred = torch.max(outputs, 1)
        test_correct_int8 += (pred == labels).sum().item()
real_int8_acc = 100. * test_correct_int8 / len(test_loader.dataset)
log_message(f"Real INT8 Deploy Accuracy (CPU): {real_int8_acc:.2f}%")

# 导出TorchScript
deploy_path = os.path.join(model_dir, "vgg16_c10_int8_deploy.pt")
traced_model = torch.jit.trace(int8_model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, deploy_path)

# 总结
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)
compression = fp32_size / int8_size if int8_size > 0 else 0

log_message("=" * 55)
log_message(f"QAT Simulated Accuracy: {best_acc:.2f}%")
log_message(f"Real INT8 Accuracy (CPU): {real_int8_acc:.2f}%")
log_message(f"Accuracy Drop after Convert: {best_acc - real_int8_acc:.2f}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT8 Deploy Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {compression:.2f}x")
log_message(f"Total Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)