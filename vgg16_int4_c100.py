import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, epochs, lr = 128, 30, 1e-4
model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# VGG16 需要手动插入 QuantStub
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)
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
        for m in self.modules():
            if type(m) == nn.Sequential:
                for i in range(len(m)):
                    if i+1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)
                    if i+1 < len(m) and type(m[i]) == nn.Linear and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

def get_int4_qat_qconfig():
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=15, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            quant_min=-8, quant_max=7, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )

log_filename = os.path.join(log_dir, f"qat_int4_vgg16_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

# --- 2. 数据处理 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('/root/autodl-tmp/data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('/root/autodl-tmp/data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False, num_workers=4)

# --- 3. 模型准备 ---
log_message("Initializing VGG16 for INT4 QAT (CIFAR-100)...")
model = QuantizableVGG16(num_classes=100).to(device)
fp32_path = os.path.join(model_dir, "fp32_vgg16_c100_best.pth")
if not os.path.exists(fp32_path): raise FileNotFoundError(f"Missing {fp32_path}")
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))

model.eval()
model.fuse_model()
model.train() 
model.qconfig = get_int4_qat_qconfig()
torch.ao.quantization.prepare_qat(model, inplace=True)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# --- 4. 训练循环 ---
best_acc, start_time = 0.0, time.time()
for epoch in range(epochs):
    model.train()
    if epoch > 7:
        model.apply(torch.ao.quantization.disable_observer)
        # VGG16 无 BN 层，但保留此逻辑以兼容代码结构
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    correct, total = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}]", leave=False):
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
    t_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            t_correct += (pred == labels).sum().item()
    
    val_acc = 100. * t_correct / len(test_loader.dataset)
    log_message(f"Epoch [{epoch+1:02d}/{epochs}] | Train Acc: {100.*correct/total:5.2f}% | Val Acc: {val_acc:5.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_dir, "vgg16_int4_c100_best.pth"))

fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
log_message("=" * 60)
log_message(f" VGG16 INT4 QAT Final Report ")
log_message(f" Best Val Accuracy : {best_acc:.2f}%")
log_message(f" Theory INT4 Size  : {fp32_size / 8:.2f} MB")
log_message(f" Total Time Taken  : {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 60)