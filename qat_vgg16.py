import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 硬件与日志配置 ---
torch.backends.quantized.engine = 'qnnpack'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/root/autodl-tmp/my_backup"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, f"qat_vgg16_{datetime.datetime.now().strftime('%m%d_%H%M')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_path, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

# --- 2. 结构定义 (必须与 FP32 一致) ---
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizableVGG16, self).__init__()
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

# --- 3. QAT 准备逻辑 ---
log_message("Initializing VGG16 QAT Phase...")
model = QuantizableVGG16(num_classes=10)
ckpt = os.path.join(save_dir, "fp32_vgg16_best.pth")
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.to(device)

# 算子融合 (论文 2.2-3 关键步骤)
model.eval()
fused_list = []
for i in range(len(model.features)):
    if isinstance(model.features[i], nn.Conv2d) and i+1 < len(model.features) and isinstance(model.features[i+1], nn.ReLU):
        fused_list.append([f'features.{i}', f'features.{i+1}'])
torch.ao.quantization.fuse_modules(model, fused_list, inplace=True)
log_message(f"Fused {len(fused_list)} [Conv+ReLU] layers for quantization.")

# 配置 QAT 模拟环境
model.train()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 4. QAT 微调循环 ---
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])), batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])), batch_size=128)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
log_message(f"{'Epoch':<10}{'QAT_TrainAcc%':<20}{'QAT_TestAcc%':<20}")

for epoch in range(15):
    model.train()
    if epoch > 3: # 模拟量化误差锁定
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    correct, total = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"VGG16 QAT {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        _, pred = torch.max(outputs, 1); total += labels.size(0); correct += (pred == labels).sum().item()

    model.eval(); t_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device); outputs = model(inputs)
            _, pred = torch.max(outputs, 1); t_correct += (pred == labels).sum().item()
    
    acc = 100. * t_correct / 10000
    log_message(f"{epoch+1:<10}{100.*correct/total:<20.2f}{acc:<20.2f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(save_dir, "vgg16_qat_best.pth"))

# --- 5. 导出部署模型与存储分析 ---
log_message("Converting to INT8 for deployment...")
model.load_state_dict(torch.load(os.path.join(save_dir, "vgg16_qat_best.pth"), map_location='cpu'))
model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(model, inplace=False)
deploy_path = os.path.join(save_dir, "vgg16_int8_deploy.pt")
traced = torch.jit.trace(int8_model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced, deploy_path)

# 计算论文所需的模型大小指标
fp32_size = os.path.getsize(ckpt) / (1024*1024)
int8_size = os.path.getsize(deploy_path) / (1024*1024)
log_message("="*50)
log_message(f"FINAL REPORT for VGG16")
log_message(f"Best INT8 Accuracy: {best_acc}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT8 Model Size: {int8_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
log_message("="*50)