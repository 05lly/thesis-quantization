import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os, time, datetime
from tqdm import tqdm

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
lr = 1e-4

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 强 INT4 QConfig ---
def get_int4_qat_qconfig():
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.HistogramObserver,  
            quant_min=0, quant_max=15,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.PerChannelMinMaxObserver,  
            quant_min=-8, quant_max=7,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )

# --- 3. 日志 ---
log_file = os.path.join(log_dir, f"int4_mobilenetv2_opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"[{t}] {msg}"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

log(f"Device: {device}")

# --- 4. 数据 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. 模型 ---
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)

model.eval()
model.fuse_model(is_qat=True)
model.train()

model.qconfig = get_int4_qat_qconfig()

# 核心：结构感知混合精度
for name, module in model.named_modules():

    # 首尾层保护
    if "features.0" in name or "classifier" in name:
        module.qconfig = None

    # depthwise conv 不量化
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels:
            module.qconfig = None

torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 6. 优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 7. 训练 ---
best_acc = 0.0
start = time.time()

log(f"{'Epoch':<10}{'TrainAcc':<15}{'ValAcc':<15}{'LR':<10}")

for epoch in range(epochs):

    model.train()

    # 延迟冻结
    if epoch > 10:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    correct, total = 0, 0

    for x, y in tqdm(train_loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    scheduler.step()

    # --- eval ---
    model.eval()
    test_correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            test_correct += pred.eq(y).sum().item()

    train_acc = 100. * correct / total
    val_acc = 100. * test_correct / len(test_loader.dataset)

    log(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{scheduler.get_last_lr()[0]:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(model_dir, "best_int4_mobilenetv2_opt.pth"))

# --- 8. 报告 ---
fp32_size = os.path.getsize(fp32_path)/(1024*1024)
def calc_model_size(model):
    fp32_params = 0
    int4_params = 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            num = sum(p.numel() for p in m.parameters())

            # 没有 qconfig → FP32
            if getattr(m, "qconfig", None) is None:
                fp32_params += num
            else:
                int4_params += num

    size_mb = (fp32_params * 4 + int4_params * 0.5) / (1024*1024)

    return fp32_params, int4_params, size_mb


fp32_p, int4_p, real_size = calc_model_size(model)

log("="*50)
log(" FINAL REPORT ")
log(f"Best Accuracy : {best_acc:.2f}%")
log(f"FP32 Size     : {fp32_size:.2f} MB")
log(f"Mixed Size  : {real_size:.2f} MB")
log(f"Compression : {fp32_size / real_size:.2f}x")
log(f"Time          : {(time.time()-start)/60:.2f} mins")
log("="*50)