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
epochs = 50   # ⭐ 提升
lr = 1e-4

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

#  核心 INT4 配置
def get_int4_qat_qconfig():
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.HistogramObserver,  # ⭐
            quant_min=0, quant_max=15,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.PerChannelMinMaxObserver,  # ⭐核心
            quant_min=-8, quant_max=7,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )

# --- 2. 日志 ---
log_filename = os.path.join(log_dir, f"best_int4_mobilenetv2_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Env: {device} | Mode: Improved INT4 QAT MobileNetV2")

# --- 3. 数据 ---
data_dir = '/root/autodl-tmp/data'

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2761)),
])

trainloader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 模型 ---
log_message("Initializing Improved INT4 QAT MobileNetV2...")

model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 100)

fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_c100_best.pth")
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)

model.eval()
model.fuse_model(is_qat=True)
model.train()

model.qconfig = get_int4_qat_qconfig()

# 优化：Selective Quantization
for name, module in model.named_modules():

    #  第一层 & 最后一层不量化
    if "features.0" in name or "classifier" in name:
        module.qconfig = None

    #depthwise conv 不量化（核心！！）
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels:  # depthwise
            module.qconfig = None

torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 5. 优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 6. 训练 ---
best_acc = 0.0
start_time = time.time()

log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'LR':<15}")

for epoch in range(epochs):
    model.train()

    # 延迟冻结
    if epoch > 20:
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    correct, total = 0, 0

    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
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

    # --- 测试 ---
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (pred == labels).sum().item()

    val_acc = 100. * test_correct / test_total
    train_acc = 100. * correct / total

    log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{scheduler.get_last_lr()[0]:<15.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(model_dir, "best_int4_mobilenetv2_c100.pth"))
    # --- 7. 总结 ---
def get_model_theory_size(model):
    fp32_params = 0
    int4_params = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            p_count = sum(p.numel() for p in m.parameters())
            if m.qconfig is None:
                fp32_params += p_count
            else:
                int4_params += p_count
    
    # 理论体积计算 
    size_mb = (fp32_params * 4 + int4_params * 0.5) / (1024 * 1024)
    return fp32_params, int4_params, size_mb

fp32_p, int4_p, theory_size = get_model_theory_size(model)
original_size = (os.path.getsize(fp32_path) / (1024 * 1024))

log_message("=" * 60)
log_message(" Improved MobileNetV2 INT4 QAT Final Report ")
log_message(f" Best Accuracy     : {best_acc:.2f}%")
log_message(f" FP32 Params Count : {fp32_p/1e6:.3f} M")
log_message(f" INT4 Params Count : {int4_p/1e6:.3f} M")
log_message(f" Original Size     : {original_size:.2f} MB")
log_message(f" Theory Mixed Size : {theory_size:.2f} MB") # ⭐ 这个才是对的
log_message(f" Compression Ratio : {original_size / theory_size:.2f} x")
log_message("=" * 60)
