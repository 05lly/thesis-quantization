import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime
from tqdm import tqdm

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 50
lr = 1e-4

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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

# --- 日志 ---
log_filename = os.path.join(log_dir, f"best_int4_resnet18_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Env: {device} | Dataset: CIFAR-100 | Mode: Strong INT4 QAT")

# --- 数据 ---
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

# --- 模型 ---
log_message("Initializing improved INT4 QAT ResNet18...")

model = models.quantization.resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 100)

fp32_path = os.path.join(model_dir, "fp32_resnet18_c100_best.pth")
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)

model.eval()
model.fuse_model(is_qat=True)
model.train()

model.qconfig = get_int4_qat_qconfig()

#  更合理的混合精度策略
model.conv1.qconfig = None
model.fc.qconfig = None

# 保护 layer1（低层特征）
for name, module in model.named_modules():
    if "layer1" in name:
        module.qconfig = None

torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# --- 训练 ---
best_acc = 0.0
start_time = time.time()
log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'LR':<15}")

for epoch in range(epochs):
    model.train()

    #  延迟冻结（关键）
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
                   os.path.join(model_dir, "best_int4_resnet18_c100.pth"))

# --- 总结 ---
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)

log_message("=" * 60)
log_message(f" Improved INT4 QAT Final Report ")
log_message(f" Best Accuracy : {best_acc:.2f}%")
log_message(f" Total Time    : {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 60)
#体积

def count_mixed_precision_size(model):
    fp32_params = 0
    int4_params = 0
    
    for name, module in model.named_modules():
        # 统计 Conv2d 和 Linear 的参数
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            num_params = sum(p.numel() for p in module.parameters())
            # 如果该层没量化（qconfig为None），归为FP32
            if getattr(module, 'qconfig', None) is None:
                fp32_params += num_params
            else:
                int4_params += num_params
                
    # 计算理论大小 (FP32: 4 bytes, INT4: 0.5 bytes)
    size_mb = (fp32_params * 4 + int4_params * 0.5) / (1024 * 1024)
    return fp32_params, int4_params, size_mb

# 在model 准备好后调用
fp_p, int_p, total_size = count_mixed_precision_size(model)
print(f"FP32 参数量: {fp_p/1e6:.2f} M")
print(f"INT4 参数量: {int_p/1e6:.2f} M")
print(f"混合精度理论体积: {total_size:.2f} MB")