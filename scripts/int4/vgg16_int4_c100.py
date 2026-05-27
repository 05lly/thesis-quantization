import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, time, datetime, random
import numpy as np
from tqdm import tqdm

# --- 1. 配置 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
lr = 1e-4
grad_clip = 1.0

model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"
data_root = "/root/autodl-tmp/data" if os.path.exists("/root/autodl-tmp") else "./data"
log_dir = "logs"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir,
    f"qat_int4_vgg16_c100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")


# --- 2. 模型定义：无 BN 的 VGG16 ---
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
            if isinstance(m, nn.Sequential):
                for i in range(len(m) - 1):
                    if isinstance(m[i], nn.Conv2d) and isinstance(m[i + 1], nn.ReLU):
                        torch.ao.quantization.fuse_modules(
                            m,
                            [str(i), str(i + 1)],
                            inplace=True,
                        )

                    if isinstance(m[i], nn.Linear) and isinstance(m[i + 1], nn.ReLU):
                        torch.ao.quantization.fuse_modules(
                            m,
                            [str(i), str(i + 1)],
                            inplace=True,
                        )


# --- 3. INT4 QAT 配置 ---
# activation: unsigned 4-bit, 0 ~ 15
# weight: signed 4-bit, -8 ~ 7, per-channel symmetric
def get_int4_qat_qconfig():
    return torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=15,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
            quant_min=-8,
            quant_max=7,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )


# --- 4. 数值稳定性检查 ---
def has_nonfinite_grad(model):
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True, name
    return False, None


def has_nonfinite_param(model):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            return True, name
    return False, None


# --- 5. 数据处理：train/test 必须分开 ---
# 训练集可以使用随机增强
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

# 测试集不要使用 RandomHorizontalFlip
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

train_dataset = datasets.CIFAR100(
    data_root,
    train=True,
    download=True,
    transform=transform_train,
)

test_dataset = datasets.CIFAR100(
    data_root,
    train=False,
    download=True,
    transform=transform_test,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True if device.type == "cuda" else False,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True if device.type == "cuda" else False,
)


# --- 6. 评估函数 ---
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return 100.0 * correct / total


# --- 7. 模型准备 ---
log_message("Initializing VGG16 for INT4 QAT on CIFAR-100...")

model = QuantizableVGG16(num_classes=100)

fp32_path = os.path.join(model_dir, "fp32_vgg16_c100_best.pth")
if not os.path.exists(fp32_path):
    raise FileNotFoundError(f"Missing FP32 checkpoint: {fp32_path}")

state_dict = torch.load(fp32_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

model.to(device)

# QAT 准备流程：eval -> fuse -> train -> qconfig -> prepare_qat
model.eval()
model.fuse_model()
model.train()

model.qconfig = get_int4_qat_qconfig()
torch.ao.quantization.prepare_qat(model, inplace=True)

model.to(device)

optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
)

criterion = nn.CrossEntropyLoss()

best_acc = 0.0
best_path = os.path.join(model_dir, "vgg16_int4_c100_best.pth")
start_time = time.time()
skipped_batches = 0

log_message("=" * 70)
log_message("Start INT4 QAT training")
log_message(f"Device      : {device}")
log_message(f"Epochs      : {epochs}")
log_message(f"Batch size  : {batch_size}")
log_message(f"LR          : {lr}")
log_message(f"Grad clip   : {grad_clip}")
log_message("=" * 70)


# --- 8. 训练循环 ---
for epoch in range(epochs):
    model.train()

    # 前几轮让 observer 统计 activation range
    # 第 9 轮开始冻结 observer，避免 scale/zero-point 后期剧烈波动
    if epoch > 7:
        model.apply(torch.ao.quantization.disable_observer)

        # 你的模型没有 BN，这句不是必要的。
        # 保留不会影响结果；如果环境报错，可以直接删掉。
        try:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        except Exception:
            pass

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1:02d}/{epochs}]",
        leave=False,
    )

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        nonfinite_param, bad_param_name = has_nonfinite_param(model)
        if nonfinite_param:
            raise RuntimeError(
                f"Non-finite parameter detected before forward: {bad_param_name}. "
                "Training has become numerically unstable. "
                "Try lr=5e-5 or grad_clip=0.5."
            )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            skipped_batches += 1
            log_message(
                f"Warning: non-finite loss at epoch {epoch + 1}; skipped batch."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        nonfinite_grad, bad_grad_name = has_nonfinite_grad(model)
        if nonfinite_grad:
            skipped_batches += 1
            log_message(
                f"Warning: non-finite gradient in {bad_grad_name} "
                f"at epoch {epoch + 1}; skipped batch."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        _, pred = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    scheduler.step()

    train_acc = 100.0 * correct / total if total > 0 else 0.0
    train_loss = running_loss / total if total > 0 else 0.0
    val_acc = evaluate(model, test_loader, device)
    current_lr = scheduler.get_last_lr()[0]

    log_message(
        f"Epoch [{epoch + 1:02d}/{epochs}] | "
        f"Train Acc: {train_acc:6.2f}% | "
        f"Val Acc: {val_acc:6.2f}% | "
        f"Loss: {train_loss:.4f} | "
        f"LR: {current_lr:.6f} | "
        f"Skipped: {skipped_batches}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        log_message(f"Saved best checkpoint: {best_path}")


# --- 9. 加载 best checkpoint，再做一次最终评估 ---
if os.path.exists(best_path):
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    final_acc = evaluate(model, test_loader, device)
else:
    final_acc = best_acc

fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)

log_message("=" * 70)
log_message("VGG16 INT4 QAT Final Report on CIFAR-100")
log_message(f"Best Val Accuracy      : {best_acc:.2f}%")
log_message(f"Final Re-eval Accuracy : {final_acc:.2f}%")
log_message(f"FP32 Model Size        : {fp32_size:.2f} MB")
log_message(f"Theory INT4 Size       : {fp32_size / 8:.2f} MB")
log_message(f"Theory Compression     : 8.00x")
log_message(f"Total Time Taken       : {(time.time() - start_time) / 60:.2f} mins")
log_message(f"Skipped Batches        : {skipped_batches}")
log_message("=" * 70)