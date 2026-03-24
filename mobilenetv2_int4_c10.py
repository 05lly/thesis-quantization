import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import brevitas.nn as qnn
import os, time, datetime
from tqdm import tqdm

# --- 配置 ---
MODEL_NAME = "MobileNetV2"
BIT_WIDTH = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40  
LR = 1e-4
BATCH_SIZE = 64
MODEL_DIR = "/root/autodl-tmp/my_backup"
LOG_DIR = "logs"
DATA_DIR = "/root/autodl-tmp/data"

os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"qat_int{BIT_WIDTH}_{MODEL_NAME.lower()}_c10.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"; print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f: f.write(full_msg + "\n")

# --- 模型构建 ---
class QuantMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantMobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = nn.Sequential()
        # 初始层
        self.features.append(nn.Sequential(
            qnn.QuantConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False, weight_bit_width=BIT_WIDTH),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True)
        ))
        # 中间层量化
        for i in range(1, len(base.features) - 1):
            block = base.features[i]
            for j, layer in enumerate(block.conv):
                if isinstance(layer, nn.Conv2d):
                    block.conv[j] = qnn.QuantConv2d(layer.in_channels, layer.out_channels, layer.kernel_size, 
                                                    stride=layer.stride, padding=layer.padding, groups=layer.groups, 
                                                    bias=False, weight_bit_width=BIT_WIDTH)
            self.features.append(block)
        # 结尾层
        last_conv = base.features[-1][0]
        self.features.append(nn.Sequential(
            qnn.QuantConv2d(last_conv.in_channels, last_conv.out_channels, kernel_size=1, bias=False, weight_bit_width=BIT_WIDTH),
            nn.BatchNorm2d(last_conv.out_channels), nn.ReLU6(inplace=True)
        ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2), qnn.QuantLinear(base.last_channel, num_classes, bias=True, weight_bit_width=BIT_WIDTH))

    def forward(self, x):
        x = self.features(x); x = self.avgpool(x)
        x = torch.flatten(x, 1); x = self.classifier(x)
        return x

# --- 数据与训练逻辑 (复用上述逻辑) ---
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(DATA_DIR, train=True, download=True, 
    transform=transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.Compose([transforms.ToTensor(), norm])),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

log_message(f"Env: {DEVICE} | Task: CIFAR-10 INT{BIT_WIDTH} QAT | Model: {MODEL_NAME}")
model = QuantMobileNetV2().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_acc, start_time = 0.0, time.time()
for epoch in range(EPOCHS):
    model.train(); train_correct, train_total = 0, 0
    for img, lbl in tqdm(train_loader, desc=f"{MODEL_NAME} INT4 Epoch {epoch+1}", leave=False):
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad(); output = model(img)
        loss = criterion(output, lbl); loss.backward(); optimizer.step()
        train_total += lbl.size(0); train_correct += (output.max(1)[1] == lbl).sum().item()
    
    model.eval(); corr = 0
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            corr += (model(img).max(1)[1] == lbl).sum().item()
    acc = 100. * corr / len(test_loader.dataset); train_acc = 100. * train_correct / train_total
    if acc > best_acc: 
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{MODEL_NAME.lower()}_int4_best.pth"))
    log_message(f"Epoch {epoch+1:<3} | TrainAcc: {train_acc:.2f}% | TestAcc: {acc:.2f}%")

log_message("=" * 55)
log_message(f"QAT Summary Report (INT{BIT_WIDTH})")
log_message(f"Best Test Accuracy: {best_acc:.2f}%")
log_message(f"Theoretical INT4 Size: {(9.21 * BIT_WIDTH/32):.2f} MB")
log_message(f"Execution Time: {(time.time()-start_time)/60:.2f} mins")
log_message("=" * 55)