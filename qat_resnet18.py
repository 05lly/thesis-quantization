import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os
from tqdm import tqdm

# ----------------------------
# 日志配置
# ----------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/qat_training_log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ----------------------------
# 超参数
# ----------------------------
batch_size = 128
epochs = 15
learning_rate = 1e-3
num_classes = 10  # 根据你数据集修改
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Training on device: {device}")

# ----------------------------
# 数据集与预处理
# ----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ----------------------------
# 构建模型
# ----------------------------
model_fp32 = models.resnet18(pretrained=True)
model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)

# ----------------------------
# 模块融合 (ResNet18 特定)
# ----------------------------
def fuse_model(model):
    # 对BasicBlock做conv+bn+relu融合
    for module_name, module in model.named_children():
        if module_name == "layer1" or module_name == "layer2" or module_name == "layer3" or module_name == "layer4":
            for block_name, block in module.named_children():
                tq.fuse_modules(block, ['conv1', 'bn1', 'relu'], inplace=True)
                tq.fuse_modules(block, ['conv2', 'bn2'], inplace=True)
    return model

model_fp32 = fuse_model(model_fp32)

# ----------------------------
# QAT 配置 (FX模式 + 默认 fbgemm)
# ----------------------------
example_input = torch.randn(1,3,224,224)
qconfig_mapping = tq.get_default_qat_qconfig_mapping('fbgemm')
model_fp32.qconfig = tq.QConfig(
    activation=tq.default_observer,
    weight=tq.default_weight_observer
)
model_fp32 = quantize_fx.prepare_qat_fx(model_fp32, qconfig_mapping, example_inputs=example_input)
model_fp32 = model_fp32.to(device)

# ----------------------------
# 损失函数、优化器、调度器
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fp32.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ----------------------------
# 训练函数
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    logging.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Top1 Accuracy={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

# ----------------------------
# 测试函数
# ----------------------------
def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.* correct/total
    loss = running_loss/total
    return loss, acc

# ----------------------------
# 主训练流程
# ----------------------------
best_acc = 0
patience_counter = 0
early_stop_patience = 5

for epoch in range(1, epochs+1):
    train_loss, train_acc = train_one_epoch(model_fp32, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = evaluate(model_fp32, test_loader, criterion)
    logging.info(f"Validation: Loss={test_loss:.4f}, Top1 Accuracy={test_acc:.2f}%")
    
    scheduler.step(test_acc)  # 调整学习率
    
    # 早停 + 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        torch.save(model_fp32.state_dict(), "qat_resnet18_best.pth")
        logging.info(f"New best model saved with accuracy {best_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break

# ----------------------------
# 加载最佳模型并转换为 INT8
# ----------------------------
model_fp32.load_state_dict(torch.load("qat_resnet18_best.pth"))
model_fp32.eval()
model_fp32.to('cpu')
model_int8 = tq.convert(model_fp32)

# ----------------------------
# INT8 模型评估
# ----------------------------
int8_loss, int8_acc = evaluate(model_int8, test_loader, criterion)
logging.info(f"INT8 Model Accuracy: {int8_acc:.2f}%")
logging.info(f"Accuracy Drop: {best_acc - int8_acc:.2f}%")

# ----------------------------
# 保存完整 INT8 模型配置
# ----------------------------
torch.save({
    'model_state_dict': model_int8.state_dict(),
    'model_architecture': 'resnet18',
    'input_size': (3,224,224),
    'num_classes': num_classes,
    'qat_config': 'fbgemm'
}, "qat_resnet18_complete.pth")