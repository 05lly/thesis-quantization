# qat_resnet18_fixed_v2.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on device: {device}")

# -------------------------------
# 数据集准备
# -------------------------------
data_root = '/root/autodl-tmp/thesis-quantization/data'  # CIFAR-10 本地路径
# 正确的预处理：Resize到224x224
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------------
# 模型定义与融合
# -------------------------------
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10
    return model

def fuse_model(model):
    """正确的ResNet18融合方式"""
    model.eval()
    # 融合每个BasicBlock中的卷积和BN
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                # 使用正确的融合方式
                tq.fuse_modules(basic_block, [['conv1', 'bn1', 'relu']], inplace=True)
                tq.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
    return model

# -------------------------------
# 初始化模型
# -------------------------------
model = create_model()
model = fuse_model(model)

# -------------------------------
# QAT准备 - 修复量化参数问题
# -------------------------------
# 使用自定义的QAT配置，确保zero_point在正确范围内
from torch.ao.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize

# 自定义fake quantize配置
my_qconfig = QConfig(
    activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                     quant_min=0,
                                     quant_max=255,
                                     dtype=torch.quint8,
                                     qscheme=torch.per_tensor_affine,
                                     reduce_range=False),
    weight=FakeQuantize.with_args(observer=PerChannelMinMaxObserver,
                                 quant_min=-128,
                                 quant_max=127,
                                 dtype=torch.qint8,
                                 qscheme=torch.per_channel_symmetric,
                                 reduce_range=False)
)

# 设置QAT配置
model.train()
model.qconfig = my_qconfig

# 准备QAT
tq.prepare_qat(model, inplace=True)
model.to(device)

# -------------------------------
# 优化器设置 - 使用更小的学习率
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# -------------------------------
# 训练函数 - 添加梯度裁剪
# -------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        loop.set_postfix({
            'loss': running_loss/total,
            'acc': 100.*correct/total
        })
    
    return running_loss/total, 100.*correct/total

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/total, 100.*correct/total

# -------------------------------
# 训练循环
# -------------------------------
best_acc = 0
num_epochs = 4

logger.info("Starting QAT training...")
for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = validate(model, test_loader)
    
    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
    logger.info(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        # 保存整个模型（包括QAT状态）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'qconfig': model.qconfig,
        }, "best_qat_model.pth")
        logger.info(f"New best model saved with accuracy {best_acc:.2f}%")
    
    scheduler.step()

# -------------------------------
# 转换为INT8
# -------------------------------
logger.info("Converting to INT8...")
model.eval()
model.to('cpu')

# 转换前确保模型处于正确的状态
model_int8 = tq.convert(model)

# 验证INT8模型
val_loss, val_acc = validate(model_int8, test_loader)
logger.info(f"INT8 Model - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
logger.info(f"Accuracy drop: {best_acc - val_acc:.2f}%")

# 保存INT8模型
torch.save({
    'model_state_dict': model_int8.state_dict(),
    'model_architecture': 'resnet18_qat_int8',
    'input_size': (3, 224, 224),
    'num_classes': 10,
    'accuracy': val_acc
}, "resnet18_int8_final.pth")

logger.info("Training and quantization completed!")

# -------------------------------
# 测试单个batch验证模型是否正常工作
# -------------------------------
def test_model_sanity():
    logger.info("Running sanity check...")
    model.eval()
    test_input = torch.randn(1, 3, 224, 224).to(device)
    try:
        with torch.no_grad():
            output = model(test_input)
        logger.info(f"Sanity check passed! Output shape: {output.shape}")
        return True
    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        return False

# 执行sanity check
test_model_sanity()