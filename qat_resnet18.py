# train_resnet18_full.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import logging
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Log file: {log_filename}")

# 记录开始时间
start_time = time.time()

# -------------------------------
# 数据集准备
# -------------------------------
data_root = '/root/autodl-tmp/thesis-quantization/data'
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# -------------------------------
# 可量化的BasicBlock
# -------------------------------
class QuantizableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add_relu(out, identity)
        return out

class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(QuantizableBasicBlock, 64, 2)
        self.layer2 = self._make_layer(QuantizableBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(QuantizableBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(QuantizableBasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)
        return x

    def fuse_model(self):
        self.eval()
        tq.fuse_modules(self, [['conv1', 'bn1', 'relu']], inplace=True)
        
        for module_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, module_name)
            for basic_block in layer:
                tq.fuse_modules(basic_block, [['conv1', 'bn1', 'relu1']], inplace=True)
                tq.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
        
        return self

# -------------------------------
# 训练函数（带详细记录）
# -------------------------------
def train_model():
    model = QuantizableResNet18(num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    logger.info("="*60)
    logger.info("Starting FULL training (30 epochs)...")
    logger.info("="*60)
    
    best_acc = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(1, 31):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/30")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix({
                'loss': f"{running_loss/total:.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accs.append(val_acc)
        
        # 日志记录
        logger.info(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model_cpu = model.to('cpu')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_cpu.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_loss': train_loss,
            }, "models/best_model.pth")
            model.to(device)
            logger.info(f"  → New best model saved! (accuracy: {best_acc:.2f}%)")
        
        scheduler.step()
    
    return model, best_acc, train_losses, val_accs

# -------------------------------
# 量化函数
# -------------------------------
def quantize_model(best_acc):
    logger.info("\n" + "="*60)
    logger.info("Starting quantization on CPU...")
    logger.info("="*60)
    
    try:
        # 加载最佳模型
        logger.info("1. Loading best model to CPU...")
        model = QuantizableResNet18(num_classes=10)
        checkpoint = torch.load("models/best_model.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 层融合
        logger.info("2. Fusing modules...")
        model = model.fuse_model()
        
        # 设置量化配置
        logger.info("3. Setting quantization config...")
        model.qconfig = tq.get_default_qconfig('fbgemm')
        
        # 准备量化
        logger.info("4. Preparing model for quantization...")
        model_prepared = tq.prepare(model, inplace=False)
        
        # 校准
        logger.info("5. Calibrating with test data (100 batches)...")
        model_prepared.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= 100:  # 用100个batch校准
                    break
                model_prepared(inputs)
        
        # 转换
        logger.info("6. Converting to INT8...")
        model_int8 = tq.convert(model_prepared, inplace=False)
        
        # 测试
        logger.info("7. Testing INT8 model...")
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model_int8(test_input)
        logger.info(f"   ✓ INT8 model output shape: {output.shape}")
        
        # 验证精度
        logger.info("8. Validating INT8 model accuracy...")
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model_int8(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        int8_acc = 100. * correct / total
        logger.info(f"   ✓ INT8 model accuracy: {int8_acc:.2f}%")
        logger.info(f"   ✓ Accuracy drop: {best_acc - int8_acc:.2f}%")
        
        # 保存INT8模型
        torch.save({
            'model_state_dict': model_int8.state_dict(),
            'fp32_acc': best_acc,
            'int8_acc': int8_acc,
            'accuracy_drop': best_acc - int8_acc
        }, "models/resnet18_int8.pth")
        logger.info("   ✓ INT8 model saved to models/resnet18_int8.pth")
        
        return model_int8, int8_acc
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# -------------------------------
# 保存训练记录
# -------------------------------
def save_training_records(best_acc, int8_acc, train_losses, val_accs):
    records = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__,
        'device': str(device),
        'total_epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'SGD',
        'scheduler': 'CosineAnnealingLR',
        'best_fp32_accuracy': float(best_acc),
        'int8_accuracy': float(int8_acc) if int8_acc else None,
        'accuracy_drop': float(best_acc - int8_acc) if int8_acc else None,
        'train_losses': [float(x) for x in train_losses],
        'validation_accuracies': [float(x) for x in val_accs],
        'total_training_time': time.time() - start_time
    }
    
    # 保存为JSON
    json_filename = f"results/training_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w') as f:
        json.dump(records, f, indent=2)
    logger.info(f"Training records saved to {json_filename}")
    
    # 保存为文本格式（易读）
    txt_filename = f"results/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(txt_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RESNET18 QUANTIZATION TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {records['training_date']}\n")
        f.write(f"PyTorch Version: {records['pytorch_version']}\n")
        f.write(f"Device: {records['device']}\n\n")
        f.write(f"Best FP32 Accuracy: {records['best_fp32_accuracy']:.2f}%\n")
        if records['int8_accuracy']:
            f.write(f"INT8 Accuracy: {records['int8_accuracy']:.2f}%\n")
            f.write(f"Accuracy Drop: {records['accuracy_drop']:.2f}%\n")
        f.write(f"Total Training Time: {records['total_training_time']:.2f} seconds\n\n")
        
        f.write("Epoch-by-Epoch Results:\n")
        f.write("-"*40 + "\n")
        for i, (loss, acc) in enumerate(zip(train_losses, val_accs), 1):
            f.write(f"Epoch {i:2d}: Train Loss={loss:.4f}, Val Acc={acc:.2f}%\n")
    
    logger.info(f"Training summary saved to {txt_filename}")

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    try:
        # 训练
        model, best_acc, train_losses, val_accs = train_model()
        logger.info(f"\n✓ Training completed! Best FP32 accuracy: {best_acc:.2f}%")
        
        # 量化
        model_int8, int8_acc = quantize_model(best_acc)
        
        # 保存记录
        save_training_records(best_acc, int8_acc, train_losses, val_accs)
        
        # 最终结果
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Best FP32 accuracy: {best_acc:.2f}%")
        if int8_acc:
            logger.info(f"INT8 accuracy: {int8_acc:.2f}%")
            logger.info(f"Accuracy drop: {best_acc - int8_acc:.2f}%")
        logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
        logger.info("="*60)
        logger.info("All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()