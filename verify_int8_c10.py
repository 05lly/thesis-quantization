import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import time
import datetime
import platform
from tqdm import tqdm

# 环境配置
current_engine = 'fbgemm' if platform.system() == 'Windows' else 'qnnpack'
torch.backends.quantized.engine = current_engine
device = torch.device("cpu")

model_dir = r"D:\Graduation_Design\thesis-quantization\models"
log_dir = r"D:\Graduation_Design\thesis-quantization\logs"
os.makedirs(log_dir, exist_ok=True)

#日志
log_filename = os.path.join(
    log_dir, f"c10_audit_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

#VGG16结构
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
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
                    if i + 1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)
                    elif i + 1 < len(m) and type(m[i]) == nn.Linear and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

#数据
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform_test),
    batch_size=64,
    shuffle=False,
    num_workers=0
)

def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

# 函数
def audit_model(model_name):
    log_message(f"\n>>> 开始测试模型: {model_name} (Engine: {current_engine})")
    log_message(f"Batch Size: {test_loader.batch_size}")

    if model_name == "VGG16":
        best_qat_path = os.path.join(model_dir, "vgg16_c10_qat_best.pth")
        fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")
        deploy_path = os.path.join(model_dir, "vgg16_c10_int8_deploy.pt")

        model = QuantizableVGG16(10)
        model.eval()
        model.fuse_model()

    elif model_name == "ResNet18":
        best_qat_path = os.path.join(model_dir, "resnet18_c10_qat_best.pth")
        fp32_path = os.path.join(model_dir, "fp32_resnet18_best.pth")
        deploy_path = os.path.join(model_dir, "resnet18_c10_int8_deploy.pt")

        model = models.quantization.resnet18(weights=None, quantize=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.eval()
        model.fuse_model(is_qat=True)

    elif model_name == "MobileNetV2":
        best_qat_path = os.path.join(model_dir, "mobilenetv2_c10_qat_best.pth")
        fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
        deploy_path = os.path.join(model_dir, "mobilenetv2_c10_int8_deploy.pt")

        model = models.quantization.mobilenet_v2(weights=None, quantize=False)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        model.eval()
        model.fuse_model(is_qat=True)

    #  加载INT模型
    if os.path.exists(deploy_path):
        log_message(f"加载 INT8 模型: {deploy_path}")
        int8_model = torch.jit.load(deploy_path)
    else:
        log_message("没有找到部署模型，开始量化转换")

        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(current_engine)
        torch.ao.quantization.prepare_qat(model, inplace=True)

        if not os.path.exists(best_qat_path):
            log_message(f"缺少权重: {best_qat_path}")
            return

        model.load_state_dict(torch.load(best_qat_path, map_location='cpu'))

        model.eval()
        int8_model = torch.ao.quantization.convert(model, inplace=False)

        traced_model = torch.jit.trace(int8_model, torch.randn(1, 3, 224, 224))
        torch.jit.save(traced_model, deploy_path)

    # 推理
    log_message("开始 INT8 推理测试")

    correct = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs = inputs.to(device)
            outputs = int8_model(inputs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()

    eval_time = time.time() - start_time
    total_images = len(test_loader.dataset)

    acc = 100. * correct / total_images
    fps = total_images / eval_time
    latency = (eval_time / total_images) * 1000

    #输出
    fp32_size = get_size_mb(fp32_path)
    int8_size = get_size_mb(deploy_path)

    log_message("=" * 60)
    log_message(f"模型: {model_name} (CIFAR-10 INT8)")
    log_message(f"Accuracy: {acc:.2f}%")
    log_message(f"FP32 Size: {fp32_size:.2f} MB")
    log_message(f"INT8 Size: {int8_size:.2f} MB")
    log_message(f"Compression: {fp32_size/int8_size:.2f}x")
    log_message(f"Total Time: {eval_time:.2f}s")
    log_message(f"FPS: {fps:.2f}")
    log_message(f"Latency: {latency:.2f} ms")
    log_message("=" * 60)

#主函数
if __name__ == "__main__":
    log_message("开始测试")

    for m in ["VGG16", "ResNet18", "MobileNetV2"]:
        audit_model(m)

    log_message("全部测试完成")