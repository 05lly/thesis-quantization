import torch
import torch.nn as nn
from torchvision import models
import torch.ao.quantization
import os
import datetime
import time

# =========================
# 1. 日志系统
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"export_vgg16_int8_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

def log(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{t}] {msg}"
    print(full)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full + "\n")


# =========================
# 2. 模型定义
# =========================
class QuantizableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizableVGG16, self).__init__()
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


# =========================
# 3. 导出流程
# =========================
def final_export():

    start_time = time.time()

    log("=" * 60)
    log("VGG16 INT8 模型导出开始")
    log("=" * 60)

    torch.backends.quantized.engine = 'qnnpack'
    log(f"Quant Backend: {torch.backends.quantized.engine}")

    model_dir = "/root/autodl-tmp/my_backup"
    checkpoint_path = os.path.join(model_dir, "vgg16_qat_best_weights.pth")
    save_path = os.path.join(model_dir, "vgg16_int8_deploy_final.pt")

    log(f"Checkpoint Path: {checkpoint_path}")
    log(f"Save Path: {save_path}")

    if not os.path.exists(checkpoint_path):
        log(f"[ERROR] 权重文件不存在: {checkpoint_path}")
        return

    # A. 初始化模型
    log("Step 1: 初始化模型结构")
    model = QuantizableVGG16(num_classes=10)

    model.train()
    model.fuse_model()
    log("模型融合完成 (Conv+ReLU / Linear+ReLU)")

    # B. 插入 FakeQuant
    log("Step 2: 插入 QAT 量化节点")
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(model, inplace=True)

    # C. 加载权重
    log("Step 3: 加载 QAT 权重")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    log("权重加载成功")

    # D. 切换 eval
    log("Step 4: 切换 eval 模式")
    model.eval()
    model.to('cpu')

    # E. 转 INT8
    log("Step 5: 执行 convert (FakeQuant → INT8)")
    int8_model = torch.ao.quantization.convert(model, inplace=False)
    log("INT8 转换完成")

    # F. TorchScript
    log("Step 6: 生成 TorchScript")
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(int8_model, example_input)

    torch.jit.save(traced_model, save_path)
    log("TorchScript 保存完成")

    # =========================
    # 4. 输出结果
    # =========================
    model_size = os.path.getsize(save_path) / (1024 * 1024)

    log("=" * 60)
    log("导出完成")
    log(f"INT8 模型路径: {save_path}")
    log(f"模型大小: {model_size:.2f} MB")
    log(f"耗时: {(time.time() - start_time):.2f} 秒")
    log(f"日志文件: {log_file}")
    log("=" * 60)


if __name__ == "__main__":
    final_export()