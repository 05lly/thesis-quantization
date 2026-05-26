import torch
import torchvision.transforms as transforms
from torchvision import datasets
import time
import numpy as np
import os
import logging
from datetime import datetime
import psutil

# --- 环境配置 ---
DATA_PATH = './data'
torch.set_num_threads(4) # 树莓派5的4核性能
torch.backends.quantized.engine = 'qnnpack' # ARM平台的加速引擎
log_filename = f"Pi5_192_Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
logger = logging.getLogger()

def log_msg(msg): 
    logger.info(msg)

def get_dataloader(model_name):
    """根据文件名自动匹配数据集和归一化参数，固定 192 尺寸"""
    name_lower = model_name.lower()
    input_size = 192
    
    if "c100" in name_lower:
        norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        testset = datasets.CIFAR100(root=DATA_PATH, train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(input_size),
                                        transforms.ToTensor(),
                                        norm
                                    ]))
        tag = "CIFAR-100"
    else:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False,
                                   transform=transforms.Compose([
                                       transforms.Resize(input_size),
                                       transforms.ToTensor(),
                                       norm
                                   ]))
        tag = "CIFAR-10"
        
    log_msg(f"  [Loader] 载入 {tag} | 输入尺寸: {input_size}")
    return torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

def evaluate_model(model_path):
    name = os.path.basename(model_path)
    log_msg(f"\n>>> 正在准备测试模型: {name}")
    
    # 恢复识别精度类型
    dtype = "INT8" if "int8" in name.lower() else "FP32"
    
    # 核心参数
    input_size = 192
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model = torch.jit.load(model_path, map_location='cpu').eval()

    # --- 推理性能测试 ---
    dummy_input = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        # 预热 30 次
        for _ in range(30): _ = model(dummy_input)

        latencies = []
        start_bench = time.time()
        for _ in range(100):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - t0) * 1000)
        total_time = time.time() - start_bench

    avg_lat = np.mean(latencies)
    fps = 100 / total_time 

    # --- 系统资源 & 精度验证 ---
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    test_loader = get_dataloader(name)
    acc = "N/A"

    if test_loader:
        log_msg(f"  正在验证 10,000 张图片的真实 {dtype} 精度...")
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = f"{100 * correct / total:.2f}%"
    realtime = "YES" if fps >= 24 else "NO"
    return [name, dtype, f"{size_mb:.2f}", f"{avg_lat:.2f}", f"{fps:.2f}", f"{mem_mb:.2f}", acc, realtime]

if __name__ == "__main__":
    target_models = [
        "resnet18_c10_int8_deploy_192.pt",
        "resnet18_c100_int8_deploy_192.pt"
    ]
    
    log_msg("="*105)
    log_msg(f"{'Model Name':<35} | {'Type':<6} | {'Size':<7} | {'Lat(ms)':<8} | {'FPS':<7} | {'Mem(MB)':<8} | {'Acc':<8} | {'RT'}")
    log_msg("-"*105)

    for i, m in enumerate(target_models):
        if os.path.exists(m):
            res = evaluate_model(m)
            log_msg(f"{res[0]:<35} | {res[1]:<6} | {res[2]:<7} | {res[3]:<8} | {res[4]:<7} | {res[5]:<8} | {res[6]:<8} | {res[7]}")
            
            if i < len(target_models) - 1:
                # 散热时间
                log_msg(f"  >> 散热保护：等待120秒进行下一次测试...")
                time.sleep(120)
        else:
            log_msg(f"错误: 找不到模型文件 {m}")

    log_msg("="*105)
    log_msg(f"\n测试结束 实验日志已保存至: {log_filename}")