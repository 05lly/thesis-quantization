import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import time
import numpy as np
import os
import logging
from datetime import datetime
import psutil

# 环境配置
DATA_PATH = './data'
torch.set_num_threads(4)  #树莓派5的4核性能
torch.backends.quantized.engine = 'qnnpack' #ARM平台的加速引擎
log_filename = f"Pi5_Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
logger = logging.getLogger()

def log_msg(msg): 
    logger.info(msg)

#数据预处理逻辑 
def get_dataloader(dataset_name):
    """
    根据模型名自动匹配 C10 或 C100 的归一化参数。
    """
    try:
        if "c100" in dataset_name.lower():
            # CIFAR-100 标准均值标准差
            norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            testset = datasets.CIFAR100(root=DATA_PATH, train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            norm
                                        ]))
            log_msg("  [数据读取] 匹配到 CIFAR-100 测试集")
        else:
            #CIFAR-10标准均值标准差
            norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(224),
                                           transforms.ToTensor(),
                                           norm
                                       ]))
            log_msg("匹配到 CIFAR-10 测试集")
        return torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    except Exception as e:
        log_msg(f"检查一下 data 路径是否正确: {e}")
        return None

#核心评估函数
def evaluate_model(model_path):
    name = os.path.basename(model_path)
    log_msg(f"\n>>> 正在准备测试模型: {name}")
    #识别精度类型
    dtype = "INT8" if "int8" in name.lower() else "FP32"
    # 加载模型并计算体积
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model = torch.jit.load(model_path, map_location='cpu').eval()

    #推理延迟与FPS测试 
    dummy_input = torch.randn(1, 3, 224, 224)
    #30次预热，让系统调度进入状态
    with torch.no_grad():
        for _ in range(30):
            _ = model(dummy_input)

        latencies = []
        #开始计时
        start_bench = time.time()
        for _ in range(100):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - t0) * 1000)
        total_bench_time = time.time() - start_bench

    avg_lat = np.mean(latencies)
    fps = 100 / total_bench_time  # 基于100次循环的总时长计算，结果更稳

    #系统资源监控
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024

    #准确率验证
    test_loader = get_dataloader(name)
    acc = "N/A"

    if test_loader:
        log_msg(f" 正在对10000张测试图片进行精度评估...")
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = f"{100 * correct / total:.2f}%"
    # 实时性标准判别
    realtime = "YES" if fps >= 24 else "NO"
    return [name, dtype, f"{size_mb:.2f}", f"{avg_lat:.2f}", f"{fps:.2f}", f"{mem_mb:.2f}", acc, realtime]

# 主程序逻辑
if __name__ == "__main__":
    
    my_models = [
        "resnet18_c10_fp32_deploy.pt",
        "resnet18_c10_int8_deploy.pt",
        "resnet18_c100_fp32_deploy.pt",
        "resnet18_c100_int8_deploy.pt",
        "vgg16_c10_fp32_deploy.pt",
        "vgg16_c10_int8_deploy.pt",
        "vgg16_c100_fp32_deploy.pt",
        "vgg16_c100_int8_deploy.pt",
        "mobilenetv2_c10_fp32_deploy.pt",
        "mobilenetv2_c10_int8_deploy.pt",
        "mobilenetv2_c100_fp32_deploy.pt",
        "mobilenetv2_int8_deploy.pt",
    ]  
    log_msg("="*105)
    log_msg(f"{'Model Name':<35} | {'Type':<6} | {'Size':<7} | {'Lat(ms)':<8} | {'FPS':<7} | {'Mem(MB)':<8} | {'Acc':<8} | {'RT'}")
    log_msg("-"*105)

    results_table = []

    for i, m in enumerate(my_models):
        if os.path.exists(m):
            # 核心评估
            res = evaluate_model(m)
            results_table.append(res)
            # 打印当前行的结果
            log_msg(f"{res[0]:<35} | {res[1]:<6} | {res[2]:<7} | {res[3]:<8} | {res[4]:<7} | {res[5]:<8} | {res[6]:<8} | {res[7]}")
            # 关键一步：如果是测试列表中的最后一个模型，就不睡了；否则睡 5 秒给 CPU 降降温
            if i < len(my_models) - 1:
                log_msg(f"  >> 散热保护：等待30秒进行下一次测试...")
                time.sleep(30)
        else:
            log_msg(f"找不到文件: {m}，请检查路径。")

    log_msg("="*105)
    log_msg(f"\n测试结束 实验日志已保存至: {log_filename}")