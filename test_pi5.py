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
torch.set_num_threads(4)  
torch.backends.quantized.engine = 'qnnpack'
log_filename = f"Pi5_Experiment_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
logger = logging.getLogger()

def log_msg(msg): 
    logger.info(msg)

def get_dataloader(dataset_name, input_size=224):
    """
    根据模型名自动匹配C10或C100的归一化参数。
    支持不同的输入尺寸
    """
    try:
        if "c100" in dataset_name.lower():
            # CIFAR-100 标准均值标准差
            norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            testset = datasets.CIFAR100(root=DATA_PATH, train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_size),
                                            transforms.ToTensor(),
                                            norm
                                        ]))
            log_msg(f"  [数据读取] 匹配到 CIFAR-100 测试集，输入尺寸: {input_size}")
        else:
            #CIFAR-10标准均值标准差
            norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(input_size),
                                           transforms.ToTensor(),
                                           norm
                                       ]))
            log_msg(f"  [数据读取] 匹配到 CIFAR-10 测试集，输入尺寸: {input_size}")
        return torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    except Exception as e:
        log_msg(f"检查一下 data 路径是否正确: {e}")
        return None

def evaluate_model(model_path, input_size=224, is_detection=False):
    """
    评估模型性能
    :param model_path: 模型文件路径
    :param input_size: 输入图片尺寸
    :param is_detection: 是否为目标检测模型
    :return: 评估结果列表
    """
    name = os.path.basename(model_path)
    log_msg(f"\n>>> 正在准备测试模型: {name}")
    
    # 识别精度类型
    if "int4" in name.lower():
        dtype = "INT4"
    elif "int8" in name.lower():
        dtype = "INT8"
    else:
        dtype = "FP32"
    
    # 加载模型并计算体积
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        log_msg(f"  [模型加载] 成功加载 TorchScript 模型")
    except:
        log_msg(f"  [模型加载] TorchScript 加载失败，尝试直接加载 state_dict")
        # 对于非 TorchScript 模型，需要根据模型名创建相应的模型结构
        # 这里仅作为示例，实际使用时需要根据具体模型结构进行修改
        model = None
        return [name, dtype, f"{size_mb:.2f}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
    
    # 内存占用测试
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 1. 推理延迟与FPS测试 (单张图片)
    dummy_input = torch.randn(1, 3, input_size, input_size)
    # 30次预热
    with torch.no_grad():
        for _ in range(30):
            _ = model(dummy_input)
    
    with torch.no_grad():
        latencies = []
        # 开始计时
        start_bench = time.time()
        for _ in range(100):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - t0) * 1000)
        total_bench_time = time.time() - start_bench
    
    avg_lat = np.mean(latencies)
    fps = 100 / total_bench_time  # 基于100次循环的总时长计算
    
    # 2. 吞吐量测试 (批量处理)
    batch_size = 4
    batch_input = torch.randn(batch_size, 3, input_size, input_size)
    
    with torch.no_grad():
        # 预热
        for _ in range(20):
            _ = model(batch_input)
        
        start_time = time.time()
        for _ in range(50):
            _ = model(batch_input)
        batch_inference_time = time.time() - start_time
    
    throughput = (50 * batch_size) / batch_inference_time
    
    # 3. 内存占用
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    mem_usage = final_memory - initial_memory
    
    # 4. 准确率验证
    acc = "N/A"
    if not is_detection:  # 仅对分类模型进行准确率测试
        test_loader = get_dataloader(name, input_size)
        if test_loader:
            log_msg(f"  [精度评估] 正在对测试图片进行精度评估...")
            correct, total = 0, 0
            with torch.no_grad():
                for i, (imgs, labels) in enumerate(test_loader):
                    if i >= 100:  # 测试前100张图片以节省时间
                        break
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = f"{100 * correct / total:.2f}%"
    
    # 5. 实时性判别
    realtime = "YES" if fps >= 24 else "NO"
    
    return [name, dtype, f"{size_mb:.2f}", f"{avg_lat:.2f}", f"{fps:.2f}", f"{throughput:.2f}", f"{mem_usage:.2f}", acc, realtime]

# 主程序
if __name__ == "__main__":
    
    # 分类模型列表
    classification_models = [
        # "resnet18_c10_fp32_deploy.pt",
        # "resnet18_c10_int8_deploy.pt",
        # "resnet18_c100_fp32_deploy.pt",
        # "resnet18_c100_int8_deploy.pt",
        # "vgg16_c10_fp32_deploy.pt",
        # "vgg16_c10_int8_deploy.pt",
        # "vgg16_c100_fp32_deploy.pt",
        # "vgg16_c100_int8_deploy.pt",
        "mobilenetv2_c10_fp32_deploy.pt",
        "mobilenetv2_c10_int8_deploy.pt",
        "mobilenetv2_c100_fp32_deploy.pt",
        "mobilenetv2_c100_int8_deploy.pt",
        # 添加INT4模型（需要先导出为TorchScript）
        # "mobilenetv2_c10_int4_deploy.pt",
        # "mobilenetv2_c100_int4_deploy.pt",
    ]  
    
    # 目标检测模型列表
    detection_models = [
        # 添加目标检测模型（需要先导出为TorchScript）
        # "simple_yolo_fp32_deploy.pt",
        # "simple_yolo_int8_deploy.pt",
        # "simple_yolo_int4_deploy.pt",
    ]
    
    log_msg("="*120)
    log_msg(f"{'Model Name':<35} | {'Type':<6} | {'Size':<7} | {'Lat(ms)':<8} | {'FPS':<7} | {'Throughput':<12} | {'Mem(MB)':<8} | {'Acc':<8} | {'RT'}")
    log_msg("-"*120)

    results_table = []

    # 测试分类模型
    for i, m in enumerate(classification_models):
        if os.path.exists(m):
            # 核心评估
            res = evaluate_model(m, input_size=224, is_detection=False)
            results_table.append(res)
            # 打印当前行的结果
            log_msg(f"{res[0]:<35} | {res[1]:<6} | {res[2]:<7} | {res[3]:<8} | {res[4]:<7} | {res[5]:<12} | {res[6]:<8} | {res[7]:<8} | {res[8]}")
            if i < len(classification_models) - 1:
                log_msg(f"  >> 散热保护：等待30秒进行下一次测试...")
                time.sleep(30)
        else:
            log_msg(f"找不到文件: {m}，请检查路径。")
    
    # 测试目标检测模型
    for i, m in enumerate(detection_models):
        if os.path.exists(m):
            # 核心评估
            res = evaluate_model(m, input_size=640, is_detection=True)
            results_table.append(res)
            # 打印当前行的结果
            log_msg(f"{res[0]:<35} | {res[1]:<6} | {res[2]:<7} | {res[3]:<8} | {res[4]:<7} | {res[5]:<12} | {res[6]:<8} | {res[7]:<8} | {res[8]}")
            if i < len(detection_models) - 1:
                log_msg(f"  >> 散热保护：等待30秒进行下一次测试...")
                time.sleep(30)
        else:
            log_msg(f"找不到文件: {m}，请检查路径。")

    log_msg("="*120)
    log_msg(f"\n测试结束 实验日志已保存至: {log_filename}")
    log_msg("\n测试指标说明:")
    log_msg("- Lat(ms): 单张图片推理延迟（毫秒）")
    log_msg("- FPS: 每秒处理帧数")
    log_msg("- Throughput: 吞吐量（每秒处理图片数，batch_size=4）")
    log_msg("- Mem(MB): 内存占用增加量")
    log_msg("- Acc: 分类准确率（目标检测模型不适用）")
    log_msg("- RT: 实时性（FPS>=24为YES）")
