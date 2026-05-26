
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import random
import time

# -----------------------------
# 基础配置
# -----------------------------
MODEL_PATH = "resnet18_c100_int8_deploy.pt"
# MODEL_PATH = "resnet18_c100_fp32_deploy.pt"

DEVICE = "cpu"
INPUT_SIZE = 224

torch.backends.quantized.engine = "qnnpack"
torch.set_num_threads(4)

# -----------------------------
# CIFAR-100 类别名
# -----------------------------
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
    'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
    'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
    'wolf', 'woman', 'worm'
]

# -----------------------------
# 数据预处理
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4865, 0.4409),
        (0.2673, 0.2564, 0.2761)
    )
])

# -----------------------------
# 加载 CIFAR100 测试集
# -----------------------------
testset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# -----------------------------
# 加载模型
# -----------------------------
print("正在加载模型...")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

print("模型加载成功")

# -----------------------------
# 随机抽取样本测试
# -----------------------------
sample_num = 10

indices = random.sample(range(len(testset)), sample_num)

correct = 0
total_time = 0

print("\n开始随机样本检测:\n")

with torch.no_grad():

    for i, idx in enumerate(indices):

        image, label = testset[idx]

        image = image.unsqueeze(0)

        start = time.time()

        output = model(image)

        end = time.time()

        pred = output.argmax(1).item()

        infer_time = (end - start) * 1000
        total_time += infer_time

        gt_name = classes[label]
        pred_name = classes[pred]

        result = "√" if pred == label else "×"

        if pred == label:
            correct += 1

        print(f"[样本 {i+1}]")
        print(f"真实类别 : {gt_name}")
        print(f"预测类别 : {pred_name}")
        print(f"推理耗时 : {infer_time:.2f} ms")
        print(f"结果     : {result}")
        print("-" * 40)

# -----------------------------
# 汇总结果
# -----------------------------
acc = correct / sample_num * 100
avg_time = total_time / sample_num
fps = 1000 / avg_time

print("\n============================")
print(f"样本数量       : {sample_num}")
print(f"预测正确       : {correct}")
print(f"样本准确率     : {acc:.2f}%")
print(f"平均推理时间   : {avg_time:.2f} ms")
print(f"FPS            : {fps:.2f}")
print("============================")