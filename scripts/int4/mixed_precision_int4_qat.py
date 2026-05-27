"""
Layer-sensitivity-aware mixed-precision INT4 QAT.

This script uses a CSV produced by layer_sensitivity_analysis.py to select
sensitive layers. Sensitive layers are assigned INT8 fake quantization, while
all other quantizable layers are assigned INT4 fake quantization.

Important:
  - This is still QAT fake quantization for low-bit research.
  - It does not guarantee real INT4 acceleration on Raspberry Pi.
  - No first/last-layer rule is applied; only the sensitivity CSV decides which
    layers are kept at INT8.

Examples:
  python scripts/int4/mixed_precision_int4_qat.py \
    --model resnet18 \
    --dataset cifar100 \
    --sensitivity-csv results/int4_sensitivity/resnet18_cifar100_test_int4_layer_sensitivity_full_xxx.csv \
    --sensitive-ratio 0.2

  python scripts/int4/mixed_precision_int4_qat.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --sensitivity-csv results/int4_sensitivity/mobilenetv2_cifar10_test_int4_layer_sensitivity_full_xxx.csv \
    --sensitive-ratio 0.2
"""

import argparse
import csv
import datetime
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import FakeQuantize, QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torchvision import datasets, models, transforms
from torchvision.models.quantization import mobilenet_v2 as quant_mobilenet_v2
from tqdm import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


def get_project_paths() -> Tuple[str, str, str]:
    if os.path.exists("/root/autodl-tmp"):
        data_dir = "/root/autodl-tmp/data"
        model_dir = "/root/autodl-tmp/my_backup"
    else:
        data_dir = "data"
        model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return data_dir, model_dir, log_dir


def get_int4_qat_qconfig() -> QConfig:
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=15,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-8,
            quant_max=7,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )


def get_int8_qat_qconfig() -> QConfig:
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )


def build_model(model_name: str, dataset_name: str) -> nn.Module:
    num_classes = 10 if dataset_name == "cifar10" else 100
    if model_name == "resnet18":
        model = models.quantization.resnet18(weights=None, quantize=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenetv2":
        model = quant_mobilenet_v2(weights=None, quantize=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def candidate_checkpoint_names(model_name: str, dataset_name: str) -> List[str]:
    if model_name == "resnet18" and dataset_name == "cifar100":
        return ["fp32_resnet18_c100_best.pth", "fp32_resnet18_best_c100.pth"]
    if model_name == "resnet18" and dataset_name == "cifar10":
        return ["fp32_resnet18_best.pth", "fp32_resnet18_c10_best.pth"]
    if model_name == "mobilenetv2" and dataset_name == "cifar100":
        return ["fp32_mobilenetv2_c100_best.pth", "fp32_mobilenetv2_best_c100.pth"]
    if model_name == "mobilenetv2" and dataset_name == "cifar10":
        return ["fp32_mobilenetv2_best.pth", "fp32_mobilenetv2_c10_best.pth"]
    return []


def resolve_checkpoint(model_dir: str, model_name: str, dataset_name: str, checkpoint: Optional[str]) -> str:
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    tried = []
    for name in candidate_checkpoint_names(model_name, dataset_name):
        path = os.path.join(model_dir, name)
        tried.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No FP32 checkpoint found. Tried:\n" + "\n".join(tried))


def load_state_dict_safely(model: nn.Module, checkpoint_path: str) -> None:
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)


def build_dataloaders(dataset_name: str, data_dir: str, batch_size: int, num_workers: int, input_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = dataset_class(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = dataset_class(root=data_dir, train=False, download=True, transform=test_transform)
    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def read_sensitivity_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sensitivity CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty sensitivity CSV: {csv_path}")
    required = {"layer_name", "accuracy_drop"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Sensitivity CSV missing columns: {missing}")
    return sorted(rows, key=lambda r: float(r["accuracy_drop"]), reverse=True)


def select_sensitive_layers(rows: List[Dict[str, str]], sensitive_ratio: float, sensitive_topk: int) -> Set[str]:
    if sensitive_topk > 0:
        keep_count = min(sensitive_topk, len(rows))
    else:
        if not 0.0 < sensitive_ratio < 1.0:
            raise ValueError("sensitive_ratio must be in (0, 1) when sensitive_topk is not set")
        keep_count = max(1, round(len(rows) * sensitive_ratio))
    return {row["layer_name"] for row in rows[:keep_count]}


def assign_mixed_precision_qconfig(model: nn.Module, sensitivity_rows: List[Dict[str, str]], sensitive_layers: Set[str], logger) -> Tuple[int, int, Set[str]]:
    """Assign INT8 qconfig to sensitive layers and INT4 qconfig to others.

    Layer names come from the sensitivity CSV generated before fusion. After
    fusion, modules such as Conv2d + ReLU may become fused modules, so we should
    not rely only on isinstance(module, nn.Conv2d). Instead, the CSV layer names
    are treated as the authoritative quantizable layer list.
    """
    int4_qconfig = get_int4_qat_qconfig()
    int8_qconfig = get_int8_qat_qconfig()
    csv_layers = {row["layer_name"] for row in sensitivity_rows}
    matched_sensitive_layers: Set[str] = set()

    # Default: all modules inherit INT4 unless explicitly overridden.
    model.qconfig = int4_qconfig
    for name, module in model.named_modules():
        if name in csv_layers:
            module.qconfig = int4_qconfig
        if name in sensitive_layers:
            module.qconfig = int8_qconfig
            matched_sensitive_layers.add(name)
            logger(f"[QCONFIG] INT8 sensitive layer: {name} ({module.__class__.__name__})")

    unmatched = sensitive_layers - matched_sensitive_layers
    if unmatched:
        logger(f"[WARN] Sensitive layers not matched after model construction/fusion: {sorted(unmatched)}")

    return len(csv_layers), len(matched_sensitive_layers), matched_sensitive_layers


def fuse_model_if_supported(model: nn.Module, is_qat: bool = True) -> None:
    if hasattr(model, "fuse_model"):
        model.eval()
        model.fuse_model(is_qat=is_qat)


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def estimate_theoretical_mixed_size_mb(fp32_size_mb: float, total_layers: int, int8_layers: int) -> float:
    if total_layers <= 0:
        return fp32_size_mb
    int8_ratio = int8_layers / total_layers
    int4_ratio = 1.0 - int8_ratio
    # Approximation by layer count. It is used only as a coarse theoretical estimate.
    return fp32_size_mb * (int8_ratio / 4.0 + int4_ratio / 8.0)


def train_mixed_precision_qat(args: argparse.Namespace) -> None:
    data_dir, model_dir, log_dir = get_project_paths()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.quantized.engine = "qnnpack"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"mixed_precision_int4_qat_{args.model}_{args.dataset}_{timestamp}.log")

    def log_message(message: str) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{now}] {message}"
        print(text)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    rows = read_sensitivity_csv(args.sensitivity_csv)
    sensitive_layers = select_sensitive_layers(rows, args.sensitive_ratio, args.sensitive_topk)
    checkpoint_path = resolve_checkpoint(model_dir, args.model, args.dataset, args.checkpoint)

    log_message("=" * 80)
    log_message("Layer-Sensitivity-Aware Mixed-Precision INT4 QAT")
    log_message(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    log_message(f"Sensitivity CSV: {args.sensitivity_csv}")
    log_message(f"FP32 checkpoint: {checkpoint_path}")
    log_message(f"Sensitive layer selection: topk={args.sensitive_topk}, ratio={args.sensitive_ratio}")
    log_message(f"Sensitive layers selected: {len(sensitive_layers)}")
    log_message("No first/last-layer rule is used; selection is only based on sensitivity ranking.")

    train_loader, test_loader = build_dataloaders(args.dataset, data_dir, args.batch_size, args.num_workers, args.input_size)
    model = build_model(args.model, args.dataset)
    load_state_dict_safely(model, checkpoint_path)
    model.to(device)

    model.train()
    total_layers, int8_layers, matched_sensitive_layers = assign_mixed_precision_qconfig(model, rows, sensitive_layers, log_message)
    int4_layers = total_layers - int8_layers
    log_message(f"CSV quantizable layers: {total_layers} | Matched INT8 sensitive layers: {int8_layers} | INT4 layers: {int4_layers}")

    fuse_model_if_supported(model, is_qat=True)
    if int8_layers == 0:
        raise RuntimeError("No sensitive layers were matched. Please check whether the sensitivity CSV matches the selected model.")

    # prepare_qat requires training mode; fusion may switch the model to eval mode.
    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_time = time.time()
    save_name = f"{args.model}_{args.dataset}_mixed_precision_int4_qat_best.pth"
    save_path = os.path.join(model_dir, save_name)

    for epoch in range(args.epochs):
        model.train()
        if epoch >= args.freeze_epoch:
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1:02d}/{args.epochs:02d}]", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc = evaluate(model, test_loader, device)
        log_message(
            f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] | "
            f"Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | "
            f"Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            log_message(f"Best checkpoint saved: {save_path}")

    fp32_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    mixed_size_mb = estimate_theoretical_mixed_size_mb(fp32_size_mb, total_layers, int8_layers)
    log_message("=" * 80)
    log_message("Mixed-Precision INT4 QAT Final Report")
    log_message(f"Best Test Accuracy: {best_acc:.2f}%")
    log_message(f"INT8 Sensitive Layers: {int8_layers}/{total_layers}")
    log_message(f"INT4 Layers: {int4_layers}/{total_layers}")
    log_message(f"Matched INT8 Layer Names: {sorted(matched_sensitive_layers)}")
    log_message(f"FP32 Checkpoint Size: {fp32_size_mb:.2f} MB")
    log_message(f"Theoretical Mixed-Precision Size: {mixed_size_mb:.2f} MB")
    log_message(f"Total Time: {(time.time() - start_time) / 60:.2f} min")
    log_message("Note: size is a theoretical estimate; this script uses fake quantization for QAT research.")
    log_message("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-sensitivity-aware mixed-precision INT4 QAT")
    parser.add_argument("--model", choices=["resnet18", "mobilenetv2"], required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--sensitivity-csv", required=True, help="CSV produced by layer_sensitivity_analysis.py")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit FP32 checkpoint path")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epoch", type=int, default=8)
    parser.add_argument("--sensitive-ratio", type=float, default=0.2, help="Top ratio of sensitive layers kept at INT8")
    parser.add_argument("--sensitive-topk", type=int, default=0, help="If >0, use exact top-k sensitive layers instead of ratio")
    return parser.parse_args()


if __name__ == "__main__":
    train_mixed_precision_qat(parse_args())
