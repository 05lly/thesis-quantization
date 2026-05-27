"""
ResNet18 block-level mixed-precision INT4 QAT.

Purpose
-------
Train a ResNet18 fake-quantized QAT model where all quantizable modules are INT4
by default, and selected residual blocks are restored to INT8. The selected
blocks should come from resnet18_block_recovery_sensitivity.py.

Example
-------
python scripts/int4/resnet18_block_mixed_precision_qat.py \
    --dataset cifar100 \
    --checkpoint /root/autodl-tmp/my_backup/fp32_resnet18_c100_best.pth \
    --int8-blocks layer3.1,layer2.0,layer2.1,layer3.0 \
    --epochs 30 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --freeze-epoch 8 \
    --device cuda

Outputs
-------
1. logs/resnet18_block_mixed_precision_qat_<dataset>_<timestamp>.log
2. results/int4_block_mixed_qat/.../training_history.csv
3. results/int4_block_mixed_qat/.../experiment_summary.json
4. best checkpoint in /root/autodl-tmp/my_backup or models

Important
---------
This script uses PyTorch fake quantization for QAT research. It does not produce
a real deployable INT4 model and does not validate real INT4 acceleration.
"""

import argparse
import csv
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import FakeQuantize, QConfig
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


@dataclass
class BlockAssignment:
    block_name: str
    module_prefixes: List[str]
    matched_modules: List[str]
    parameter_count: int


class ExperimentLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, message: str) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {message}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def get_project_paths() -> Tuple[str, str, str, str]:
    if os.path.exists("/root/autodl-tmp"):
        data_dir = "/root/autodl-tmp/data"
        model_dir = "/root/autodl-tmp/my_backup"
    else:
        data_dir = "data"
        model_dir = "models"

    result_dir = os.path.join("results", "int4_block_mixed_qat")
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return data_dir, model_dir, result_dir, log_dir


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


def build_resnet18(dataset_name: str) -> nn.Module:
    num_classes = 10 if dataset_name == "cifar10" else 100
    model = models.quantization.resnet18(weights=None, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def candidate_checkpoint_names(dataset_name: str) -> List[str]:
    if dataset_name == "cifar100":
        return ["fp32_resnet18_c100_best.pth", "fp32_resnet18_best_c100.pth"]
    if dataset_name == "cifar10":
        return ["fp32_resnet18_best.pth", "fp32_resnet18_c10_best.pth"]
    return []


def resolve_checkpoint(model_dir: str, dataset_name: str, checkpoint: Optional[str]) -> str:
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    tried = []
    for name in candidate_checkpoint_names(dataset_name):
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


def build_dataloaders(
    dataset_name: str,
    data_dir: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = dataset_class(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = dataset_class(root=data_dir, train=False, download=True, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def parse_int8_blocks(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def is_under_prefix(module_name: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return True
    return False


def block_to_prefixes(block_name: str) -> List[str]:
    if block_name == "stem":
        return ["conv1"]
    if block_name == "classifier":
        return ["fc"]
    return [block_name]


def validate_resnet18_block_names(blocks: Sequence[str]) -> None:
    valid = {f"layer{i}.{j}" for i in range(1, 5) for j in range(2)}
    valid.update({"stem", "classifier"})
    invalid = [b for b in blocks if b not in valid]
    if invalid:
        raise ValueError(
            "Invalid --int8-blocks entries: "
            + ", ".join(invalid)
            + "\nValid entries are: "
            + ", ".join(sorted(valid))
        )


def count_params_for_prefixes(model: nn.Module, prefixes: Sequence[str]) -> int:
    seen = set()
    total = 0
    for name, module in model.named_modules():
        if is_under_prefix(name, prefixes):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


def collect_quantizable_module_names(model: nn.Module, prefixes: Sequence[str]) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if is_under_prefix(name, prefixes) and isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def collect_block_assignments(model: nn.Module, int8_blocks: Sequence[str]) -> List[BlockAssignment]:
    assignments = []
    for block in int8_blocks:
        prefixes = block_to_prefixes(block)
        matched_modules = collect_quantizable_module_names(model, prefixes)
        parameter_count = count_params_for_prefixes(model, prefixes)
        assignments.append(
            BlockAssignment(
                block_name=block,
                module_prefixes=prefixes,
                matched_modules=matched_modules,
                parameter_count=parameter_count,
            )
        )
    return assignments


def fuse_model_if_supported(model: nn.Module, is_qat: bool = True) -> None:
    if hasattr(model, "fuse_model"):
        model.eval()
        model.fuse_model(is_qat=is_qat)


def assign_block_mixed_qconfig(
    model: nn.Module,
    int8_blocks: Sequence[str],
    logger: ExperimentLogger,
) -> Tuple[int, int, int, List[str]]:
    int4_qconfig = get_int4_qat_qconfig()
    int8_qconfig = get_int8_qat_qconfig()

    int8_prefixes: List[str] = []
    for block in int8_blocks:
        int8_prefixes.extend(block_to_prefixes(block))

    model.qconfig = int4_qconfig

    total_quantizable = 0
    int8_quantizable = 0
    int4_quantizable = 0
    matched_int8_modules: List[str] = []

    for name, module in model.named_modules():
        if not name:
            continue

        if is_under_prefix(name, int8_prefixes):
            module.qconfig = int8_qconfig
        else:
            module.qconfig = int4_qconfig

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_quantizable += 1
            if is_under_prefix(name, int8_prefixes):
                int8_quantizable += 1
                matched_int8_modules.append(name)
                logger.log(f"[QCONFIG] INT8 block module: {name} ({module.__class__.__name__})")
            else:
                int4_quantizable += 1

    logger.log(
        f"[QCONFIG] Quantizable modules: {total_quantizable} | "
        f"INT8 modules: {int8_quantizable} | INT4 modules: {int4_quantizable}"
    )
    logger.log(f"[QCONFIG] Matched INT8 module names: {matched_int8_modules}")
    return total_quantizable, int8_quantizable, int4_quantizable, matched_int8_modules


def estimate_quantized_size_mb(model: nn.Module, int8_blocks: Sequence[str]) -> float:
    int8_prefixes: List[str] = []
    for block in int8_blocks:
        int8_prefixes.extend(block_to_prefixes(block))

    seen = set()
    total_bits = 0
    for name, module in model.named_modules():
        for p in module.parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            bit = 8 if is_under_prefix(name, int8_prefixes) else 4
            total_bits += p.numel() * bit

    return total_bits / 8 / (1024 ** 2)


def freeze_observers_and_bn(model: nn.Module) -> None:
    model.apply(torch.ao.quantization.disable_observer)
    try:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    except Exception:
        pass


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
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


def write_history_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_block_mixed_qat(args: argparse.Namespace) -> None:
    data_dir, model_dir, result_root, log_dir = get_project_paths()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    int8_blocks = parse_int8_blocks(args.int8_blocks)
    validate_resnet18_block_names(int8_blocks)

    output_dir = os.path.join(
        result_root,
        f"resnet18_{args.dataset}_block_mixed_qat_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"resnet18_block_mixed_precision_qat_{args.dataset}_{timestamp}.log",
    )
    logger = ExperimentLogger(log_path)

    checkpoint_path = resolve_checkpoint(model_dir, args.dataset, args.checkpoint)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.quantized.engine = "qnnpack"

    logger.log("=" * 80)
    logger.log("ResNet18 Block-Level Mixed-Precision INT4 QAT")
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Device: {device}")
    logger.log(f"FP32 checkpoint: {checkpoint_path}")
    logger.log(f"INT8 blocks: {int8_blocks}")
    logger.log(f"Epochs: {args.epochs}")
    logger.log(f"LR: {args.lr}")
    logger.log(f"Weight decay: {args.weight_decay}")
    logger.log(f"Freeze observers / BN from epoch: {args.freeze_epoch}")
    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Log path: {log_path}")
    logger.log("This script uses fake quantization for QAT research.")
    logger.log("=" * 80)

    train_loader, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        data_dir=data_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet18(args.dataset)
    load_state_dict_safely(model, checkpoint_path)
    model.to(device)

    # Standard eager QAT order: load FP32 -> fuse -> assign qconfig -> train -> prepare_qat.
    fuse_model_if_supported(model, is_qat=True)
    block_assignments = collect_block_assignments(model, int8_blocks)
    total_q, int8_q, int4_q, matched_int8_modules = assign_block_mixed_qconfig(
        model,
        int8_blocks=int8_blocks,
        logger=logger,
    )
    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0
    history: List[Dict[str, object]] = []

    best_ckpt_path = os.path.join(
        model_dir,
        f"resnet18_{args.dataset}_block_mixed_int4_qat_best.pth",
    )

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        if epoch >= args.freeze_epoch:
            freeze_observers_and_bn(model)

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch [{epoch + 1:02d}/{args.epochs:02d}]", leave=False)
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
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
        lr_now = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch + 1,
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "loss": round(train_loss, 6),
            "lr": lr_now,
        }
        history.append(row)

        logger.log(
            f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] | "
            f"Train Acc: {train_acc:6.2f}% | "
            f"Test Acc: {test_acc:6.2f}% | "
            f"Loss: {train_loss:.4f} | "
            f"LR: {lr_now:.6f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "dataset": args.dataset,
                    "model_name": "resnet18",
                    "method": "block_mixed_precision_int4_qat",
                    "int8_blocks": int8_blocks,
                    "matched_int8_modules": matched_int8_modules,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            logger.log(f"Best checkpoint saved: {best_ckpt_path}")

    total_time = time.time() - start_time

    history_csv = os.path.join(output_dir, "training_history.csv")
    write_history_csv(history_csv, history)

    fp32_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    theoretical_size_mb = estimate_quantized_size_mb(model, int8_blocks)

    summary = {
        "experiment": "resnet18_block_mixed_precision_int4_qat",
        "timestamp": timestamp,
        "dataset": args.dataset,
        "fp32_checkpoint": checkpoint_path,
        "best_checkpoint": best_ckpt_path,
        "best_test_accuracy": round(best_acc, 4),
        "best_epoch": best_epoch,
        "int8_blocks": int8_blocks,
        "block_assignments": [asdict(x) for x in block_assignments],
        "matched_int8_modules": matched_int8_modules,
        "total_quantizable_modules": total_q,
        "int8_quantizable_modules": int8_q,
        "int4_quantizable_modules": int4_q,
        "fp32_checkpoint_size_mb": round(fp32_size_mb, 6),
        "theoretical_mixed_precision_size_mb": round(theoretical_size_mb, 6),
        "training_history_csv": history_csv,
        "log_path": log_path,
        "output_dir": output_dir,
        "total_time_minutes": round(total_time / 60.0, 4),
        "args": vars(args),
        "note": (
            "Theoretical size is parameter-bit based. This script uses PyTorch fake "
            "quantization for QAT research and does not validate real INT4 deployment."
        ),
    }
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.log("=" * 80)
    logger.log("Block-Level Mixed-Precision INT4 QAT Final Report")
    logger.log(f"Best Test Accuracy: {best_acc:.2f}%")
    logger.log(f"Best Epoch: {best_epoch}")
    logger.log(f"INT8 Blocks: {int8_blocks}")
    logger.log(f"Matched INT8 Modules: {matched_int8_modules}")
    logger.log(f"INT8 Quantizable Modules: {int8_q}/{total_q}")
    logger.log(f"INT4 Quantizable Modules: {int4_q}/{total_q}")
    logger.log(f"FP32 Checkpoint Size: {fp32_size_mb:.2f} MB")
    logger.log(f"Theoretical Mixed-Precision Size: {theoretical_size_mb:.2f} MB")
    logger.log(f"Training history CSV: {history_csv}")
    logger.log(f"JSON summary: {summary_path}")
    logger.log(f"Total Time: {total_time / 60.0:.2f} min")
    logger.log("Note: size is theoretical; this script uses fake quantization.")
    logger.log("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet18 block-level mixed-precision INT4 QAT")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional explicit FP32 checkpoint path")
    parser.add_argument(
        "--int8-blocks",
        required=True,
        help=(
            "Comma-separated ResNet18 units restored to INT8. "
            "Examples: layer3.1,layer2.0,layer2.1,layer3.0 or stem,layer3.1"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epoch", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    train_block_mixed_qat(parse_args())
