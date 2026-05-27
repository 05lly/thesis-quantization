"""
ResNet18 block-level recovery sensitivity analysis for INT4/INT8 fake quantization.

Purpose
-------
This script evaluates which ResNet18 structural unit is most useful to restore
from INT4 fake quantization to INT8 fake quantization.

It answers this question:
    In an all-INT4 fake-quantized ResNet18, which residual block or structural
    unit gives the largest accuracy recovery when it is assigned INT8 instead?

This is different from destructive layer sensitivity:
    destructive: high precision model + one layer INT4 -> accuracy drop
    recovery:    all INT4 model + one block INT8      -> accuracy gain

Outputs
-------
The script creates a timestamped experiment directory containing:
    1. block_recovery_ranking.csv
    2. experiment_summary.json
    3. experiment.log
    4. selected_topk_units.txt

Example
-------
python scripts/int4/resnet18_block_recovery_sensitivity.py \
    --dataset cifar100 \
    --split val \
    --val-size 5000 \
    --calib-samples 1024 \
    --batch-size 128 \
    --topk 3

Optional short QAT search per candidate:
python scripts/int4/resnet18_block_recovery_sensitivity.py \
    --dataset cifar100 \
    --finetune-epochs 1 \
    --calib-samples 1024

Important
---------
This script uses PyTorch fake quantization. It is intended for experimental
analysis of INT4 numerical effects. It does not produce a real deployable INT4
model and does not guarantee real INT4 speedup.
"""

import argparse
import csv
import datetime
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import FakeQuantize, QConfig
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


@dataclass
class RecoveryUnit:
    """A structural unit whose qconfig can be restored to INT8."""

    unit_name: str
    unit_type: str
    module_prefixes: List[str]
    module_names: List[str]
    parameter_count: int


class ExperimentRecorder:
    """Minimal experiment logger that writes both stdout and a log file."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_path = os.path.join(self.output_dir, "experiment.log")

    def log(self, message: str) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {message}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Paths, model, checkpoint, and data
# ---------------------------------------------------------------------------


def get_project_paths() -> Tuple[str, str, str]:
    """Match the path style used by the existing repository scripts."""
    if os.path.exists("/root/autodl-tmp"):
        data_dir = "/root/autodl-tmp/data"
        model_dir = "/root/autodl-tmp/my_backup"
    else:
        data_dir = "data"
        model_dir = "models"

    result_dir = os.path.join("results", "int4_block_recovery")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return data_dir, model_dir, result_dir


def get_int4_qat_qconfig() -> QConfig:
    """INT4 fake quantization qconfig: W4A4, per-channel symmetric weight."""
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
    """INT8 fake quantization qconfig: W8A8, per-channel symmetric weight."""
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


def build_datasets(dataset_name: str, data_dir: str, input_size: int):
    if dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_eval_set = dataset_class(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    train_aug_set = dataset_class(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = dataset_class(root=data_dir, train=False, download=True, transform=eval_transform)
    return train_eval_set, train_aug_set, test_set


def make_subset(dataset, max_samples: int):
    if max_samples > 0 and max_samples < len(dataset):
        return Subset(dataset, list(range(max_samples)))
    return dataset


def build_loaders(args: argparse.Namespace, data_dir: str):
    train_eval_set, train_aug_set, test_set = build_datasets(
        args.dataset, data_dir, args.input_size
    )

    if args.split == "test":
        eval_set = test_set
    elif args.split == "val":
        if args.val_size <= 0 or args.val_size >= len(train_eval_set):
            raise ValueError(
                f"val_size must be in [1, {len(train_eval_set) - 1}], got {args.val_size}"
            )
        train_size = len(train_eval_set) - args.val_size
        generator = torch.Generator().manual_seed(args.val_seed)
        _, eval_set = random_split(train_eval_set, [train_size, args.val_size], generator)
    else:
        raise ValueError(f"Unsupported split: {args.split}")

    eval_set = make_subset(eval_set, args.max_eval_samples)
    calib_set = make_subset(train_eval_set, args.calib_samples)
    train_set = make_subset(train_aug_set, args.max_train_samples)

    pin_memory = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    calib_loader = DataLoader(
        calib_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, calib_loader, eval_loader


# ---------------------------------------------------------------------------
# Structural units and qconfig assignment
# ---------------------------------------------------------------------------


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def is_under_prefix(module_name: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return True
    return False


def count_own_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def count_params_for_prefixes(model: nn.Module, prefixes: Sequence[str]) -> int:
    """Count parameters under prefixes without double-counting shared parameter objects."""
    seen = set()
    total = 0
    for name, module in model.named_modules():
        if is_under_prefix(name, prefixes):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


def collect_module_names_for_prefixes(model: nn.Module, prefixes: Sequence[str]) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if not name:
            continue
        if is_under_prefix(name, prefixes) and isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def collect_resnet18_recovery_units(
    model: nn.Module,
    include_stem: bool = True,
    include_classifier: bool = True,
) -> List[RecoveryUnit]:
    """Return ResNet18 structural units: optional stem, 8 residual blocks, optional fc."""
    units: List[RecoveryUnit] = []

    if include_stem:
        prefixes = ["conv1"]
        units.append(
            RecoveryUnit(
                unit_name="stem",
                unit_type="stem_conv",
                module_prefixes=prefixes,
                module_names=collect_module_names_for_prefixes(model, prefixes),
                parameter_count=count_params_for_prefixes(model, prefixes),
            )
        )

    for layer_idx in range(1, 5):
        layer = getattr(model, f"layer{layer_idx}")
        for block_idx in range(len(layer)):
            unit_name = f"layer{layer_idx}.{block_idx}"
            prefixes = [unit_name]
            units.append(
                RecoveryUnit(
                    unit_name=unit_name,
                    unit_type="residual_block",
                    module_prefixes=prefixes,
                    module_names=collect_module_names_for_prefixes(model, prefixes),
                    parameter_count=count_params_for_prefixes(model, prefixes),
                )
            )

    if include_classifier:
        prefixes = ["fc"]
        units.append(
            RecoveryUnit(
                unit_name="classifier",
                unit_type="fc",
                module_prefixes=prefixes,
                module_names=collect_module_names_for_prefixes(model, prefixes),
                parameter_count=count_params_for_prefixes(model, prefixes),
            )
        )

    return units


def fuse_model_if_supported(model: nn.Module, is_qat: bool = True) -> None:
    if hasattr(model, "fuse_model"):
        model.eval()
        model.fuse_model(is_qat=is_qat)


def assign_block_recovery_qconfig(
    model: nn.Module,
    int8_units: Sequence[RecoveryUnit],
) -> Dict[str, str]:
    """Assign INT4 to all modules, then INT8 to the selected units."""
    int4_qconfig = get_int4_qat_qconfig()
    int8_qconfig = get_int8_qat_qconfig()

    model.qconfig = int4_qconfig
    int8_prefixes: List[str] = []
    for unit in int8_units:
        int8_prefixes.extend(unit.module_prefixes)

    assignment: Dict[str, str] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if is_under_prefix(name, int8_prefixes):
            module.qconfig = int8_qconfig
            assignment[name] = "int8"
        else:
            module.qconfig = int4_qconfig
            assignment[name] = "int4"

    return assignment


def estimate_quantized_size_mb(model: nn.Module, int8_units: Sequence[RecoveryUnit]) -> float:
    """Theoretical parameter size: selected units use 8 bit, other params use 4 bit."""
    int8_prefixes: List[str] = []
    for unit in int8_units:
        int8_prefixes.extend(unit.module_prefixes)

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


# ---------------------------------------------------------------------------
# Evaluation and optional short QAT finetuning
# ---------------------------------------------------------------------------


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


def calibrate_fake_quant_observers(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> None:
    """Run data through the fake-quant model so activation observers get ranges."""
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs)


def freeze_quant_observers_and_bn(model: nn.Module) -> None:
    model.apply(torch.ao.quantization.disable_observer)
    try:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    except Exception:
        # Some PyTorch versions or modules may not expose freeze_bn_stats for all modules.
        pass


def finetune_qat_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    freeze_epoch: int,
    recorder: ExperimentRecorder,
    tag: str,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        if epoch >= freeze_epoch:
            freeze_quant_observers_and_bn(model)

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"{tag} epoch {epoch + 1}/{epochs}", leave=False)
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
        eval_acc = evaluate(model, eval_loader, device)
        best_acc = max(best_acc, eval_acc)
        recorder.log(
            f"[{tag}] epoch={epoch + 1:02d}/{epochs:02d} "
            f"train_acc={train_acc:.2f}% eval_acc={eval_acc:.2f}% "
            f"loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )

    return best_acc


def build_prepared_candidate_model(
    dataset_name: str,
    checkpoint_path: str,
    device: torch.device,
    int8_units: Sequence[RecoveryUnit],
) -> nn.Module:
    model = build_resnet18(dataset_name)
    load_state_dict_safely(model, checkpoint_path)
    model.to(device)

    # Fuse before qconfig assignment to match the existing QAT style.
    fuse_model_if_supported(model, is_qat=True)
    model.train()
    assign_block_recovery_qconfig(model, int8_units)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def evaluate_candidate_config(
    args: argparse.Namespace,
    checkpoint_path: str,
    train_loader: DataLoader,
    calib_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    int8_units: Sequence[RecoveryUnit],
    recorder: ExperimentRecorder,
    tag: str,
) -> Tuple[float, float]:
    start = time.time()
    model = build_prepared_candidate_model(args.dataset, checkpoint_path, device, int8_units)

    if args.calib_samples != 0:
        calibrate_fake_quant_observers(
            model,
            calib_loader,
            device,
            max_batches=args.calib_batches,
        )
        if args.freeze_after_calib:
            freeze_quant_observers_and_bn(model)

    if args.finetune_epochs > 0:
        acc = finetune_qat_model(
            model,
            train_loader,
            eval_loader,
            device,
            epochs=args.finetune_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            freeze_epoch=args.freeze_epoch,
            recorder=recorder,
            tag=tag,
        )
    else:
        acc = evaluate(model, eval_loader, device)

    elapsed = time.time() - start
    return acc, elapsed


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_block_recovery_sensitivity(args: argparse.Namespace) -> None:
    data_dir, model_dir, result_root = get_project_paths()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        result_root,
        f"resnet18_{args.dataset}_{args.split}_block_recovery_{timestamp}",
    )
    recorder = ExperimentRecorder(output_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.quantized.engine = "qnnpack"

    checkpoint_path = resolve_checkpoint(model_dir, args.dataset, args.checkpoint)
    train_loader, calib_loader, eval_loader = build_loaders(args, data_dir)

    structure_model = build_resnet18(args.dataset)
    load_state_dict_safely(structure_model, checkpoint_path)
    units = collect_resnet18_recovery_units(
        structure_model,
        include_stem=not args.exclude_stem,
        include_classifier=not args.exclude_classifier,
    )

    recorder.log("=" * 80)
    recorder.log("ResNet18 Block-Level Recovery Sensitivity Analysis")
    recorder.log(f"Dataset: {args.dataset}")
    recorder.log(f"Split: {args.split}")
    recorder.log(f"Device: {device}")
    recorder.log(f"Checkpoint: {checkpoint_path}")
    recorder.log(f"Output directory: {output_dir}")
    recorder.log(f"Evaluation samples: {len(eval_loader.dataset)}")
    recorder.log(f"Calibration samples: {len(calib_loader.dataset)}")
    recorder.log(f"Candidate recovery units: {len(units)}")
    recorder.log(f"Optional candidate finetune epochs: {args.finetune_epochs}")
    recorder.log("This script uses fake quantization; it does not validate real INT4 deployment.")
    recorder.log("=" * 80)

    fp32_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    all_int4_size_mb = estimate_quantized_size_mb(structure_model, int8_units=[])

    # Baseline: all units remain INT4.
    recorder.log("[BASELINE] Evaluating all-INT4 fake-quantized model...")
    all_int4_acc, all_int4_elapsed = evaluate_candidate_config(
        args=args,
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        calib_loader=calib_loader,
        eval_loader=eval_loader,
        device=device,
        int8_units=[],
        recorder=recorder,
        tag="all_int4",
    )
    recorder.log(
        f"[BASELINE] all_INT4_acc={all_int4_acc:.4f}% "
        f"size={all_int4_size_mb:.4f}MB elapsed={all_int4_elapsed:.2f}s"
    )

    rows: List[Dict[str, object]] = []
    for idx, unit in enumerate(units, start=1):
        recorder.log(
            f"[CANDIDATE {idx:02d}/{len(units):02d}] Restore unit to INT8: "
            f"{unit.unit_name} | modules={unit.module_names}"
        )
        candidate_acc, elapsed = evaluate_candidate_config(
            args=args,
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            calib_loader=calib_loader,
            eval_loader=eval_loader,
            device=device,
            int8_units=[unit],
            recorder=recorder,
            tag=f"restore_{unit.unit_name}",
        )

        mixed_size_mb = estimate_quantized_size_mb(structure_model, int8_units=[unit])
        size_increase_mb = mixed_size_mb - all_int4_size_mb
        recovery_gain = candidate_acc - all_int4_acc

        row = {
            "rank": 0,
            "unit_index": idx,
            "unit_name": unit.unit_name,
            "unit_type": unit.unit_type,
            "module_prefixes": ";".join(unit.module_prefixes),
            "module_names": ";".join(unit.module_names),
            "parameters": unit.parameter_count,
            "all_int4_acc": round(all_int4_acc, 4),
            "recovered_acc": round(candidate_acc, 4),
            "recovery_gain": round(recovery_gain, 4),
            "all_int4_size_mb": round(all_int4_size_mb, 6),
            "mixed_size_mb": round(mixed_size_mb, 6),
            "size_increase_mb": round(size_increase_mb, 6),
            "gain_per_mb": round(recovery_gain / size_increase_mb, 6) if size_increase_mb > 0 else 0.0,
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)

        recorder.log(
            f"[RESULT] {unit.unit_name:<12} acc={candidate_acc:.4f}% "
            f"gain={recovery_gain:+.4f}% size+={size_increase_mb:.6f}MB "
            f"elapsed={elapsed:.2f}s"
        )

    ranked_rows = sorted(rows, key=lambda r: float(r["recovery_gain"]), reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    csv_path = os.path.join(output_dir, "block_recovery_ranking.csv")
    write_csv(csv_path, ranked_rows)

    topk_rows = ranked_rows[: max(1, args.topk)]
    selected_txt_path = os.path.join(output_dir, "selected_topk_units.txt")
    with open(selected_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top-{len(topk_rows)} units by block-level recovery gain\n")
        f.write(f"All-INT4 baseline accuracy: {all_int4_acc:.4f}%\n\n")
        for row in topk_rows:
            f.write(
                f"rank={row['rank']} unit={row['unit_name']} "
                f"gain={row['recovery_gain']} acc={row['recovered_acc']} "
                f"modules={row['module_names']}\n"
            )

    summary = {
        "experiment": "resnet18_block_recovery_sensitivity",
        "timestamp": timestamp,
        "args": vars(args),
        "paths": {
            "checkpoint": checkpoint_path,
            "output_dir": output_dir,
            "csv": csv_path,
            "log": recorder.log_path,
            "selected_topk_units": selected_txt_path,
        },
        "baseline": {
            "all_int4_accuracy": round(all_int4_acc, 4),
            "fp32_checkpoint_size_mb": round(fp32_size_mb, 6),
            "theoretical_all_int4_param_size_mb": round(all_int4_size_mb, 6),
            "elapsed_seconds": round(all_int4_elapsed, 2),
        },
        "top_units": topk_rows,
        "all_units": ranked_rows,
        "method_note": (
            "Recovery gain is computed as Acc(all INT4 + one unit INT8) - Acc(all INT4). "
            "The experiment uses PyTorch fake quantization for analysis, not real INT4 deployment."
        ),
    }
    json_path = os.path.join(output_dir, "experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    recorder.log("=" * 80)
    recorder.log("Final ranking by block-level recovery gain:")
    for row in ranked_rows[: args.topk]:
        recorder.log(
            f"rank={row['rank']:>2} | {row['unit_name']:<12} "
            f"gain={row['recovery_gain']:>+8.4f}% | "
            f"acc={row['recovered_acc']:>8.4f}% | "
            f"modules={row['module_names']}"
        )
    recorder.log(f"CSV saved: {csv_path}")
    recorder.log(f"JSON summary saved: {json_path}")
    recorder.log(f"Selected top-k units saved: {selected_txt_path}")
    recorder.log(f"Log saved: {recorder.log_path}")
    recorder.log("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet18 block-level recovery sensitivity for INT4/INT8 fake quantization"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional explicit FP32 checkpoint path")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--input-size", type=int, default=224)

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Use val split from CIFAR training set or official CIFAR test set",
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=-1,
        help="Use <=0 for all samples in selected eval split",
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=1024,
        help="Number of training samples used to calibrate fake-quant observers; set 0 to skip",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=-1,
        help="Maximum calibration batches; <=0 means all calib samples",
    )
    parser.add_argument(
        "--freeze-after-calib",
        action="store_true",
        default=True,
        help="Freeze observers after calibration before evaluation",
    )

    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="Optional short QAT finetuning epochs for each candidate; 0 means eval-only ranking",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=-1,
        help="Optional cap for training samples when finetune-epochs > 0",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epoch", type=int, default=1)

    parser.add_argument("--topk", type=int, default=5, help="How many top units to print/save")
    parser.add_argument("--exclude-stem", action="store_true", help="Do not evaluate conv1 as a unit")
    parser.add_argument(
        "--exclude-classifier", action="store_true", help="Do not evaluate fc as a unit"
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_block_recovery_sensitivity(parse_args())
