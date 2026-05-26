"""
Layer sensitivity analysis for INT4 pseudo quantization.

This script evaluates how sensitive each Conv2d / Linear layer is to INT4
fake quantization under a specific model + dataset combination. It is intended
for analysis, not for real INT4 deployment.

Recommended experiments:
  1. MobileNetV2 + CIFAR-10
  2. MobileNetV2 + CIFAR-100
  3. ResNet18    + CIFAR-100

Example:
  python scripts/int4/layer_sensitivity_analysis.py --model resnet18 --dataset cifar100 --split val --val-size 5000
  python scripts/int4/layer_sensitivity_analysis.py --model mobilenetv2 --dataset cifar10 --split val --val-size 5000
  python scripts/int4/layer_sensitivity_analysis.py --model mobilenetv2 --dataset cifar100 --split val --val-size 5000
"""

import argparse
import copy
import csv
import datetime
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.models.quantization import mobilenet_v2 as quant_mobilenet_v2


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


class Int4FakeQuantWrapper(nn.Module):
    """Wrap a layer and fake-quantize its weight to signed INT4 during forward.

    The wrapped module still runs with floating-point tensors. This is a
    sensitivity-analysis tool: it simulates the numerical error caused by INT4
    quantization for one layer at a time.
    """

    def __init__(self, module: nn.Module, per_channel: bool = True) -> None:
        super().__init__()
        self.module = module
        self.per_channel = per_channel

    @staticmethod
    def _fake_quant_weight_int4(weight: torch.Tensor, per_channel: bool = True) -> torch.Tensor:
        qmin, qmax = -8, 7
        if per_channel and weight.dim() >= 2:
            reduce_dims = tuple(range(1, weight.dim()))
            max_abs = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_abs = weight.detach().abs().max()
        scale = torch.clamp(max_abs / qmax, min=1e-8)
        qweight = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return qweight * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qweight = self._fake_quant_weight_int4(self.module.weight, self.per_channel)
        if isinstance(self.module, nn.Conv2d):
            return nn.functional.conv2d(
                x,
                qweight,
                self.module.bias,
                self.module.stride,
                self.module.padding,
                self.module.dilation,
                self.module.groups,
            )
        if isinstance(self.module, nn.Linear):
            return nn.functional.linear(x, qweight, self.module.bias)
        raise TypeError(f"Unsupported wrapped module type: {type(self.module)}")


def get_project_paths() -> Tuple[str, str, str]:
    if os.path.exists("/root/autodl-tmp"):
        data_dir = "/root/autodl-tmp/data"
        model_dir = "/root/autodl-tmp/my_backup"
    else:
        data_dir = "data"
        model_dir = "models"
    result_dir = os.path.join("results", "int4_sensitivity")
    os.makedirs(result_dir, exist_ok=True)
    return data_dir, model_dir, result_dir


def build_model(model_name: str, dataset_name: str) -> nn.Module:
    num_classes = 10 if dataset_name == "cifar10" else 100
    if model_name == "resnet18":
        model = models.quantization.resnet18(weights=None, quantize=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenetv2":
        # Quantizable MobileNetV2 keeps naming closer to QAT scripts.
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
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=True)


def build_dataloader(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    input_size: int,
    split: str,
    val_size: int,
    val_seed: int,
    max_samples: int,
) -> DataLoader:
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        dataset_class = datasets.CIFAR10
    elif dataset_name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if split == "test":
        dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)
    elif split == "val":
        train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
        if val_size <= 0 or val_size >= len(train_dataset):
            raise ValueError(f"val_size must be in [1, {len(train_dataset) - 1}], got {val_size}")
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(val_seed)
        _, dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    else:
        raise ValueError(f"Unsupported split: {split}")

    if max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def collect_quantizable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


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


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def run_sensitivity_analysis(args: argparse.Namespace) -> str:
    data_dir, model_dir, result_dir = get_project_paths()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataloader = build_dataloader(
        args.dataset,
        data_dir,
        args.batch_size,
        args.num_workers,
        args.input_size,
        args.split,
        args.val_size,
        args.val_seed,
        args.max_samples,
    )
    checkpoint_path = resolve_checkpoint(model_dir, args.model, args.dataset, args.checkpoint)

    base_model = build_model(args.model, args.dataset)
    load_state_dict_safely(base_model, checkpoint_path)
    base_model.to(device)
    base_model.eval()

    print(f"[INFO] Model: {args.model} | Dataset: {args.dataset} | Split: {args.split} | Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    if args.split == "val":
        print(f"[INFO] Validation split is sampled from training set | val_size={args.val_size} | seed={args.val_seed}")
    print(f"[INFO] Evaluation samples: {len(dataloader.dataset)} | Input size: {args.input_size}")

    baseline_start = time.time()
    baseline_acc = evaluate(base_model, dataloader, device)
    print(f"[INFO] FP32 baseline accuracy: {baseline_acc:.2f}% | Time: {(time.time() - baseline_start):.1f}s")

    layers = collect_quantizable_layers(base_model)
    print(f"[INFO] Quantizable layers: {len(layers)}")

    rows: List[Dict[str, object]] = []
    for idx, (layer_name, layer_module) in enumerate(layers, start=1):
        test_model = copy.deepcopy(base_model).to(device)
        original_layer = get_module_by_name(test_model, layer_name)
        wrapped_layer = Int4FakeQuantWrapper(original_layer, per_channel=not args.per_tensor)
        set_module_by_name(test_model, layer_name, wrapped_layer)

        start = time.time()
        acc = evaluate(test_model, dataloader, device)
        elapsed = time.time() - start
        drop = baseline_acc - acc

        row = {
            "rank_placeholder": 0,
            "layer_index": idx,
            "layer_name": layer_name,
            "layer_type": layer_module.__class__.__name__,
            "parameters": parameter_count(layer_module),
            "baseline_acc": round(baseline_acc, 4),
            "int4_single_layer_acc": round(acc, 4),
            "accuracy_drop": round(drop, 4),
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(f"[{idx:03d}/{len(layers):03d}] {layer_name:<45} acc={acc:6.2f}% drop={drop:6.2f}% time={elapsed:5.1f}s")

    ranked_rows = sorted(rows, key=lambda x: float(x["accuracy_drop"]), reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_placeholder"] = rank

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_tag = "full" if args.max_samples <= 0 else f"n{args.max_samples}"
    split_tag = f"val{args.val_size}_seed{args.val_seed}" if args.split == "val" else "test"
    output_path = os.path.join(result_dir, f"{args.model}_{args.dataset}_{split_tag}_int4_layer_sensitivity_{sample_tag}_{timestamp}.csv")

    fieldnames = [
        "rank_placeholder",
        "layer_index",
        "layer_name",
        "layer_type",
        "parameters",
        "baseline_acc",
        "int4_single_layer_acc",
        "accuracy_drop",
        "elapsed_seconds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    print("\n[INFO] Top sensitive layers:")
    for row in ranked_rows[: args.topk]:
        print(
            f"  rank={row['rank_placeholder']:>2} | {row['layer_name']:<45} "
            f"drop={row['accuracy_drop']:>6.2f}% | acc={row['int4_single_layer_acc']:>6.2f}%"
        )
    print(f"[INFO] CSV saved to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT4 layer sensitivity analysis")
    parser.add_argument("--model", choices=["resnet18", "mobilenetv2"], required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional explicit FP32 checkpoint path")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Use val to split validation samples from CIFAR training set, or test to use official test set")
    parser.add_argument("--val-size", type=int, default=5000, help="Number of samples split from CIFAR training set when --split val")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for train/validation split")
    parser.add_argument("--max-samples", type=int, default=-1, help="Use <=0 for all samples in the selected split")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--per-tensor", action="store_true", help="Use per-tensor INT4 weight simulation instead of per-channel")
    return parser.parse_args()


if __name__ == "__main__":
    run_sensitivity_analysis(parse_args())
