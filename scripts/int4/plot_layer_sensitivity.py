import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def detect_column(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        low = col.lower()
        for cand in candidates:
            if cand.lower() in low:
                return col
    raise ValueError(f"Cannot detect columns from {candidates}. Available columns: {list(df.columns)}")


def safe_name(text):
    return re.sub(r"[^\w\-_.]+", "_", str(text)).strip("_") or "layer_sensitivity"


def short_name(name, max_len=38):
    name = str(name)
    return name if len(name) <= max_len else "..." + name[-(max_len - 3):]


def read_csv(path):
    df = pd.read_csv(path)
    layer_col = detect_column(df, ["layer_name", "layer", "name", "module_name", "module"])
    drop_col = detect_column(df, ["accuracy_drop", "acc_drop", "drop", "sensitivity", "accuracy_loss"])
    out = pd.DataFrame({
        "layer_name": df[layer_col].astype(str),
        "accuracy_drop": pd.to_numeric(df[drop_col], errors="coerce")
    }).dropna()
    out = out.sort_values("accuracy_drop", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def plot_topk(df, path, title, topk):
    top = df.head(topk).iloc[::-1]
    labels = [short_name(x) for x in top["layer_name"]]
    plt.figure(figsize=(11, max(5, 0.35 * len(top) + 1.5)))
    plt.barh(labels, top["accuracy_drop"])
    plt.xlabel("Accuracy Drop (percentage points)")
    plt.ylabel("Layer")
    plt.title(f"{title} - Top {len(top)} Sensitive Layers")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_curve(df, path, title):
    plt.figure(figsize=(11, 5.5))
    plt.plot(range(1, len(df) + 1), df["accuracy_drop"], marker="o", linewidth=1)
    plt.xlabel("Layer Rank by Sensitivity")
    plt.ylabel("Accuracy Drop (percentage points)")
    plt.title(f"{title} - Full Layer Sensitivity Ranking")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cumulative(df, path, title):
    drops = df["accuracy_drop"].clip(lower=0)
    total = drops.sum()
    y = drops.cumsum() / total * 100 if total > 0 else drops.cumsum()
    plt.figure(figsize=(11, 5.5))
    plt.plot(range(1, len(df) + 1), y, marker="o", linewidth=1)
    plt.xlabel("Top-k Sensitive Layers")
    plt.ylabel("Cumulative Contribution (%)" if total > 0 else "Cumulative Accuracy Drop")
    plt.title(f"{title} - Cumulative Sensitivity Contribution")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from INT4 layer sensitivity CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = read_csv(csv_path)
    title = args.title or csv_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "figures" / safe_name(csv_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    topk = min(args.topk, len(df))
    df.to_csv(out_dir / "layer_sensitivity_ranked.csv", index=False, encoding="utf-8-sig")
    plot_topk(df, out_dir / f"top{topk}_layer_sensitivity_bar.png", title, topk)
    plot_curve(df, out_dir / "full_layer_sensitivity_curve.png", title)
    plot_cumulative(df, out_dir / "cumulative_sensitivity_curve.png", title)

    print("Saved:")
    print(out_dir / "layer_sensitivity_ranked.csv")
    print(out_dir / f"top{topk}_layer_sensitivity_bar.png")
    print(out_dir / "full_layer_sensitivity_curve.png")
    print(out_dir / "cumulative_sensitivity_curve.png")


if __name__ == "__main__":
    main()
