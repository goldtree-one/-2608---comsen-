from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from task4_common import (OUTPUT_DIR, Assets, add_asset_argument, check_file,
                          find_asset_dir, load_model, print_device_info,
                          print_memory_info, save_json)

DEFAULT_WORKERS = 2


def parse_label_map(value: str) -> dict[int, int]:
    text = (value or "").strip().lower()
    if text in {"", "none", "off", "no"}:
        return {}
    mapping: dict[int, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        src, dst = part.split(":")
        mapping[int(src)] = int(dst)
    return mapping


def load_dataset_cfg(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    if cfg.get("path"):
        base = (data_yaml.parent / cfg["path"]).resolve()
    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        path = (base / rel).resolve()
        if not path.exists() and rel.startswith("../"):
            path = (base / rel[3:]).resolve()
        splits[split] = path
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return cfg, splits, list(names)


def build_fixed_test(data_yaml: Path, output_dir: Path, mapping: dict[int, int]) -> Path:
    cfg, splits, names = load_dataset_cfg(data_yaml)
    test_images = splits["test"]
    test_labels = test_images.parent / "labels"
    fixed = output_dir / "test_fixed"
    dst_images = fixed / "images"
    dst_labels = fixed / "labels"

    src_images = sorted(p for p in test_images.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in dst_images.iterdir() if p.is_file()}
    for img in src_images:
        if img.name not in existing:
            shutil.copy2(img, dst_images / img.name)

    before: dict[str, int] = {}
    after: dict[str, int] = {}
    n_boxes = 0
    for lbl in sorted(test_labels.glob("*.txt")):
        out_lines = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            before[str(cls)] = before.get(str(cls), 0) + 1
            cls = mapping.get(cls, cls)
            after[str(cls)] = after.get(str(cls), 0) + 1
            parts[0] = str(cls)
            out_lines.append(" ".join(parts))
            n_boxes += 1
        (dst_labels / lbl.name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    fixed_yaml = output_dir / "data_test_fixed.yaml"
    payload = {
        "path": fixed.as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(names),
        "names": names,
    }
    fixed_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                          encoding="utf-8")

    save_json(output_dir / "test_label_map.json", {
        "source_test_images": str(test_images),
        "source_test_labels": str(test_labels),
        "fixed_test_root": str(fixed),
        "mapping": {str(k): v for k, v in mapping.items()},
        "images": len(src_images),
        "boxes": n_boxes,
        "boxes_before": before,
        "boxes_after": after,
        "note": "修正副本只用于评估，不修改原始数据集标签。",
    })
    return fixed_yaml


def count_split_images(data_yaml: Path, split: str) -> int | None:
    _, splits, _ = load_dataset_cfg(data_yaml)
    path = splits.get(split)
    if not path or not path.exists():
        return None
    return len([p for p in path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def evaluate_split(model, data_yaml: Path, split: str, tag: str,
                   args, weights_path: Path, label: str | None = None) -> dict:
    save_dir = args.output / "runs" / f"eval_{tag}"
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=args.plots,
        max_det=args.max_det,
        project=str(args.output / "runs"),
        name=f"eval_{tag}",
        exist_ok=True,
        verbose=False,
    )

    names = {int(k): v for k, v in metrics.names.items()}
    per_class = {name: {"ap50": None, "ap50_95": None, "precision": None, "recall": None}
                 for name in names.values()}
    nt_per_class = metrics.nt_per_class
    if nt_per_class is not None:
        for i, name in names.items():
            if i < len(nt_per_class):
                per_class[name]["instances"] = int(nt_per_class[i])
    nt_per_image = metrics.nt_per_image
    if nt_per_image is not None:
        for i, name in names.items():
            if i < len(nt_per_image):
                per_class[name]["images_with_class"] = int(nt_per_image[i])

    for idx, ap50, ap, precision, recall in zip(
            metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap,
            metrics.box.p, metrics.box.r):
        name = names[int(idx)]
        per_class[name].update({
            "ap50": round(float(ap50), 4),
            "ap50_95": round(float(ap), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        })

    row = {
        "tag": tag,
        "label": label or split,
        "split": split,
        "data": str(data_yaml),
        "weights": str(weights_path),
        "images": count_split_images(data_yaml, split),
        "instances": int(np.sum(nt_per_class)) if nt_per_class is not None else None,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 3) for k, v in metrics.speed.items()},
    }

    metrics_dir = args.output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / f"{tag}_metrics.json", row)

    for src_name, dst_name in [
        ("confusion_matrix.png", f"confusion_matrix_{tag}.png"),
        ("confusion_matrix_normalized.png", f"confusion_matrix_{tag}_normalized.png"),
        ("BoxPR_curve.png", f"pr_curve_{tag}.png"),
    ]:
        src = save_dir / src_name
        if src.exists():
            shutil.copy2(src, metrics_dir / dst_name)

    print(f"[评估] {label or split:<8} mAP50={row['map50']:.4f}  "
          f"mAP50-95={row['map50_95']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")
    return row


def plot_map_bars(rows: list[dict], path: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    b1 = ax.bar(x - width / 2, [r["map50"] for r in rows], width,
                label="mAP50", color="#4C78A8")
    b2 = ax.bar(x + width / 2, [r["map50_95"] for r in rows], width,
                label="mAP50-95", color="#F58518")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP")
    ax.set_title("Detection mAP by split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[图] {path}")


def plot_per_class(rows: list[dict], path: Path) -> None:
    class_names = []
    for row in rows:
        for name in row["per_class"]:
            if name not in class_names:
                class_names.append(name)
    if not class_names:
        return
    x = np.arange(len(class_names))
    width = 0.8 / max(len(rows), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)
    for ax, metric, title in zip(axes, ("ap50", "ap50_95"), ("mAP50", "mAP50-95")):
        for i, row in enumerate(rows):
            values = [row["per_class"].get(name, {}).get(metric) or 0.0 for name in class_names]
            offset = (i - (len(rows) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=row["label"])
            for bar in bars:
                if bar.get_height() > 0:
                    ax.annotate(f"{bar.get_height():.2f}",
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                textcoords="offset points", xytext=(0, 2),
                                ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title)
        ax.set_title(f"Per-class {title}")
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    print(f"[图] {path}")


def plot_training_curve(results_csv: Path, path: Path) -> None:
    if not results_csv or not results_csv.exists():
        print(f"[跳过] 没找到 {results_csv}，无法绘制训练过程 mAP 曲线")
        return
    epochs, map50, map95 = [], [], []
    with results_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(float(row["epoch"])))
                map50.append(float(row["metrics/mAP50(B)"]))
                map95.append(float(row["metrics/mAP50-95(B)"]))
            except (KeyError, ValueError):
                continue
    if not epochs:
        print(f"[跳过] {results_csv} 中没有 mAP 列")
        return

    best_idx = int(np.argmax(map95))
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    ax.plot(epochs, map50, label="mAP50", color="#4C78A8", linewidth=2)
    ax.plot(epochs, map95, label="mAP50-95", color="#F58518", linewidth=2)
    ax.scatter([epochs[best_idx]], [map95[best_idx]], color="#D62728", zorder=5)
    ax.annotate(f"best mAP50-95={map95[best_idx]:.3f}\nepoch {epochs[best_idx]}",
                (epochs[best_idx], map95[best_idx]),
                textcoords="offset points", xytext=(-95, 38), fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "#D62728"})
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, max(epochs) + 8)
    ax.set_title("Validation mAP during training")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[图] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4-4 训练与 train/val/test mAP 评估")
    add_asset_argument(parser)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"输出目录，默认 {OUTPUT_DIR}")
    parser.add_argument("--data", type=Path, default=None,
                        help="data.yaml 路径；不填则用素材目录里的那份")
    parser.add_argument("--weights", type=Path, default=None,
                        help="训练起点或评估权重；训练默认用 drone.pt，评估默认用 best.pt")
    parser.add_argument("--device", default=None, help="'0' 用 GPU，'cpu' 用 CPU，不填自动")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，默认 640")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--batch", type=int, default=16, help="批大小，默认 16")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"DataLoader 进程数，默认 {DEFAULT_WORKERS}")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值，默认 30")
    parser.add_argument("--name", default="drone_yolo11n", help="训练输出子目录名")
    parser.add_argument("--eval-only", action="store_true",
                        help="跳过训练，直接用 --weights 评估")
    parser.add_argument("--eval-splits", default="train,val,test",
                        help="要评估的划分，逗号分隔，默认 train,val,test")
    parser.add_argument("--test-label-map", default="0:1",
                        help="test 标签类别修正，默认 0:1（本数据集 test 图片是无人机但标签为 0）；"
                             "如标签正确请传 none")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="每张图最多保留的检测框数，默认 1000（避免 300 截断）")
    parser.add_argument("--results-csv", type=Path, default=None,
                        help="训练过程 results.csv；默认从权重目录自动查找")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                        help="是否让 Ultralytics 生成混淆矩阵/PR 曲线，默认是")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 70)
    print("Task4-4 训练与 mAP 评估")
    print("=" * 70)

    assets = Assets(find_asset_dir(args.asset_dir))
    assets.describe()
    print_device_info(args.device)

    data_yaml = check_file(args.data or assets.dataset_yaml,
                           "请确认素材目录下有 dataset/drone_and_bird/data.yaml")
    cfg, splits, names = load_dataset_cfg(data_yaml)
    print(f"[数据] {data_yaml}")
    print(f"[数据] nc={len(names)}, names={names}")
    for split, path in splits.items():
        n = len(list(path.glob("*"))) if path.exists() else 0
        print(f"[数据] {split:<5} -> {path}  ({n} 个文件)")

    label_map = parse_label_map(args.test_label_map)
    print(f"[测试标签] 映射={label_map or '不修正'}")

    if not args.eval_only:
        model = load_model(args.weights or assets.drone_weights, "drone.pt 应在素材目录下")
        print(f"\n[训练] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
              f"device={args.device or 'auto'} workers={args.workers}")
        print_memory_info(args.workers)
        try:
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                patience=args.patience,
                project=str(args.output / "runs"),
                name=args.name,
                exist_ok=True,
                seed=0,
                pretrained=True,
                optimizer="auto",
                plots=True,
                val=True,
                max_det=args.max_det,
            )
        except (MemoryError, SystemError) as exc:
            print(f"\n[错误] 训练中断：{type(exc).__name__}: {exc}")
            print("       可尝试 --workers 0 --batch 8")
            return

    best = args.weights or (args.output / "runs" / args.name / "weights" / "best.pt")
    best = check_file(best, "训练应产出 best.pt；或用 --weights 指定已有权重")

    model = load_model(best, "评估权重")
    print(f"[评估] 权重={best}")

    split_list = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    rows: list[dict] = []
    raw_test = None

    for split in split_list:
        if split == "test":
            raw_test = evaluate_split(model, data_yaml, "test", "test_raw", args,
                                      weights_path=best, label="test(raw)")
            if label_map:
                fixed_yaml = build_fixed_test(data_yaml, args.output, label_map)
                row = evaluate_split(model, fixed_yaml, "test", "test", args,
                                     weights_path=best, label="test(fixed)")
            else:
                row = raw_test
            rows.append(row)
        else:
            rows.append(evaluate_split(model, data_yaml, split, split, args,
                                       weights_path=best))

    metrics_dir = args.output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / "metrics_summary.json", {
        "weights": str(best),
        "data": str(data_yaml),
        "test_label_map": {str(k): v for k, v in label_map.items()},
        "results": rows,
        "test_raw": raw_test,
    })

    plot_map_bars(rows, metrics_dir / "eval_map_bars.png")
    plot_per_class(rows, metrics_dir / "eval_map_per_class.png")

    results_csv = args.results_csv or (Path(best).parent.parent / "results.csv")
    plot_training_curve(results_csv, metrics_dir / "train_map_curve.png")

    print("\n" + "=" * 70)
    print("【mAP 汇总】")
    print("=" * 70)
    print(f"{'split':<12}{'mAP50':>10}{'mAP50-95':>12}{'P':>10}{'R':>10}")
    for row in rows:
        print(f"{row['label']:<12}{row['map50']:>10.4f}{row['map50_95']:>12.4f}"
              f"{row['precision']:>10.4f}{row['recall']:>10.4f}")
    if raw_test is not None and label_map:
        print("\n[注意] test(raw) 使用原始标签，test(fixed) 使用修正后的标签；"
              "报告中请写清用的是哪一个。")
    print(f"\n[保存] 指标与图片 -> {metrics_dir}")


if __name__ == "__main__":
    main()
