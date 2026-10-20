from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from task4_common import OUTPUT_DIR, add_asset_argument, find_asset_dir, save_json

CLASS_NAMES = ["bird", "drone"]
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def parse_label_file(path: Path):
    boxes = []
    n_detection = 0
    n_polygon = 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, n_detection, n_polygon
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        n = len(parts)
        if n == 0:
            continue
        cls = int(float(parts[0]))
        if n == 5:
            xc, yc, w, h = (float(v) for v in parts[1:])
            n_detection += 1
        elif n >= 7 and (n - 1) % 2 == 0:
            coords = [float(v) for v in parts[1:]]
            xs, ys = coords[0::2], coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            xc = (x_min + x_max) / 2
            yc = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            n_polygon += 1
        else:
            raise ValueError(f"{path.name} 第 {line_no} 行列数异常（{n} 列）：{line!r}")
        boxes.append((cls, xc, yc, w, h))
    return boxes, n_detection, n_polygon


def find_data_root(explicit: Path | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    assets = find_asset_dir(None)
    candidate = assets / "dataset" / "drone_and_bird"
    return candidate.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4-1 数据集校验与统计")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="数据集目录（含 data.yaml）；不填则自动探测")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"结果输出目录，默认 {OUTPUT_DIR}")
    add_asset_argument(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = find_data_root(args.data_root)

    print("=" * 70)
    print("Task4-1 数据集分析")
    print("=" * 70)
    print(f"[数据] {data_root}")

    if not data_root.exists():
        raise FileNotFoundError(f"找不到数据集目录：{data_root}\n提示：用 --data-root 指定含 data.yaml 的目录")

    report = {
        "data_root": str(data_root),
        "class_names": CLASS_NAMES,
        "splits": {},
        "warnings": [],
        "problems": [],
    }

    total_boxes = 0
    all_areas_norm: list[float] = []
    all_aspect: list[float] = []
    class_counter_all: Counter = Counter()
    size_bucket_all: Counter = Counter()
    image_sizes: Counter = Counter()
    problems: list[str] = []
    warnings: list[str] = []
    split_class_counts: dict[str, dict[str, int]] = {}
    split_max_boxes: dict[str, int] = {}

    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            problems.append(f"{split}: 缺少 images/ 或 labels/ 目录")
            continue

        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        labels = sorted(lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems
        if missing_lbl:
            problems.append(f"{split}: {len(missing_lbl)} 张图缺少标签文件")
        if orphan_lbl:
            problems.append(f"{split}: {len(orphan_lbl)} 个标签没有对应图像")

        split_info = {
            "images": len(images),
            "labels": len(labels),
            "boxes": 0,
            "background_images": 0,
            "class_counts": Counter(),
            "size_buckets": Counter(),
            "image_sizes": Counter(),
            "max_boxes_per_image": 0,
            "label_format_lines": {"detection": 0, "polygon": 0},
        }

        for img_path in images:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            split_info["image_sizes"][f"{img_w}x{img_h}"] += 1
            image_sizes[f"{img_w}x{img_h}"] += 1

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            boxes, n_detection, n_polygon = parse_label_file(lbl_path)
            split_info["label_format_lines"]["detection"] += n_detection
            split_info["label_format_lines"]["polygon"] += n_polygon
            split_info["max_boxes_per_image"] = max(split_info["max_boxes_per_image"], len(boxes))
            if not boxes:
                split_info["background_images"] += 1
                continue

            for cls, xc, yc, w, h in boxes:
                if cls not in (0, 1):
                    problems.append(f"{lbl_path.name}: 非法类别 {cls}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                    problems.append(f"{lbl_path.name}: 坐标越界 {(xc, yc, w, h)}")
                    continue
                if w <= 0 or h <= 0:
                    problems.append(f"{lbl_path.name}: 宽高非正 w={w} h={h}")
                    continue

                px_w, px_h = w * img_w, h * img_h
                area = px_w * px_h

                split_info["boxes"] += 1
                split_info["class_counts"][CLASS_NAMES[cls]] += 1
                class_counter_all[CLASS_NAMES[cls]] += 1
                total_boxes += 1

                if area < SMALL_MAX:
                    bucket = "small(<32^2)"
                elif area < MEDIUM_MAX:
                    bucket = "medium(32^2~96^2)"
                else:
                    bucket = "large(>=96^2)"
                split_info["size_buckets"][bucket] += 1
                size_bucket_all[bucket] += 1

                all_areas_norm.append(area / (img_w * img_h))
                all_aspect.append(px_w / px_h if px_h > 0 else 0)

        split_info["class_counts"] = dict(split_info["class_counts"])
        split_info["size_buckets"] = dict(split_info["size_buckets"])
        split_info["image_sizes"] = dict(split_info["image_sizes"])
        report["splits"][split] = split_info
        split_class_counts[split] = dict(split_info["class_counts"])
        split_max_boxes[split] = split_info["max_boxes_per_image"]

        print(f"\n【{split}】")
        print(f"  图像数        : {split_info['images']}")
        print(f"  标签文件数    : {split_info['labels']}")
        print(f"  目标框总数    : {split_info['boxes']}")
        print(f"  纯背景图      : {split_info['background_images']}")
        print(f"  类别分布      : {split_info['class_counts']}")
        print(f"  尺寸分布      : {split_info['size_buckets']}")
        print(f"  单图最多目标  : {split_info['max_boxes_per_image']}")
        print(f"  标签格式行数  : {split_info['label_format_lines']}")

    print("\n" + "=" * 70)
    print("【整体汇总】")
    print("=" * 70)
    print(f"  总目标框数    : {total_boxes}")
    print(f"  总体类别分布  : {dict(class_counter_all)}")

    bird_n = class_counter_all.get("bird", 0)
    drone_n = class_counter_all.get("drone", 0)
    imbalance_ratio = None
    if bird_n and drone_n:
        imbalance_ratio = max(bird_n, drone_n) / min(bird_n, drone_n)
        print(f"  类别不平衡比  : {imbalance_ratio:.2f} : 1")

    print(f"  总体尺寸分布  : {dict(size_bucket_all)}")
    small_ratio = size_bucket_all.get("small(<32^2)", 0) / total_boxes if total_boxes else 0
    print(f"  小目标占比    : {small_ratio * 100:.1f}%")

    areas = np.array(all_areas_norm)
    aspects = np.array(all_aspect)
    if len(areas):
        print(f"  归一化面积    : 中位数={np.median(areas):.5f}  均值={areas.mean():.5f}")
        print(f"  等效边长      : 中位数={np.sqrt(np.median(areas)) * 100:.2f}% 图像宽")
        print(f"  宽高比        : 中位数={np.median(aspects):.2f}  "
              f"范围=[{aspects.min():.2f}, {aspects.max():.2f}]")
    print(f"  图像尺寸种类  : {len(image_sizes)} 种，最常见 = {image_sizes.most_common(3)}")

    train_classes = split_class_counts.get("train", {})
    test_classes = split_class_counts.get("test", {})
    if len(test_classes) == 1 and len(train_classes) > 1:
        only_cls = next(iter(test_classes))
        warnings.append(
            f"test 只包含类别 {only_cls}（{test_classes[only_cls]} 个框）；"
            f"train 包含 {train_classes}。若 test 图片实际是无人机，"
            "说明标签类别索引可能导出错了；评估时可用 --test-label-map 0:1。"
        )

    for split, max_boxes in split_max_boxes.items():
        if max_boxes > 300:
            warnings.append(
                f"{split} 单图最多 {max_boxes} 个目标，Ultralytics 默认 max_det=300 会截断；"
                f"评估时请设置 max_det>={max_boxes}。"
            )

    polygon_total = sum(s["label_format_lines"]["polygon"] for s in report["splits"].values())
    if polygon_total:
        warnings.append(
            f"数据集中有 {polygon_total} 行多边形格式标签，本脚本按外接矩形统计；"
            "Ultralytics 检测训练只使用框，混用格式时多边形会被忽略。"
        )
    if problems:
        warnings.append(f"发现 {len(problems)} 个标签格式问题，详见 problems 字段。")

    report["total_boxes"] = total_boxes
    report["overall_class_counts"] = dict(class_counter_all)
    report["overall_size_buckets"] = dict(size_bucket_all)
    report["small_object_ratio"] = round(small_ratio, 4)
    report["class_imbalance_ratio"] = round(imbalance_ratio, 4) if imbalance_ratio else None
    report["image_size_variants"] = dict(image_sizes)
    report["split_class_counts"] = split_class_counts
    report["split_max_boxes"] = split_max_boxes
    report["problems"] = problems
    report["warnings"] = warnings

    save_json(args.output / "dataset_analysis.json", report)

    if warnings:
        print("\n【警告】")
        for item in warnings:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
