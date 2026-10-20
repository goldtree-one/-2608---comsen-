from __future__ import annotations

import argparse

from task4_common import (Assets, add_asset_argument, add_predict_arguments,
                          find_asset_dir, load_model, save_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4-2 官方预训练权重推理图片")
    add_asset_argument(parser)
    add_predict_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 70)
    print("Task4-2 官方预训练权重推理 3 张图片")
    print("=" * 70)

    assets = Assets(find_asset_dir(args.asset_dir))
    assets.describe()

    images = sorted(assets.example_dir.glob("*.jpg")) + sorted(assets.example_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"{assets.example_dir} 下没有找到 jpg/png 图片。\n"
            "提示：用 --asset-dir 指定含 example/ 的素材目录。"
        )
    print(f"\n[输入] 待推理图片 {len(images)} 张：" + "、".join(p.name for p in images))

    model = load_model(
        assets.coco_weights,
        "首次运行会自动下载 yolo11n.pt（约 5.6 MB），请保持联网；也可手动放到项目目录",
        auto_download=True,
    )

    save_dir = args.output / "predict_images"
    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=False,
    )

    print(f"\n[结果] 共处理 {len(results)} 张图片")
    summary = []
    for img_path, result in zip(images, results):
        boxes = result.boxes
        print(f"\n  {img_path.name}  (图像尺寸 {result.orig_shape})")
        if boxes is None or len(boxes) == 0:
            print("      未检测到目标")
            summary.append({"image": img_path.name, "detections": []})
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        dets = []
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            name = result.names[cls_id]
            print(f"      {name:<12} 置信度={conf:.3f}  "
                  f"框=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            dets.append({
                "class": name,
                "conf": round(float(conf), 4),
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            })
        summary.append({"image": img_path.name, "detections": dets})

    save_json(args.output / "predict_images_summary.json", summary)
    print(f"[保存] 标注图片 -> {save_dir}")


if __name__ == "__main__":
    main()
