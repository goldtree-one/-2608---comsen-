from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from task4_common import OUTPUT_DIR

HERE = Path(__file__).resolve().parent

STEPS = [
    ("task4-1-data.py", "数据集体检与统计", False, False, False),
    ("task4-2-infer.py", "官方权重推理 3 张图片", False, True, True),
    ("task4-3-video.py", "无人机视频推理与追踪", False, True, True),
    ("task4-4-train.py", "训练与 train/val/test mAP 评估", True, False, True),
    ("task4-5-export.py", "ONNX 导出与量化对比", False, False, True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4 一键顺序执行")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--batch", type=int, default=16, help="批大小，默认 16")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader 进程数，默认 0")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，默认 640")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--device", default=None, help="'0' 用 GPU，'cpu' 用 CPU，不填自动")
    parser.add_argument("--test-label-map", default="0:1",
                        help="传给训练脚本的 test 标签修正，默认 0:1；不修正传 none")
    parser.add_argument("--asset-dir", type=Path, default=None, help="题目素材目录")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"输出目录，默认 {OUTPUT_DIR}")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练步骤")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print("=" * 70)
    print("Task4 全流程执行")
    print("=" * 70)
    print(f"[模式] {'预览' if args.dry_run else '执行'}")
    print(f"[配置] epochs={args.epochs} batch={args.batch} workers={args.workers} "
          f"imgsz={args.imgsz} conf={args.conf} iou={args.iou} "
          f"device={args.device or 'auto'}")
    print(f"[测试标签] {args.test_label_map}")

    failures: list[str] = []
    for idx, (script_name, desc, is_train, takes_predict_args, takes_size_args) in \
            enumerate(STEPS, 1):
        if is_train and args.skip_train:
            print(f"\n{'=' * 70}\n[步骤 {idx}/5] {desc} -- 已跳过\n{'=' * 70}")
            continue

        script = HERE / script_name
        if not script.exists():
            print(f"\n[步骤 {idx}/5] 跳过：找不到 {script}")
            failures.append(f"{script_name}（文件缺失）")
            continue

        cmd = [sys.executable, str(script)]
        if args.output:
            cmd += ["--output", str(args.output)]
        if args.asset_dir:
            cmd += ["--asset-dir", str(args.asset_dir)]
        if takes_size_args:
            cmd += ["--imgsz", str(args.imgsz)]
            if args.device:
                cmd += ["--device", str(args.device)]
        if takes_predict_args:
            cmd += ["--conf", str(args.conf), "--iou", str(args.iou)]
        if is_train:
            cmd += ["--epochs", str(args.epochs), "--batch", str(args.batch),
                    "--workers", str(args.workers),
                    "--test-label-map", str(args.test_label_map)]

        print(f"\n{'=' * 70}")
        print(f"[步骤 {idx}/5] {desc}")
        print(f"[命令] {' '.join(cmd)}")
        print("=" * 70)
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        completed = subprocess.run(cmd)
        cost = time.perf_counter() - t0
        if completed.returncode != 0:
            print(f"\n[失败] {script_name} 返回码 {completed.returncode}，耗时 {cost:.1f}s")
            print(f"       可单独重跑：python {script_name}")
            failures.append(script_name)
            break
        print(f"[完成] {script_name} 耗时 {cost:.1f}s")

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print("预览结束，未执行任何步骤")
    elif failures:
        print(f"流程中断，失败步骤：{', '.join(failures)}")
    else:
        print(f"全部完成。输出目录：{(args.output or OUTPUT_DIR).resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
