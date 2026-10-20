from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from task4_common import (OUTPUT_DIR, Assets, add_asset_argument, check_file,
                          find_asset_dir, load_model, print_device_info,
                          save_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task4-5 ONNX 导出与量化对比")
    add_asset_argument(parser)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"输出目录，默认 {OUTPUT_DIR}")
    parser.add_argument("--weights", type=Path, default=None,
                        help="要导出的权重；默认用训练产出的 best.pt")
    parser.add_argument("--data", type=Path, default=None,
                        help="int8 量化校准用的 data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="导出/基准输入尺寸")
    parser.add_argument("--device", default=None, help="'cpu' 或 '0'，不填自动")
    parser.add_argument("--bench-device", default="cpu",
                        help="速度基准设备，默认 cpu；ONNX 分支固定用 CPU")
    parser.add_argument("--bench-repeats", type=int, default=10,
                        help="每个格式计时次数，取中位数，默认 10")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本，默认 12")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印步骤，不真正导出")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 70)
    print("Task4-5 模型导出与量化")
    print("=" * 70)

    assets = Assets(find_asset_dir(args.asset_dir))
    assets.describe()
    print_device_info(args.device)

    weights = args.weights or (args.output / "runs" / "drone_yolo11n" / "weights" / "best.pt")
    if not Path(weights).exists():
        print(f"[提示] 未找到 {weights}，改用素材目录的 drone.pt")
        weights = assets.drone_weights
    weights = check_file(weights, "请先运行 task4-4-train.py 生成 best.pt，或用 --weights 指定")

    data_yaml = check_file(args.data or assets.dataset_yaml, "请确认 data.yaml 路径")
    sample = next(iter(sorted(assets.example_dir.glob("*.jpg"))), None)
    if sample is None:
        raise FileNotFoundError(f"{assets.example_dir} 下没有 jpg 图片，无法做速度基准")

    print(f"\n[权重] 源权重     : {weights}")
    print(f"[权重] 大小       : {weights.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"[基准] 设备       : {args.bench_device}（ONNX 固定 CPU）")
    print(f"[基准] 基准图片   : {sample}")
    print(f"[校准] int8 数据  : {data_yaml}")

    if args.dry_run:
        print("\n[dry-run] 将要执行的步骤：")
        print("  1) PyTorch (.pt) 基准测速")
        print("  2) export(format='onnx') -> fp32 ONNX")
        print("  3) export(format='onnx', half=True) -> fp16 ONNX")
        print("  4) export(format='onnx', int8=True, data=data.yaml) -> int8 ONNX")
        print("  5) 汇总大小与 FPS，写 export_benchmark.json")
        return

    model = load_model(weights, "请确认权重路径")
    base_mb = weights.stat().st_size / 1024 / 1024
    results = {
        "source_weights": str(weights),
        "source_size_mb": round(base_mb, 2),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "device": args.device or "auto",
        "bench_device": args.bench_device,
        "bench_repeats": args.bench_repeats,
        "benchmark_sample": sample.name,
        "exports": {},
    }

    def keep(path, name: str) -> Path:
        dst = Path(path).with_name(name)
        shutil.copy2(path, dst)
        return dst

    def bench(tag: str, wpath, device: str | None = None) -> dict:
        bench_device = device or args.bench_device
        m = load_model(wpath, "导出产物")
        for _ in range(3):
            m.predict(str(sample), imgsz=args.imgsz, device=bench_device, verbose=False)
        times = []
        for _ in range(max(args.bench_repeats, 1)):
            t0 = time.perf_counter()
            m.predict(str(sample), imgsz=args.imgsz, device=bench_device, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        ms = float(np.median(times))
        mb = Path(wpath).stat().st_size / 1024 / 1024
        print(f"  {tag:<22} 大小={mb:>7.2f} MB   单帧={ms:>7.2f} ms   "
              f"约 {1000 / ms:>6.2f} FPS")
        return {"size_mb": round(mb, 2), "ms_per_frame": round(ms, 2),
                "fps": round(1000 / ms, 2)}

    print("\n[基准] 原始 PyTorch 权重")
    results["exports"]["pytorch_fp32"] = bench("PyTorch (.pt)", weights)

    print("\n[导出] ONNX (fp32)")
    try:
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, dynamic=False,
                                 simplify=True, opset=args.opset)
        onnx_fp32 = keep(onnx_path, "best_fp32.onnx")
        results["exports"]["onnx_fp32"] = bench("ONNX (fp32)", onnx_fp32, "cpu")
    except Exception as exc:
        print(f"  [跳过] fp32 ONNX 导出失败：{type(exc).__name__}: {exc}")
        print("        可尝试：pip install onnx onnxslim")
        results["exports"]["onnx_fp32"] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n[导出] ONNX (fp16)")
    try:
        onnx_half = model.export(format="onnx", imgsz=args.imgsz, dynamic=False,
                                 simplify=True, opset=args.opset, half=True)
        onnx_fp16 = keep(onnx_half, "best_fp16.onnx")
        results["exports"]["onnx_fp16"] = bench("ONNX (fp16)", onnx_fp16, "cpu")
    except Exception as exc:
        print(f"  [跳过] fp16 导出失败：{type(exc).__name__}: {exc}")
        results["exports"]["onnx_fp16"] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n[导出] ONNX (int8)")
    try:
        onnx_int8 = model.export(format="onnx", imgsz=args.imgsz, dynamic=False,
                                 simplify=True, opset=args.opset, int8=True,
                                 data=str(data_yaml))
        results["exports"]["onnx_int8"] = bench("ONNX (int8)", onnx_int8, "cpu")
    except Exception as exc:
        print(f"  [跳过] int8 导出失败：{type(exc).__name__}: {exc}")
        print("        可尝试：pip install onnxruntime")
        results["exports"]["onnx_int8"] = {"error": f"{type(exc).__name__}: {exc}"}

    base = results["exports"]["pytorch_fp32"]
    print("\n[对比] 相对原始 PyTorch 权重")
    print(f"  {'格式':<16}{'体积':>10}{'FPS':>10}")
    for tag, info in results["exports"].items():
        if "error" in info:
            continue
        size_pct = info["size_mb"] / base["size_mb"] * 100
        fps_pct = info["fps"] / base["fps"] * 100 if base["fps"] else 0.0
        print(f"  {tag:<16}{size_pct:>9.1f}%{fps_pct:>9.1f}%")

    save_json(args.output / "export_benchmark.json", results)
    print("\n[提醒] 若体积下降但 FPS 也下降，请先确认基准测试是否在同一设备/同一推理引擎上。")


if __name__ == "__main__":
    main()
