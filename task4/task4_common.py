from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "task4_results"
LEGACY_ASSET_DIR = Path(r"E:\comsenmate\算法组招新题目 (1)\task4_yolo")


def find_asset_dir(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("TASK4_ASSET_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT,
        PROJECT_ROOT / "task4_yolo",
        Path.cwd(),
        Path.cwd() / "task4_yolo",
        PROJECT_ROOT.parent / "task4_yolo",
        PROJECT_ROOT.parent.parent / "task4_yolo",
        PROJECT_ROOT.parent / "算法组招新题目 (1)" / "task4_yolo",
        LEGACY_ASSET_DIR,
    ]
    for item in candidates:
        if (item / "drone.pt").exists():
            return item.resolve()
    return candidates[0]


class Assets:
    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = Path(asset_dir)
        self.drone_weights = self.asset_dir / "drone.pt"
        self.example_dir = self.asset_dir / "example"
        self.dataset_yaml = self.asset_dir / "dataset" / "drone_and_bird" / "data.yaml"
        self.dataset_root = self.dataset_yaml.parent
        self.coco_weights = self._find_coco_weights()

    def _find_coco_weights(self) -> Path:
        candidates = [
            PROJECT_ROOT / "yolo11n.pt",
            PROJECT_ROOT / "weights" / "yolo11n.pt",
            PROJECT_ROOT.parent / "yolo11n.pt",
            PROJECT_ROOT.parent / "weights" / "yolo11n.pt",
            self.asset_dir / "yolo11n.pt",
            self.asset_dir / "weights" / "yolo11n.pt",
            Path.cwd() / "yolo11n.pt",
        ]
        for item in candidates:
            if item.exists():
                return item
        return candidates[0]

    def describe(self) -> None:
        print(f"[路径] 素材目录   : {self.asset_dir}")
        print(f"[路径] drone.pt   : {'存在' if self.drone_weights.exists() else '缺失'}")
        print(f"[路径] example/   : {'存在' if self.example_dir.exists() else '缺失'}")
        print(f"[路径] data.yaml  : {'存在' if self.dataset_yaml.exists() else '缺失'}")


def add_asset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-dir", type=Path, default=None,
                        help="素材目录（含 drone.pt），不填则自动探测")


def add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，默认 640")
    parser.add_argument("--device", default=None, help="'0' 用 GPU，'cpu' 用 CPU，不填自动")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"输出目录，默认 {OUTPUT_DIR}")


def human_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024:.2f} MB"


def print_device_info(device_arg) -> None:
    print("\n[设备自检]")
    try:
        import torch
    except ModuleNotFoundError:
        print("  [跳过] 当前环境没有 torch")
        return
    print(f"  torch 版本        : {torch.__version__}")
    print(f"  torch CUDA 版本   : {torch.version.cuda}")
    print(f"  CUDA 是否可用     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量          : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU[{i}] 名称      : {torch.cuda.get_device_name(i)}")
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        arch_list = torch.cuda.get_arch_list()
        print(f"  GPU 计算能力      : {sm}")
        print(f"  torch 编译支持架构: {arch_list}")
        if arch_list and sm not in arch_list:
            print(f"  [警告] 当前 PyTorch 未编译 {sm} 架构，可能回退 CPU")
    else:
        print("  [警告] CUDA 不可用，将使用 CPU")
    print(f"  本次 --device     : {device_arg if device_arg else '未指定，交给 Ultralytics 自动选择'}")


def get_memory_status() -> dict:
    status: dict = {}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            status["total_phys_gb"] = st.ullTotalPhys / 1024 ** 3
            status["avail_phys_gb"] = st.ullAvailPhys / 1024 ** 3
            status["commit_limit_gb"] = st.ullTotalPageFile / 1024 ** 3
            status["commit_avail_gb"] = st.ullAvailPageFile / 1024 ** 3
            status["load_percent"] = st.dwMemoryLoad
    except Exception:
        pass
    return status


def print_memory_info(workers: int) -> None:
    st = get_memory_status()
    print("\n[内存自检]")
    if not st:
        print("  [跳过] 未能读取系统内存信息")
        return
    print(f"  物理内存      : {st['avail_phys_gb']:.2f} GB 可用 / {st['total_phys_gb']:.2f} GB 总量")
    print(f"  提交内存      : {st['commit_avail_gb']:.2f} GB 可用 / {st['commit_limit_gb']:.2f} GB 上限")
    print(f"  DataLoader    : workers={workers}")
    need = 0.4 * max(workers, 0) + 1.5
    avail = st["avail_phys_gb"]
    print(f"  预计需要      : 约 {need:.1f} GB")
    if avail >= need * 1.5:
        print("  [OK] 内存余量充足")
    elif avail < need:
        print(f"  [警告] 可用物理内存只有 {avail:.2f} GB，可能不够")
    else:
        print(f"  [注意] 内存偏紧（可用 {avail:.2f} GB）")


def check_file(path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}\n提示：{hint}")
    return path


def load_model(weights, hint: str = "请确认权重路径", auto_download: bool = False):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "\n[错误] 没有安装 ultralytics，无法推理/训练。\n"
            "       请执行：pip install -r requirements_task4.txt\n"
            f"       （原始报错：{exc}）"
        ) from None

    if auto_download and not Path(weights).exists():
        print(f"[模型] 本地没有 {Path(weights).name}，尝试自动下载 ...")
    else:
        weights = check_file(weights, hint)

    print(f"[模型] 加载 {Path(weights).name} ...")
    model = YOLO(str(weights))
    print(f"[模型] 类别数 = {len(model.names)}，类别 = {model.names}")
    actual = Path(weights)
    if actual.exists():
        print(f"[模型] 权重文件大小 = {human_size(actual.stat().st_size)}")
    else:
        actual = Path(getattr(model, "ckpt_path", "") or "")
        if actual.is_file():
            print(f"[模型] 权重实际位置 = {actual}（{human_size(actual.stat().st_size)}）")
    return model


def summarize_speed(results, label: str) -> dict:
    if not results:
        return {}
    pre = float(np.mean([r.speed["preprocess"] for r in results]))
    inf = float(np.mean([r.speed["inference"] for r in results]))
    post = float(np.mean([r.speed["postprocess"] for r in results]))
    total = pre + inf + post
    stats = {
        "label": label,
        "frames": len(results),
        "preprocess_ms": round(pre, 2),
        "inference_ms": round(inf, 2),
        "postprocess_ms": round(post, 2),
        "total_ms_per_frame": round(total, 2),
        "fps_end_to_end": round(1000 / total, 2) if total > 0 else 0.0,
        "fps_inference_only": round(1000 / inf, 2) if inf > 0 else 0.0,
    }
    print(f"\n[速度] {label}")
    print(f"  帧数        : {stats['frames']}")
    print(f"  预处理      : {stats['preprocess_ms']} ms/帧")
    print(f"  推理        : {stats['inference_ms']} ms/帧")
    print(f"  后处理      : {stats['postprocess_ms']} ms/帧")
    print(f"  端到端      : {stats['total_ms_per_frame']} ms/帧 -> {stats['fps_end_to_end']} FPS")
    print(f"  仅推理      : {stats['fps_inference_only']} FPS")
    return stats


def iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[保存] {path}")


def now() -> float:
    return time.perf_counter()
