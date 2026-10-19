# -*- coding: utf-8 -*-
"""task3_common.py —— 任务三两个脚本共用的“小工具箱”


这个文件本身【不能被直接运行】，它只提供函数给 3a / 3b 调用。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402  （放在环境变量之后是刻意的）

# 本文件所在目录 = 拆分后的脚本目录
HERE = Path(__file__).resolve().parent

#
DEFAULT_OUTPUT = HERE / "task3_results"


def set_seed(seed: int) -> None:

    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # GPU 上的随机数发生器是独立的，要单独播种
        torch.cuda.manual_seed_all(seed)


def r_squared(target: np.ndarray, prediction: np.ndarray) -> float:

    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return float(1 - residual / total) if total > 0 else float("nan")


def add_output_argument(parser: argparse.ArgumentParser) -> None:
    """给命令行解析器加上统一的 --output 参数（两个脚本共用同一套写法）。"""
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"结果输出目录，默认 {DEFAULT_OUTPUT}（不存在会自动创建）",
    )
