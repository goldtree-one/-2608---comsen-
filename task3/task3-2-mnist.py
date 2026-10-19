# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from task3_common import add_output_argument, set_seed  # noqa: E402

# 原 3.py 里写死的数据路径，这里留作“最后的兜底候选”
LEGACY_DATA_PATH = Path(r"E:\comsenmate\算法组招新题目 (1)\task3\train-00000-of-00001.parquet")


def find_data_file() -> Path:
    """自动寻找题目给的 MNIST parquet 文件，找不到就返回兜底路径（后面会报错提示）。"""
    here = Path(__file__).resolve().parent
    candidates = [
        # 1) 数据文件就和脚本放在一起
        here / "train-00000-of-00001.parquet",
        # 2) 拆分包里的 data/ 子目录
        here / "data" / "train-00000-of-00001.parquet",
        # 3) 命令行当前目录
        Path.cwd() / "train-00000-of-00001.parquet",
        # 4) 题目素材目录（本机真实位置）
        LEGACY_DATA_PATH,
    ]
    for c in candidates:
        if c.is_file():
            return c
    # 5) 再宽松一点：在候选目录里搜任何 train-*.parquet
    for folder in (here, here / "data", Path.cwd(), LEGACY_DATA_PATH.parent):
        if folder.is_dir():
            hits = sorted(folder.glob("train-*.parquet"))
            if hits:
                return hits[0]
    return LEGACY_DATA_PATH


def load_mnist(path: Path):
    """读取 parquet 里的图片与标签，返回张量 x（图像）和 y（标签）。

    对应 3.py 第 127~149 行。

    parquet 每行的 image 列是一个 dict，里面 key='bytes' 存着图片的原始字节；
    所以要用 BytesIO 把它包成“文件对象”，再交给 PIL 解码。
    这里逐行做了三项校验，防止“数据不对却一路跑到底、最后得出莫名其妙的准确率”。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            "提示：用 --data 指定 train-00000-of-00001.parquet 的完整路径。")

    frame = pd.read_parquet(path, engine="pyarrow")   # pandas + pyarrow 读 parquet
    if not {"image", "label"}.issubset(frame.columns):
        raise ValueError(f"期望有 image 和 label 两列，实际是：{list(frame.columns)}")

    images = []
    for row, cell in enumerate(frame["image"]):
        if not isinstance(cell, dict) or cell.get("bytes") is None:
            raise ValueError(f"第 {row} 行没有内嵌图片字节（image.bytes 为空）。")
        with Image.open(BytesIO(cell["bytes"])) as picture:   # 图像库：PIL
            gray = picture.convert("L")                        # 转成单通道灰度图
            if gray.size != (28, 28):
                raise ValueError(f"第 {row} 行尺寸是 {gray.size}，期望 (28, 28)。")
            images.append(np.array(gray, dtype=np.uint8))

    pixels = np.stack(images)                                  # (N, 28, 28)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if len(labels) < 30 or not np.isin(labels, np.arange(10)).all():
        raise ValueError("至少需要 30 张图，且标签必须是 0~9 的整数。")

    # 归一化到 [0,1]，并补上通道维：(N,28,28) → (N,1,28,28)
    x = torch.from_numpy(pixels).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(labels.copy()).long()
    print(f"[数据] 已读取 {len(y)} 张图片；x={tuple(x.shape)}，y={tuple(y.shape)}")
    return x, y


class MLP(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),              # (N,1,28,28) → (N,784)，把图拉成一维向量
            nn.Linear(784, 256),       # 第一层：784 个输入 → 256 个隐藏单元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),       # 第二层：256 → 128
            nn.ReLU(),
            nn.Linear(128, 10),        # 输出层：128 → 10（对应 0~9 十个数字）
        )

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()   # 评估时不需要梯度，关掉可省显存、跑得更快
def evaluate(model, loader, device) -> float:
    """在给定数据集上算准确率（预测对的比例）。

    对应 3.py 第 167~176 行。
    argmax(dim=1) 的意思是：10 个输出里挑分数最大的那个下标，就是预测的数字。
    """
    model.eval()               # 切到评估模式：关闭 Dropout
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predicted = model(x).argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return correct / total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="task-3 题三：多层感知机(MLP) 识别 MNIST 手写数字")
    add_output_argument(parser)  # --output
    parser.add_argument("--data", type=Path, default=None,
                        help="MNIST parquet 文件路径；不填则自动探测")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数，默认 10")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="批大小，默认 128")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="只用前 N 张图做快速冒烟测试；0 表示用全部数据")
    return parser


def train(args) -> dict:
    """完整训练流程：读数据 → 划分 → 训练 → 评估 → 画 4 张图 → 写 JSON。"""
    data_path = args.data or find_data_file()
    x, y = load_mnist(data_path)

    # 如果只做冒烟测试，就截取前 N 张
    if args.max_samples and args.max_samples < len(y):
        x, y = x[:args.max_samples], y[:args.max_samples]
        print(f"[提示] --max-samples 生效，仅使用前 {len(y)} 张图")

    # ------------------------------------------------------------------
    # 步骤 1：划分训练集 / 验证集 / 测试集 = 8 : 1 : 1
    # ------------------------------------------------------------------
    # 先打乱全部下标，再按 80% / 10% / 10% 切开。
    # 为什么要三个集合？
    #   训练集：用来更新权重
    #   验证集：用来“挑选”最好的那一轮模型（不参与更新权重）
    #   测试集：最后只测一次，代表模型对没见过的数据的真实水平
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(y), generator=generator)
    train_end, val_end = int(0.8 * len(y)), int(0.9 * len(y))
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]

    loaders = [
        DataLoader(TensorDataset(x[ids], y[ids]), batch_size=args.batch_size,
                   shuffle=shuffle, num_workers=0)
        for ids, shuffle in [(train_ids, True), (val_ids, False), (test_ids, False)]
    ]
    train_loader, val_loader, test_loader = loaders
    # 说明：只有训练集需要 shuffle=True（每轮打乱顺序，避免模型记住样本顺序）；
    # 验证/测试集打乱没有意义。num_workers=0 表示在主进程里读数据，
    # 在 Windows 上最稳（多进程 spawn 容易吃内存，见 4.py 里的详细解释）。

    # ------------------------------------------------------------------
    # 步骤 2：建模型、定损失、定优化器
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失：多分类任务的标准损失，内部含 softmax
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_val = -1.0
    best_path = args.output / "mnist_best.pt"
    print(f"[设备] {device}；训练/验证/测试 = "
          f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # ------------------------------------------------------------------
    # 步骤 3：训练循环（每个 epoch 走一遍训练集）
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()                     # 切到训练模式：开启 Dropout
        loss_sum, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)   # 清梯度，否则会累加
            logits = model(batch_x)                 # ① 前向传播
            loss = criterion(logits, batch_y)       # ② 算损失
            loss.backward()                         # ③ 反向传播（自动求导）
            optimizer.step()                        # ④ 更新参数

            count = batch_y.size(0)
            # 按样本数加权累加，这样最后一个不满的 batch 也不会把均值算歪
            loss_sum += loss.item() * count
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += count

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc_online": correct / total,      # 训练过程中的在线准确率
            "val_acc": val_acc,
            "test_acc_heldout": test_acc,             # 顺带观察，但不用于挑选模型
        }
        history.append(record)

        # 只按“验证集准确率”保存最优模型（这是防止过拟合的标准做法）
        if val_acc > best_val:
            best_val = val_acc
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        print(f"  epoch {epoch:>3}/{args.epochs}  "
              f"loss={record['train_loss']:.4f}  "
              f"train_acc={record['train_acc_online']:.4f}  "
              f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ------------------------------------------------------------------
    # 步骤 4：载入最优权重，得到最终测试准确率
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_test_acc = evaluate(model, test_loader, device)
    table = pd.DataFrame(history)

    # ------------------------------------------------------------------
    # 步骤 5：画题目要求的三张曲线图（分开画，便于单独放进文档）
    # ------------------------------------------------------------------
    # 图1：损失函数 Loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["epoch"], table["train_loss"], marker="o", c="#1f77b4")
    ax.set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "mnist_loss.png", dpi=160)
    plt.close(fig)

    # 图2：训练准确度 Train Acc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["epoch"], table["train_acc_online"] * 100, marker="o", c="#2ca02c")
    ax.set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "mnist_train_acc.png", dpi=160)
    plt.close(fig)

    # 图3：测试准确度 Test Acc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["epoch"], table["test_acc_heldout"] * 100, marker="o",
            c="#ff7f0e", label="Test Acc")
    ax.set(title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "mnist_test_acc.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 步骤 6：预测结果可视化（3×3 九宫格，标题写 真实值 | 预测值）
    # ------------------------------------------------------------------
    sample_x, sample_y = x[test_ids[:9]], y[test_ids[:9]]
    model.eval()
    with torch.no_grad():
        sample_prediction = model(sample_x.to(device)).argmax(dim=1).cpu()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, image, truth, guess in zip(axes.flat, sample_x, sample_y, sample_prediction):
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        # 预测正确用黑字，预测错误用红字，一眼就能看出错在哪
        ax.set_title(f"True: {truth.item()} | Pred: {guess.item()}",
                     color="black" if truth.item() == guess.item() else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "mnist_predictions.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 步骤 7：写实验摘要 JSON
    # ------------------------------------------------------------------
    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "data_file": str(data_path),
        "max_samples": args.max_samples or None,
        "split_counts": [len(train_ids), len(val_ids), len(test_ids)],
        "test_source": "题目提供的 parquet 里随机切出的 10%（不是 MNIST 官方测试集）",
        "selected_by": "只按验证集准确率挑选最优权重",
        "best_val_acc": best_val,
        "selected_model_test_acc": final_test_acc,
        "device": str(device),
    }
    (args.output / "mnist_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs 和 --batch-size 必须是正整数")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 结果目录：{args.output.resolve()}")
    set_seed(args.seed)
    summary = train(args)

    print("\n[结果] 最佳验证准确率 : "
          f"{summary['best_val_acc']:.4f}")
    print(f"[结果] 最优模型测试准确率 : "
          f"{summary['selected_model_test_acc']:.4f}")
    print("\n[文件] mnist_metrics.json / mnist_best.pt / mnist_loss.png / "
          "mnist_train_acc.png / mnist_test_acc.png / mnist_predictions.png")
    print(f"  →  {args.output.resolve()}")


if __name__ == "__main__":
    main()
