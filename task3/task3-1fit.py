# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 用不弹窗的后端，只存图片文件（服务器/无桌面环境也能跑）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from task3_common import add_output_argument, r_squared, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# 真值参数（题目给定），用来造数据 + 最后对照
# 真值来源：题目 PDF 第 3~4 页题二；PDF 里公式是图片，这里是转录结果。
# ---------------------------------------------------------------------------
TRUE_PARAMETERS = [2.0, 1.5, 3.0, 0.5, -0.5, 0.8, 2026.0]  # a1 a2 a3 a4 a5 a6 b
PARAMETER_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6", "b"]

# 采样区间：x1、x2 都在 [-3, 3]（与题目示例图横轴一致）
DOMAIN_LOW, DOMAIN_HIGH = -3.0, 3.0


class FunctionModel(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        # 6 个系数 a1..a6，用 float64（双精度）保证数值稳定。
        # 这些都是 nn.Parameter，等价于“打开了 requires_grad=True”，会被自动求导。
        self.coefficients = nn.Parameter(torch.zeros(6, dtype=torch.float64))
        # 截距项。0 维张量（就是一个数），同样是要学的参数。
        self.centered_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, x):

        x1, x2 = x[:, 0], x[:, 1]          # 第 0 列是 x1，第 1 列是 x2
        a1, a2, a3, a4, a5, a6 = self.coefficients
        return (
            a1 * x1 ** 2
            + a2 * x1 * x2
            + a3 * torch.sin(x1)
            + a4 * torch.cos(2 * x2)
            + torch.exp(a5 * x1)           # 注意：a5 在指数上，属于“非线性参数”
            + a6 * x2
            + self.centered_bias
        )


def true_function(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """题目给的“不含噪声的真实函数”，用来造数据和画左图。"""
    a1, a2, a3, a4, a5, a6, b = TRUE_PARAMETERS
    return (
        a1 * x1 ** 2 + a2 * x1 * x2 + a3 * np.sin(x1)
        + a4 * np.cos(2 * x2) + np.exp(a5 * x1) + a6 * x2 + b
    )


def build_parser() -> argparse.ArgumentParser:
    """定义本脚本支持的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="task-3 题二：PyTorch 自动求导 + 梯度下降 · 最小二乘法函数拟合")
    add_output_argument(parser)  # --output
    parser.add_argument("--fit-steps", type=int, default=20000,
                        help="梯度下降迭代次数，默认 20000（对应 3.py 的 --fit-steps）")
    parser.add_argument("--n-points", type=int, default=1500,
                        help="随机采样点数，默认 1500（与 3.py 实际行为一致）")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="学习率，默认 0.02（Adam 优化器的步长）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--n-plot-points", type=int, default=50,
                        help="3D 图上画多少个采样点，默认 50（点太多会糊成一团）")
    return parser


def fit(args) -> dict:
    """造数据 → 梯度下降 → 算指标 → 画图 → 写 JSON，返回结果字典。"""
    # ------------------------------------------------------------------
    # 步骤 1：现场造数据
    # ------------------------------------------------------------------
    # 用 numpy 的 default_rng（现代随机数发生器，可复现）
    rng = np.random.default_rng(args.seed)
    points = rng.uniform(DOMAIN_LOW, DOMAIN_HIGH, size=(args.n_points, 2))
    x1, x2 = points[:, 0], points[:, 1]

    clean = true_function(x1, x2)                                  # 不含噪声的真值
    observed = clean + rng.uniform(-3, 3, size=len(clean))         # 加上 -3~3 的均匀噪声

    # 中心化：整体减去均值，让网络输出落在 0 附近（见 FunctionModel 里的说明）
    offset = float(observed.mean())
    features = torch.tensor(points, dtype=torch.float64)
    targets = torch.tensor(observed - offset, dtype=torch.float64)

    # ------------------------------------------------------------------
    # 步骤 2：梯度下降主循环
    # ------------------------------------------------------------------
    model = FunctionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()      # 均方误差 = “最小二乘”里的那个平方和

    losses: list[float] = []
    for step in range(args.fit_steps):
        optimizer.zero_grad(set_to_none=True)   # 清空上一次的梯度（不清会累加！）
        prediction = model(features)            # ① 前向传播：算预测值
        loss = criterion(prediction, targets)   # ② 算损失
        if not torch.isfinite(loss).item():
            # 数值爆炸（nan / inf）时给出可操作的错误提示，而不是静默产出垃圾结果
            raise RuntimeError("损失出现 nan/inf，训练发散；请调小 --lr 后重试。")
        loss.backward()                         # ③ 反向传播：自动求出 7 个参数的梯度
        optimizer.step()                        # ④ 按梯度更新参数
        losses.append(loss.item())

    # ------------------------------------------------------------------
    # 步骤 3：取出拟合结果 + 算三个 R²
    # ------------------------------------------------------------------
    with torch.no_grad():
        fitted = model(features).numpy() + offset          # 把 offset 加回去，还原真实尺度
        coefficients = model.coefficients.numpy().tolist()
        bias = model.centered_bias.item() + offset         # 截距也要还原

    # 为了画光滑曲面，另外铺一张规整网格（训练用的是散乱随机点，画不出曲面）
    axis = np.linspace(DOMAIN_LOW, DOMAIN_HIGH, 60)
    grid1, grid2 = np.meshgrid(axis, axis)
    grid_points = torch.tensor(np.column_stack((grid1.ravel(), grid2.ravel())),
                               dtype=torch.float64)
    true_surface = true_function(grid1, grid2)
    with torch.no_grad():
        fitted_surface = model(grid_points).numpy().reshape(grid1.shape) + offset

    # 三个 R²，分别回答三个不同的问题
    scores = {
        # 拟合函数 vs 含噪数据：题目要求输出的那个 R²，上限由噪声决定（约 0.95）
        "r2_noisy_samples": r_squared(observed, fitted),
        # 拟合函数 vs 无噪声真值：越接近 1 说明学到的规律越准
        "r2_clean_samples": r_squared(clean, fitted),
        # 拟合曲面 vs 真值曲面（在网格上算）：图形层面的整体吻合度
        "r2_clean_grid": r_squared(true_surface, fitted_surface),
    }

    result = {
        "coefficients_a1_to_a6": coefficients,
        "b": bias,
        "true_parameters": TRUE_PARAMETERS,
        "steps": args.fit_steps,
        "n_points": args.n_points,
        "learning_rate": args.lr,
        "centering_offset": offset,
        "final_mse": float(np.mean((observed - fitted) ** 2)),
        **scores,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "function_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 步骤 4：画题目要求的那张对比图（左：真值曲面；右：拟合曲面）
    # ------------------------------------------------------------------
    # 为了公平对比，两张图用同一个 z 轴范围
    z_low = min(true_surface.min(), fitted_surface.min(), observed.min(), fitted.min())
    z_high = max(true_surface.max(), fitted_surface.max(), observed.max(), fitted.max())

    # 训练用了全部 n_points 个点，但画图只随机抽 50 个，否则散点会糊成一片
    plot_rng = np.random.default_rng(args.seed)   # 单独开一个发生器，不影响训练用的随机数
    idx_plot = plot_rng.choice(len(points), size=min(args.n_plot_points, len(points)),
                              replace=False)
    points_plot, observed_plot, fitted_plot = points[idx_plot], observed[idx_plot], fitted[idx_plot]

    fig = plt.figure(figsize=(14, 6))

    # 左子图：Ground Truth（真值曲面 + 红色采样点）
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(grid1, grid2, true_surface, cmap="viridis", alpha=0.85)
    ax1.scatter(points_plot[:, 0], points_plot[:, 1], observed_plot,
                c="red", s=12, alpha=0.7, label="Sampled Points")
    ax1.set_title(f"Ground Truth Function ($R^2 = {scores['r2_clean_grid']:.4f}$)", fontsize=14)
    ax1.set_xlabel("$x_1$", fontsize=12)
    ax1.set_ylabel("$x_2$", fontsize=12)
    ax1.set_zlabel("$y$", fontsize=12)
    ax1.set_zlim(z_low, z_high)
    ax1.legend(loc="upper right")
    ax1.view_init(elev=25, azim=-60)     # 固定观察角度，两张图视角一致

    # 右子图：Fitted（拟合曲面 + 蓝色预测点）
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(grid1, grid2, fitted_surface, cmap="plasma", alpha=0.85)
    ax2.scatter(points_plot[:, 0], points_plot[:, 1], fitted_plot,
                c="blue", s=12, alpha=0.7, label="Fitted Points")
    # 可选改进：如果想让右图标题显示“拟合 vs 含噪数据”的 R²，
    # 把下面这行的 scores['r2_clean_grid'] 改成 scores['r2_noisy_samples'] 即可。
    ax2.set_title(f"Fitted Function ($R^2 = {scores['r2_clean_grid']:.4f}$)", fontsize=14)
    ax2.set_xlabel("$x_1$", fontsize=12)
    ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_zlabel("$y$", fontsize=12)
    ax2.set_zlim(z_low, z_high)
    ax2.legend(loc="upper right")
    ax2.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(args.output / "function_surfaces.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 步骤 5：画损失曲线（单独一张图，和上面的对比图分开）
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(xlabel="Optimization step", ylabel="MSE", title="Function fitting loss")
    fig.tight_layout()
    fig.savefig(args.output / "function_loss.png", dpi=160)
    plt.close(fig)

    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.fit_steps < 1 or args.n_points < 1 or args.n_plot_points < 1:
        parser.error("--fit-steps、--n-points、--n-plot-points 必须是正整数")

    print(f"[输出] 结果目录：{args.output.resolve()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}"
          + (f"（{torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))

    set_seed(args.seed)          # 固定随机种子，保证可复现
    result = fit(args)

    # 打印参数对照表，一眼看出拟合得准不准
    print("\n[参数对照]  真值  →  拟合值")
    for name, truth, got in zip(PARAMETER_NAMES, TRUE_PARAMETERS,
                                result["coefficients_a1_to_a6"] + [result["b"]]):
        print(f"  {name:<3} {truth:>12.4f}  →  {got:>14.6f}   （误差 {abs(truth - got):.6f}）")

    print(f"\n[R²] 拟合 vs 含噪数据（题目要求的 R²）: {result['r2_noisy_samples']:.4f}")
    print(f"[R²] 拟合 vs 无噪真值                : {result['r2_clean_samples']:.4f}")
    print(f"[R²] 曲面 vs 真值曲面                : {result['r2_clean_grid']:.4f}")
    print(f"\n[文件] function_metrics.json / function_surfaces.png / function_loss.png"
          f"  →  {args.output.resolve()}")


if __name__ == "__main__":
    main()
