import numpy as np
import matplotlib.pyplot as plt

from VisFly.utils.FigFashion.FigFashion import FigFon
FigFon.set_fashion("IEEE")
plt.rcParams["hatch.color"] = "black"
plt.rcParams["hatch.linewidth"] = 1.0

def compute_esdf_circles(grid_x, grid_y, circles):
    """
    计算多个圆的有符号距离场。
    circles: list of (center, radius), center=(cx, cy)
    """
    # 初始化esdf为一个很大的正数（表示远离所有圆）
    esdf = np.full_like(grid_x, np.inf)
    for (cx, cy), r in circles:
        dx = grid_x - cx
        dy = grid_y - cy
        dist = np.sqrt(dx*dx + dy*dy) - r
        # 取所有圆中的最小有符号距离
        esdf = np.minimum(esdf, dist)
    return esdf


def main():
    # 网格设置
    x_min, x_max, y_min, y_max = -5.0, 5.0, -1.5, 1.5
    resolution = 0.01
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # ----- 定义多个圆 -----
    circles = [
        ((4.7, 0.6), 0.2),   # 第一个圆
        ((2.8, -0.55), 0.2),   # 第二个圆
        ((1.7, 0.7), 0.2), # 第三个圆
        ((-0.2, -1.3), 0.2),  # 第二个圆
        ((-2.1, -0.1), 0.2),  # 第三个圆
        ((-3.7, -1.4), 0.2),  # 第三个圆
        ((-4.2, 1.3), 0.2),  # 第三个圆

        # 可以继续添加更多圆
    ]
    # ---------------------

    # 计算ESDF
    esdf = compute_esdf_circles(grid_x, grid_y, circles)
    # 只显示外部区域（距离>0）
    esdf_outside = np.ma.array(esdf, mask=esdf <= 0.0)

    # 计算梯度（基于整个ESDF场）
    grad_y, grad_x = np.gradient(esdf, resolution, resolution)
    grad_x = np.ma.array(grad_x, mask=esdf <= 0.0)
    grad_y = np.ma.array(grad_y, mask=esdf <= 0.0)

    # 绘图
    fig, ax = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, share_legend=False, HeightScale=0.8)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 填充外部区域
    im = ax.contourf(
        grid_x, grid_y, esdf_outside,
        levels=60, cmap="viridis", alpha=0.8,
        antialiased=False, linewidths=3
    )

    # 绘制梯度向量（步长可根据网格大小调整）
    step = 18
    ax.quiver(
        grid_x[::step, ::step], grid_y[::step, ::step],
        grad_x[::step, ::step], grad_y[::step, ::step],
        color="black", angles="xy", scale_units="xy",
        scale=15.0, width=0.002, alpha=0.9,
    )

    # 绘制零水平面（即所有圆的边界）
    ax.contour(grid_x, grid_y, esdf, levels=[0.0], colors="black", linewidths=0.7)

    # ----- 为每个圆添加边界和填充图案 -----
    for (cx, cy), r in circles:
        # 圆边界
        circle = plt.Circle((cx, cy), r, fill=False, color="black", linewidth=0.7)
        ax.add_patch(circle)
        # 填充图案
        hatch_circle = plt.Circle(
            (cx, cy), r, fill=False,
            edgecolor="black", linewidth=0.5, hatch="//////////"
        )
        ax.add_patch(hatch_circle)
    # ---------------------------------

    ax.set_aspect("equal")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim([-1.5, 1.5])

    # 保存图像（路径请根据实际情况修改）
    fig.savefig(
        "/home/lfx-desktop/files/avoidance/debug/esdf_circles.png",
        dpi=600, bbox_inches="tight", pad_inches=0, facecolor="white",
    )

if __name__ == "__main__":
    main()