import numpy as np
import matplotlib.pyplot as plt

# =========================
# 采样方案 A：网格节点式（最像“网格点”，天然不挨着）
# =========================
def grid_like_points(bounds: np.ndarray,
                     num_points: int,
                     jitter: float = 0.0,
                     seed: int = None) -> np.ndarray:
    """
    网格节点式采样：点基本像网格中心，彼此间距规整，不容易挨着。
    bounds: shape (2,3), [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    jitter: 0~0.49，占cell尺寸比例的轻微抖动。想“像网格节点”，建议0或0.05。
    """
    assert bounds.shape == (2, 3)
    b0 = bounds[0].astype(float)
    b1 = bounds[1].astype(float)
    assert np.all(b1 >= b0), "bounds max must be >= min"

    rng = np.random.default_rng(seed)

    lengths = b1 - b0
    if np.all(lengths == 0):
        return np.repeat(b0[None, :], num_points, axis=0)

    nonzero = lengths > 0
    d = int(nonzero.sum())
    vol = float(np.prod(lengths[nonzero])) if d > 0 else 0.0
    s = (vol / max(num_points, 1)) ** (1.0 / max(d, 1)) if vol > 0 else 1.0

    counts = np.ones(3, dtype=int)
    for i in range(3):
        if lengths[i] > 0:
            counts[i] = max(1, int(np.round(lengths[i] / max(s, 1e-12))))
        else:
            counts[i] = 1

    def total(c): return int(c[0] * c[1] * c[2])

    while total(counts) < num_points:
        cell_sizes = np.where(lengths > 0, lengths / counts, -np.inf)
        k = int(np.argmax(cell_sizes))
        counts[k] += 1

    tot = total(counts)
    choose = rng.choice(tot, size=num_points, replace=False)

    ix, iy, iz = np.unravel_index(choose, counts)
    idxs = np.stack([ix, iy, iz], axis=1).astype(float)

    cell = np.where(counts > 0, lengths / counts, 0.0)
    pts = b0 + (idxs + 0.5) * cell

    if jitter > 0:
        j = float(np.clip(jitter, 0.0, 0.49))
        amp = j * cell
        noise = rng.uniform(-amp, amp, size=pts.shape)
        noise[:, ~nonzero] = 0.0
        pts = pts + noise
        pts = np.minimum(np.maximum(pts, b0), b1)

    return pts


# =========================
# 采样方案 B：best-candidate（更随机但会“铺开”）
# =========================
def best_candidate_points(bounds: np.ndarray,
                          num_points: int,
                          k: int = 80,
                          seed: int  = None) -> np.ndarray:
    """
    Mitchell best-candidate：每次从k个随机候选里选“离已有点最远”的那个，减少近邻挤堆。
    k: 候选数量，越大越分散但越慢（常用30~200）。
    """
    assert bounds.shape == (2, 3)
    b0 = bounds[0].astype(float)
    b1 = bounds[1].astype(float)
    assert np.all(b1 >= b0), "bounds max must be >= min"

    rng = np.random.default_rng(seed)
    span = b1 - b0

    pts = np.empty((num_points, 3), dtype=float)
    pts[0] = b0 + rng.random(3) * span

    for i in range(1, num_points):
        cand = b0 + rng.random((k, 3)) * span
        diff = cand[:, None, :] - pts[None, :i, :]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist2 = dist2.min(axis=1)
        pts[i] = cand[int(np.argmax(min_dist2))]

    return pts


# =========================
# Debug：最近邻距离统计（KDTree优先，没scipy就退化O(N^2)）
# =========================
def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """
    返回每个点到其最近邻的距离，shape (N,)
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    # 优先用 scipy.spatial.cKDTree（快）
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)  # 第0个是自己，取第1个
        return dists[:, 1]
    except Exception:
        # 退化：O(N^2)（N几百/一两千还能接受）
        nn = np.empty(n, dtype=float)
        for i in range(n):
            diff = points - points[i]
            dist2 = np.sum(diff * diff, axis=1)
            dist2[i] = np.inf
            nn[i] = np.sqrt(np.min(dist2))
        return nn


def summarize_nn(points: np.ndarray) -> dict:
    nn = nearest_neighbor_distances(points)
    if nn.size == 0:
        return {"N": len(points)}
    return {
        "N": len(points),
        "nn_min": float(np.min(nn)),
        "nn_mean": float(np.mean(nn)),
        "nn_median": float(np.median(nn)),
        "nn_p10": float(np.percentile(nn, 10)),
        "nn_p90": float(np.percentile(nn, 90)),
    }


# =========================
# Debug：绘制（3D散点 + 三视图投影 + 最近邻距离直方图）
# =========================
def _set_axes_equal_3d(ax):
    # 让3D坐标轴等比例，避免视觉误判
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_points_debug(points: np.ndarray,
                      bounds: np.ndarray = None,
                      title: str = "",
                      bins_hist: int = 30,
                      show: bool = True,
                      save_path: str = None):
    """
    points: (N,3)
    bounds: (2,3) 可选，用于画边界框/限制坐标轴
    """
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2 and points.shape[1] == 3

    stats = summarize_nn(points)
    nn = nearest_neighbor_distances(points)

    fig = plt.figure(figsize=(12, 9))

    # 3D scatter
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=12)
    ax3d.set_title("3D scatter")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")

    # bounds 限制
    if bounds is not None:
        b0 = bounds[0].astype(float)
        b1 = bounds[1].astype(float)
        ax3d.set_xlim(b0[0], b1[0])
        ax3d.set_ylim(b0[1], b1[1])
        ax3d.set_zlim(b0[2], b1[2])

    _set_axes_equal_3d(ax3d)

    # XY / XZ / YZ projections
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(points[:, 0], points[:, 1], s=10)
    ax_xy.set_title("XY projection")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", adjustable="box")
    if bounds is not None:
        ax_xy.set_xlim(bounds[0, 0], bounds[1, 0])
        ax_xy.set_ylim(bounds[0, 1], bounds[1, 1])

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter(points[:, 0], points[:, 2], s=10)
    ax_xz.set_title("XZ projection")
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_aspect("equal", adjustable="box")
    if bounds is not None:
        ax_xz.set_xlim(bounds[0, 0], bounds[1, 0])
        ax_xz.set_ylim(bounds[0, 2], bounds[1, 2])

    # NN distance histogram
    ax_hist = fig.add_subplot(2, 2, 4)
    if nn.size > 0:
        ax_hist.hist(nn, bins=bins_hist)
        ax_hist.set_title("Nearest-neighbor distance histogram")
        ax_hist.set_xlabel("distance"); ax_hist.set_ylabel("count")
        txt = (f"N={stats['N']}\n"
               f"nn_min={stats['nn_min']:.4g}\n"
               f"nn_median={stats['nn_median']:.4g}\n"
               f"nn_mean={stats['nn_mean']:.4g}\n"
               f"p10={stats['nn_p10']:.4g}, p90={stats['nn_p90']:.4g}")
        ax_hist.text(0.98, 0.98, txt, ha="right", va="top", transform=ax_hist.transAxes)
    else:
        ax_hist.set_title("Nearest-neighbor distance histogram (N<2)")

    fig.suptitle(title if title else "Point distribution debug", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return stats


# =========================
# 示例：跑一下
# =========================
if __name__ == "__main__":
    bounds = np.array([[0, 0, 3],
                       [10, 6, 3]], dtype=float)

    N = 300

    ptsA = grid_like_points(bounds, N, jitter=0.05, seed=0)
    print("Grid-like stats:", plot_points_debug(ptsA, bounds, title="Grid-like (jitter=0.05)", show=True))

    ptsB = best_candidate_points(bounds, N, k=80, seed=0)
    print("Best-candidate stats:", plot_points_debug(ptsB, bounds, title="Best-candidate (k=80)", show=True))
