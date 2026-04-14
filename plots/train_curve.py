from matplotlib import pyplot as plt
import numpy as np
from VisFly.utils.FigFashion.FigFashion import FigFon
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter1d
import os
import glob

FigFon.set_fashion("IEEE")


def load_tensorboard_file(folder_path, tag="rollout/ep_rew_mean", smooth_window=10):
    event_files = glob.glob(os.path.join(folder_path, "events.out.tfevents.*"))
    if not event_files:
        return np.array([]), np.array([]), np.array([])

    if len(event_files) == 1:
        file_path = event_files[0]
    else:
        file_path = max(event_files, key=os.path.getctime)

    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()

    scalar_events = ea.Scalars(tag)
    timesteps = np.array([e.step for e in scalar_events], dtype=np.float64)
    raw_values = np.array([e.value for e in scalar_events], dtype=np.float64)

    values = raw_values.copy()
    if smooth_window and values.size > smooth_window:
        w = int(smooth_window)
        padded = np.pad(values, (w // 2, w // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(w) / w, mode="valid")
        values = smoothed[: values.size]

    return values, timesteps, raw_values


def normalize_with_base(values, base_min, base_max, clip=False):
    if values.size == 0:
        return values
    if base_max <= base_min:
        out = np.zeros_like(values, dtype=np.float64)
    else:
        out = (values - base_min) / (base_max - base_min)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def compute_global_norm_base(run_dirs, tag="rollout/ep_rew_mean", smooth_window=10):
    gmin, gmax = np.inf, -np.inf
    for run_dir in run_dirs:
        smoothed, _, _ = load_tensorboard_file(run_dir, tag=tag, smooth_window=smooth_window)
        if smoothed.size == 0:
            continue
        gmin = min(gmin, float(np.min(smoothed)))
        gmax = max(gmax, float(np.max(smoothed)))

    if not np.isfinite(gmin) or not np.isfinite(gmax):
        return 0.0, 1.0
    if gmax <= gmin:
        return gmin, gmin + 1.0
    return gmin, gmax


def rolling_bounds(values, window_size):
    if window_size <= 1 or values.size == 0:
        return values.copy(), values.copy()

    w = int(window_size)
    half = w // 2
    padded = np.pad(values, (half, half), mode="edge")
    lower = np.empty_like(values, dtype=np.float64)
    upper = np.empty_like(values, dtype=np.float64)

    for i in range(values.size):
        seg = padded[i : i + w]
        lower[i] = np.min(seg)
        upper[i] = np.max(seg)

    sigma = max(1.0, w / 6.0)
    lower = gaussian_filter1d(lower, sigma=sigma, mode="nearest")
    upper = gaussian_filter1d(upper, sigma=sigma, mode="nearest")
    return lower, upper


def plot_train_curves(
    ax,
    run_dirs,
    labels,
    tag="rollout/ep_rew_mean",
    smooth_window=10,
    show_presmooth_shadow=True,
    shadow_alpha=0.14,
    norm_base=None,  # (global_min, global_max)
):
    pair_count = min(len(run_dirs), len(labels))

    for i in range(pair_count):
        smoothed, timesteps, raw_values = load_tensorboard_file(
            run_dirs[i], tag=tag, smooth_window=smooth_window
        )
        if smoothed.size == 0 or timesteps.size == 0:
            continue

        if norm_base is not None:
            bmin, bmax = norm_base
            smoothed = normalize_with_base(smoothed, bmin, bmax, clip=True)
            raw_values = normalize_with_base(raw_values, bmin, bmax, clip=True)

        line, = ax.plot(timesteps, smoothed, label=labels[i])

        if show_presmooth_shadow and raw_values.size > 0:
            lower, upper = rolling_bounds(raw_values, max(1, int(smooth_window)))
            ax.fill_between(
                timesteps, lower, upper,
                color=line.get_color(), alpha=shadow_alpha, linewidth=0
            )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Normalized Episode Reward")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(loc="lower right")


def main(base_dir, run_names, legends, save_path=None):
    fig, ax = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, HeightScale=0.8)
    run_dirs = [os.path.join(base_dir, n) for n in run_names]

    # 仅这一批 run_dirs 一起归一化
    global_base = compute_global_norm_base(
        run_dirs=run_dirs,
        tag="rollout/ep_rew_mean",
        smooth_window=10,
    )

    plot_train_curves(
        ax=ax,
        run_dirs=run_dirs,
        labels=legends,
        smooth_window=10,
        show_presmooth_shadow=True,
        shadow_alpha=0.14,
        norm_base=global_base,
    )

    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "train_curve.png")

    fig.savefig(save_path, dpi=300)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 一批实验（只这一批一起归一化）
    # path 定义（只保留一组）
    base_dir = os.path.join(project_root, "exps", "real_world", "saved", "navigation")

    run_names = [
        "PPO_final_2",
        # "SHAC_std_noRemap_deReso_3",
        "SHAC_std_Remap_thre1_1",
    ]
    legends = ["PPO", "SHAC", "SHAC_Reshaping"]
    legends = ["PPO", "SHAC"]

    main(base_dir=base_dir, run_names=run_names, legends=legends)





