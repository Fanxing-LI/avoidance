import os, sys
import numpy  as np
sys.path.append(os.getcwd())
# from VisFly.envs.HoverEnv import HoverEnv2 as HoverEnv
from envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.NavigationEnv import NavigationEnv
from algorithms.BPTT_series.SHAC import SHAC
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.SAC import SAC
import torch as th
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config
import cv2


env_alias = {
    "hovering": HoverEnv,
    "objTracking": ObjectTrackingEnv,
    "navigation": NavigationEnv,

}


def colorize_depth_torch(depth: th.Tensor, cmap=cv2.COLORMAP_JET):
    # depth: H×W or B×H×W
    d = depth.detach().float()
    if d.ndim == 3:  # batch
        imgs = []
        for i in range(d.shape[0]):
            imgs.append(colorize_depth_torch(d[i], cmap))
        return th.stack([th.from_numpy(img) for img in imgs], dim=0)
    # normalize to [0,255]
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min + 1e-8)
    d_u8 = (d_norm * 255).clamp(0, 255).to(th.uint8).cpu().numpy()
    # apply colormap -> BGR uint8
    colored = cv2.applyColorMap(d_u8, cmap)
    return colored  # H×W×3, uint8 BGR

def colorize_depth_numpy(depth: np.ndarray, cmap=cv2.COLORMAP_JET, invert=False):
    d = depth.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min + 1e-8)
    d_u8 = (d_norm * 255).astype(np.uint8)
    if invert:
        d_u8 = 255 - d_u8  # reverse mapping
    return cv2.applyColorMap(d_u8, cmap)

# --- Custom colormap builder ---
# Build a 256-color LUT by linearly interpolating between two colors sampled from an existing cv2 colormap.
# You can choose the sampling positions (0..255) within the base map.

def build_two_color_colormap(base_cmap: int, pos1: int, pos2: int, reverse: bool = False) -> np.ndarray:
    """
    base_cmap: one of cv2.COLORMAP_* constants
    pos1, pos2: integers in [0, 255] indicating where to sample the two endpoint colors from the base colormap
    reverse: if True, reverse the gradient direction
    returns: (256, 1, 3) uint8 LUT suitable for cv2.applyColorMap
    """
    pos1 = int(np.clip(pos1, 0, 255))
    pos2 = int(np.clip(pos2, 0, 255))

    # Create a ramp image to sample the base colormap LUT
    ramp = np.arange(256, dtype=np.uint8)
    ramp_img = ramp.reshape(256, 1)
    base_lut = cv2.applyColorMap(ramp_img, base_cmap)  # (256,1,3) BGR uint8

    c1 = base_lut[pos1, 0].astype(np.float32)  # BGR
    c2 = base_lut[pos2, 0].astype(np.float32)

    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    if reverse:
        t = 1.0 - t
    # Interpolate per-channel
    lut = (c1[None, :] * (1.0 - t[:, None]) + c2[None, :] * (t[:, None]))
    lut = np.clip(lut, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return lut

# Helper to apply a custom LUT colormap to a normalized depth

def apply_custom_colormap(depth: np.ndarray, lut: np.ndarray, invert: bool = False) -> np.ndarray:
    d = depth.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min + 1e-8)
    d_u8 = (d_norm * 255).astype(np.uint8)
    if invert:
        d_u8 = 255 - d_u8
    # cv2.applyColorMap accepts either a colormap code or a custom LUT (256x1x3)
    return cv2.applyColorMap(d_u8, lut)


env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/navigation.yaml')

eval_env = NavigationEnv(
    **env_config["eval_env"]
)
eval_env.reset()
img = eval_env.render()[0]
depth = eval_env.sensor_obs["depth2"][0][0]
# show img and save
# Example: build a custom colormap by picking two colors from COLORMAP_JET at positions 30 and 220
custom_lut = build_two_color_colormap(cv2.COLORMAP_WINTER, pos1=100, pos2=250, reverse=False)
# You may also try different base maps, e.g., COLORMAP_WINTER or COLORMAP_TURBO
# custom_lut = build_two_color_colormap(cv2.COLORMAP_TURBO, pos1=20, pos2=200)

# Optional input preprocessing: log scaling for better dynamic range on depth
depth_img = apply_custom_colormap(np.log10(depth + 1e-6), custom_lut, invert=False)

# img = colorize_depth_numpy(img, cmap=cv2.COLORMAP_TURBO)
cv2.imshow("render", img)
# cv2.imshow("depth", depth_img)
cv2.waitKey(100)
cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+"/rendered_image.png", img)
cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+"/rendered_depth.png", depth_img)
