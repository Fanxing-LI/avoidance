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


env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/navigation.yaml')

eval_env = NavigationEnv(
    **env_config["eval_env"]
)
eval_env.reset()
img = eval_env.render()[0]
depth = eval_env.sensor_obs["depth2"][0][0]
# show img and save
depth_img = colorize_depth_numpy(depth, cmap=cv2.COLORMAP_TURBO, invert=False)

# img = colorize_depth_numpy(img, cmap=cv2.COLORMAP_TURBO)
cv2.imshow("render", img)
# cv2.imshow("depth", depth_img)
cv2.waitKey(100)
cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+"/rendered_image.png", img)
cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+"/rendered_depth.png", depth_img)
