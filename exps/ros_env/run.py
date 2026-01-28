import os, sys

sys.path.append(os.getcwd())
# from VisFly.envs.HoverEnv import HoverEnv2 as HoverEnv
from envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.RealNavigationEnv import RealNavigationEnv
from algorithms.BPTT_series.SHAC import SHAC
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.SAC import SAC
import torch as th
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--env", "-e", type=str, default="navigation")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--scene", "-sc", type=str, default="0.08tree")
    parser.add_argument("--velocity","-v", type=float, default=6.0)
    return parser


def main(debug_env=False, scene=None, velocity=None):

    args = parse_args().parse_args()

    save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"
    load_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/../std/saved/{args.env}/"
    print(args.scene)
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.scene}.yaml')
    env_config["eval_env"]["target"][0][0] = args.velocity
    env_config["eval_env"]["max_rand_velocity"] = args.velocity
    eval_env = RealNavigationEnv(
        **env_config["eval_env"]
    )
    return eval_env

if __name__ == "__main__":
    main()