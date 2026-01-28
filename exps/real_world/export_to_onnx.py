import os, sys

sys.path.append(os.getcwd())
from VisFly.envs.HoverEnv import HoverEnv
from envs.RealNavigationEnv import RealNavigationEnv

from algorithms.BPTT_series.BPTT import BPTT
from algorithms.BPTT_series.SHAC import SHAC
from VisFly.utils.algorithms.PPO import PPO
import torch as th
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config
import json


# th.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    return parser

alg_alias = {
    "BPTT": BPTT,
    "SHAC": SHAC,
}

args = parse_args().parse_args()

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
save_folder = script_dir + f"/saved/navigation/"

# config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/navigation/{args.algorithm}.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/navigation.yaml')

env_config["env"]["random_kwargs"]["state_generator"]["kwargs"][0]["position"]["half"] = [1.0, 1.0, 0.1]

eval_env = RealNavigationEnv(
    **env_config["eval_env"]
)
model = alg_alias[args.algorithm].load((save_folder + args.weight), env=eval_env)

# export model.policy to onnx
def export_policy_to_onnx(model, save_path):
    # Create a wrapper that handles the dictionary input conversion
    class PolicyWrapper(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, state, depth):
            # Convert tensor input back to dictionary format expected by policy
            obs_dict = {"state": state, "depth": depth}
            return self.policy.predict(obs_dict, deterministic=True, sample=True)

    # Wrap the policy
    wrapped_policy = PolicyWrapper(model.policy)

    # Create dummy input as tensor
    dummy_state = th.randn(eval_env.num_envs, eval_env.observation_space["state"].shape[0])
    depth_shape = eval_env._observations["depth"].shape[1:]
    dummy_depth = th.randn(eval_env.num_envs, *depth_shape)

    th.onnx.export(
        wrapped_policy,
        (dummy_state, dummy_depth),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state', "depth"],
        output_names=['output'],
    )
    print(f"Model exported to {save_path}")


name = args.weight.replace(".zip","")
export_policy_to_onnx(model, save_folder + f"{name}_policy.onnx")
