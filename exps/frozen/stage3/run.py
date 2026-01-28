import os, sys

sys.path.append(os.getcwd())

from envs.NavigationEnv import NavigationEnv
from algorithms.BPTT_series.SHAC import SHAC
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.SAC import SAC
import torch as th
import numpy as np
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config
from policy import StagePolicy
from gymnasium import spaces
# th.autograd.set_detect_anomaly(True)
from regulate import regulate

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="navigation")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    parser.add_argument("--chi_policy", "-cp", type=str, default="SHAC", )
    parser.add_argument("--chi_policy_w", "-cpw", type=str, default=None, )

    return parser


alg_alias = {
    "PPO": PPO,
    "SHAC": SHAC,
    "BPTT": BPTT,
    "SAC": SAC,
}


def main(debug_env=False):
    args = parse_args().parse_args()
    current_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_folder = current_folder+ f"/saved/{args.env}/"
    chi_load_folder = current_folder + f"/../stage1/saved/{args.env}/"

    config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/{args.algorithm}.yaml')
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/{args.env}.yaml')

    if not args.train:
        env_config["eval_env"]["visual"] = True

    config["algorithm"]["policy"] = StagePolicy
    config["algorithm"]["policy_kwargs"]["inter_action_space"] = \
        spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    # config["algorithm"]["policy_kwargs"]["children_policy_cfg"] = chi_load_folder+args.chi_policy_w

    # if train mode, train the model
    if args.train:
        env = NavigationEnv(
            **env_config["env"]
        )
        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            **config["algorithm"]
        )

        if args.weight is not None:
            # model.load_parameters(save_folder + args.weight)
            try:
                model = alg_alias[args.algorithm].load(save_folder + args.weight, env=env)
            except FileNotFoundError:
                model = alg_alias[args.algorithm].load(
                    save_folder.replace("stage3", "stage2")+args.weight, env=env
                )
        model.create_save_path(save_folder, args.comment)

        model.learn(**config["learn"])
        model.save()

    else:
        eval_env = NavigationEnv(
            **env_config["eval_env"]
        )

        if debug_env:
            return eval_env

        model = alg_alias[args.algorithm].load(save_folder + args.weight, env=eval_env)
        from exps.test import Test
        test_handle = Test(
            model=model,
            save_path=save_folder + "/test",
            name=args.weight
        )
        test_handle.test(**config["test"])


if __name__ == "__main__":
    main()
