import os, sys

sys.path.append(os.getcwd())
# from VisFly.envs.HoverEnv import HoverEnv2 as HoverEnv
from envs.HoverEnv import HoverEnv
from envs.ObjectTrackingEnv import ObjectTrackingEnv
from envs.NavigationEnv import NavigationEnv
from envs.RealNavigationEnv import RealNavigationEnv
from algorithms.BPTT_series.SHAC import SHAC
from algorithms.BPTT_series.SHACSample import SHACSample
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.SAC import SAC
import torch as th
import sys
import os
import argparse
from VisFly.utils.common import load_yaml_config


# th.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments', add_help=False)
    parser.add_argument('--comment', '-c', type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="SHAC")
    parser.add_argument("--env", "-e", type=str, default="navigation")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, )
    return parser


env_alias = {
    "hovering": HoverEnv,
    "objTracking": ObjectTrackingEnv,
    "navigation": NavigationEnv,

}

alg_alias = {
    "PPO": PPO,
    "SHAC": SHAC,
    "BPTT": BPTT,
    "SAC": SAC,
    "SHACSample": SHACSample,
}

def main(debug_env=False):
    args = parse_args().parse_args()

    save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"
    load_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/../std/saved/{args.env}/"
    config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/{args.env}/{args.algorithm}.yaml')
    env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/{args.env}.yaml')

    if not args.train:
        env_config["eval_env"]["visual"] = True

    # if train mode, train the model
    if args.train:
        env = env_alias[args.env](
            **env_config["env"]
        )
        if "Sample" in args.algorithm:
            train_env = env_alias[args.env](
                **env_config["env"]
            )
            config["algorithm"]["train_env"] = train_env

        model = alg_alias[args.algorithm](
            env=env,
            seed=args.seed,
            comment=args.comment,
            save_path=save_folder,
            **config["algorithm"]
        )

        if args.weight is not None:
            # model = model.load(path=save_folder + args.weight, env=env)
            try:
                model.load_parameters(load_folder + args.weight)
            except FileNotFoundError:
                model.load_parameters(save_folder + args.weight)
            model.create_save_path(args.comment)

        model.learn(**config["learn"])
        model.save()

    else:
        eval_env = RealNavigationEnv(
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