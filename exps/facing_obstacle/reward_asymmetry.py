import matplotlib.pyplot as plt
import torch as th
from VisFly.utils.common import load_yaml_config
import os, sys
from CenterObstacleEnv import CenterObstacleEnv as NavigationEnv
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")

path ="BPTT_deterministic_share0.6_max12_1.zip"

load_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/../std/saved/navigation/"
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/"

config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/navigation/BPTT.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/navigation.yaml')

eval_env = NavigationEnv(
    **env_config["eval_env"]
)

model = BPTT.load(load_folder + path, env=eval_env)
from exps.test import Test

test_handle = Test(
    model=model,
    save_path=save_folder + "/test",
    name="critic"
)
test_handle.test(**config["test"])
state = th.cat(test_handle.state_all, dim=0)
start_i = 30
v = state[:, 7:10].norm(dim=1)[start_i:]
min_v, index = v.min(0)
depth = test_handle.obs_all[start_i + index.item()]["depth"]
state_input = test_handle.obs_all[start_i + index.item()]["state"]

fig, axeses = FigFon.get_figure_axes((1,4), HeightScale=1, Column=1)

loop = th.arange(-0.4,0.4,0.01)
action = th.zeros((len(loop), 4))
state_input = th.tile(state_input, (len(loop),1))
depth_input = th.tile(depth, (len(loop),1,1,1))
obs = {
    "state": state_input.to("cuda"),
    "depth": depth_input.to("cuda")
}

for action_id in range(4):
    action_new = action.clone().to("cuda")
    action_new[:,action_id] = loop
    q_value = model.policy.critic(obs, action_new)[0]
    axeses[action_id].plot(loop.cpu(), q_value.detach().cpu())
    axeses[action_id].set_title(f"Action dimension {action_id}")

plt.show()
fig.savefig(save_folder + "/reward_asymmetry.png", dpi=300)

