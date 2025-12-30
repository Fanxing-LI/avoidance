import matplotlib.pyplot as plt
import torch as th
from VisFly.utils.common import load_yaml_config
import os, sys
from CenterObstacleEnv import CenterObstacleEnv as NavigationEnv
from algorithms.BPTT_series.BPTT import BPTT
from VisFly.utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")

path ="BPTT_deterministic_share0.6_max12_1.zip"
path = "SHAC_checkpointBeforeFieldTest_RandScaleScene_ColDis0.1_share0.4_maxrate3_deter_sameScene_4.zip"
load_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/../std/saved/navigation/"
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/"

config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/alg_cfgs/navigation/BPTT.yaml')
env_config = load_yaml_config(os.path.dirname(os.path.abspath(__file__)) + f'/env_cfgs/navigation.yaml')

eval_env = NavigationEnv(
    **env_config["eval_env"]
)

model = BPTT.load(load_folder + path, env=eval_env)
from exps.test import Test

# test_handle = Test(
#     model=model,
#     save_path=save_folder + "/test",
#     name="critic"
# )
# test_handle.test(**config["test"])
# state = th.cat(test_handle.state_all, dim=0)


depth_map = th.full((1, 12, 16), 6.0)

# 天花板：距离2,4,6渐变（第0-5行）
depth_map[0, 0, :] = 2.0  # 第1行：距离2
depth_map[0, 1, :] = 2.0  # 第2行：距离2
depth_map[0, 2, :] = 4.0  # 第3行：距离4
depth_map[0, 3, :] = 4.0  # 第4行：距离4
depth_map[0, 4, :] = 6.0  # 第5行：距离6
depth_map[0, 5, :] = 6.0  # 第6行：距离6

# 地板：距离2,4,6渐变（第6-11行）
depth_map[0, 6, :] = 2.0   # 第7行：距离2
depth_map[0, 7, :] = 2.0   # 第8行：距离2
depth_map[0, 8, :] = 4.0   # 第9行：距离4
depth_map[0, 9, :] = 4.0   # 第10行：距离4
depth_map[0, 10, :] = 6.0  # 第11行：距离6
depth_map[0, 11, :] = 6.0  # 第12行：距离6

# 柱子：距离1，宽4列（第4-8列），高6行（第3-8行）
depth_map[0, 3:9, 4:8] = 1.0

vel = [4,6,8]

state_input = th.tensor([[3,0,0,
                         1,0,0,0,
                         0,0,0,
                         0,0,0]],dtype=th.float32)

depth = depth_map.unsqueeze(0)
fig, axeses = FigFon.get_figure_axes((len(vel),4), HeightScale=1, Column=2, sharey=True ,share_legend=True)

loop = th.arange(-0.4,0.4,0.01)
action = th.zeros((len(loop), 4))
state_input = th.tile(state_input, (len(loop),1))
depth_input = th.tile(depth, (len(loop),1,1,1)).to(th.float32)
# obs = {
#     "state": state_input.to("cuda"),
#     "depth": 1/(1+depth_input.to("cuda")/4)
# }
for i,v in enumerate(vel):
    state_input[:,7] = v/10
    obs = {
        "state": state_input.to("cuda"),
        "depth": 1/(1+depth_input.to("cuda")/4)
    }
    for action_id in range(4):
        action_new = action.clone().to("cuda")
        action_new[:,action_id] = loop
        q_value = model.policy.critic(obs, action_new)[0]
        # q_value = th.stack(model.policy.critic(obs, action_new)).squeeze().T
        axeses[i,action_id].plot(loop.cpu(), q_value.detach().cpu())
        axeses[i,action_id].set_title(f"Action dimension {action_id}")
    # legend
# for ax in axeses:
#     ax.legend([f"vel {v}" for v in vel], title="Forward Velocity")
#     ax.set_xlabel("Action value")
#     ax.set_ylabel("Q value")
# FigFon.set_shared_legend(axes=axeses, legend_labels=[f"vel {v}" for v in vel], title="Forward Velocity", ncol=2)
plt.tight_layout()
plt.show()
fig.savefig(save_folder + "/reward_asymmetry.png", dpi=300)

