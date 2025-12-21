import torch as th
from typing import Dict

from envs.NavigationEnv import NavigationEnv


class CenterObstacleEnv(NavigationEnv):
    def __init__(self, *args, **kwargs):
        super(CenterObstacleEnv, self).__init__(*args, **kwargs)

    def get_observation(self, indices=None) -> Dict:
        obs = super().get_observation(indices)
        # When the agent moves beyond a threshold, override depth with a synthetic map
        # if self.position[0, 0] > 5:
        #     # Create a (num_envs, 12, 16) depth tensor
        #     fake_depth = th.full((self.num_envs, 12, 16), 20.0, device=self.device)
        #
        #     # Top three rows: gradient 15 -> 10 -> 5 (simulate ceiling)
        #     fake_depth[:, 0, :] = 15.0
        #     fake_depth[:, 1, :] = 10.0
        #     fake_depth[:, 2, :] = 5.0
        #
        #     # Bottom three rows: gradient 15 -> 10 -> 5 (simulate ground)
        #     fake_depth[:, -3, :] = 5.0
        #     fake_depth[:, -2, :] = 10.0
        #     fake_depth[:, -1, :] = 15.0
        #
        #     # Middle four columns set to 3 across all rows
        #     mid_start = (16 // 2) - 2  # 6 for width=16
        #     mid_end = mid_start + 4    # columns 6..9
        #     fake_depth[:, :, mid_start:mid_end] = 3.0
        #
        #     # Override depth in observation
        #     obs["depth"] = (1/(1+th.tensor(fake_depth/4))).unsqueeze(dim=0)

        return obs
