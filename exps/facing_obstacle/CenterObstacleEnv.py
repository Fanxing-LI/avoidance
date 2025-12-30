import torch as th
from typing import Dict

from envs.NavigationEnv import NavigationEnv


class CenterObstacleEnv(NavigationEnv):
    def __init__(self, *args, **kwargs):
        super(CenterObstacleEnv, self).__init__(*args, **kwargs)

    def get_observation(self, indices=None) -> Dict:
        obs = super().get_observation(indices)

        return obs

    def get_success(self) -> th.Tensor:
        return self.position[:,0]>=24