import os

import numpy as np
from habitat_sim.sensor import SensorType

from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from habitat_sim import SensorType
from gymnasium import spaces
from VisFly.utils.randomization import TargetUniformRandomizer
from VisFly.utils.type import TensorDict

dl = lambda  x: x.clone().detach()

def get_along_vertical_vector(base, obj):
    base_norm = base.norm(dim=1, keepdim=True)
    obj_norm = obj.norm(dim=1, keepdim=True)
    base_normal = base / (base_norm + 1e-8)
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1)
    return along_obj_norm.squeeze(), vertical_obj_norm, base_norm.squeeze()


class NavigationEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = {},
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = True,
            max_target_dis: float = 7.0,
            target_random: bool = True,
            *args,
            **kwargs
    ):

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
            *args,
            **kwargs
        )

        self.pre_define_target = target is not None
        if target is not None:
            if not isinstance(target[0], list):
                self.target = th.ones((self.num_envs, 1)) @ th.as_tensor(target)
            else:
                self.target = th.as_tensor(target)
        else:
            self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([[15, 0., 1]])
        self.max_target_dis = max_target_dis
        self.success_radius = 0.5

        self.target_randomizers = [
            TargetUniformRandomizer(
                position={
                    "mean":[0,0,0],"half":[max_target_dis, max_target_dis, 1.0]
                },
                min_dis=0.5,
                max_dis=max_target_dis,
                is_collision_func=self.envs.sceneManager.get_point_is_collision,
                scene_id=i // self.num_agent_per_scene
            ) for i in range(self.num_envs)
        ]

    def _reset_attr(self, indices=None):
        super()._reset_attr(indices)

        if not self.pre_define_target:
            indices = np.arange(self.num_envs) if indices is None else indices
            for i in indices:
                pos, _, _, _ = self.target_randomizers[i].safe_generate(1, position=dl(self.position[i]))
                self.target[i] = pos[0]

    def get_observation(
            self,
            indices=None
    ) -> Dict:

        # target cat
        if self.max_target_dis:
            rela = self.target - self.position
            unit_rela = rela / (rela.norm(dim=1, keepdim=True)+1e-6)
            distance = rela.norm(dim=1, keepdim=True).clip(0, self.max_target_dis)
            new_target = (unit_rela * distance+self.position).detach()
        else:
            new_target = self.target

        orientation = self.envs.dynamics._orientation.clone()
        rela = new_target - self.position
        rela_dis = rela.norm(dim=1, keepdim=True)
        # rela_dis = th.ones((self.num_envs, 1), device=self.device)
        normal_rela = rela #/ rela_dis.clamp_min(1.0).detach()
        head_target = orientation.world_to_head(normal_rela.T).T
        head_velocity = orientation.world_to_head((self.velocity-0).T).T
        state = th.hstack([
            # rela / 10,
            head_target,
            self.orientation,
            # self.velocity / 10,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        obs = TensorDict({
            "state": state,
            "depth": 1/(1+th.tensor(self.sensor_obs["depth"]).clamp_min(0.1))
        })

        if "depth2" in list(self.observation_space.keys()):
            obs["depth2"] = th.tensor(self.sensor_obs["depth2"])
        return obs

    def get_success(self) -> th.Tensor:
        return th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
        return ((self.position - self.target).norm(dim=1) <= self.success_radius) & \
                (self.velocity.norm(dim=1) <= 0.05)

    def get_reward(self) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1

        pos_r = (self.position - self.target).norm(dim=1) * -0.005
        vel_r = (self.velocity - 0).norm(dim=1) * -0.003
        ang_r = (self.angular_velocity - 0).norm(dim=1) * -0.005

        # act_r = self._action.norm(dim=1).cpu() * -0.001
        act_change_r = (self.envs.dynamics._pre_action[-1].to(self.device).T -
                        self._action.to(self.device)
                        ).norm(dim=-1) * -0.000

        #  heading alignment
        unit_velocity = self.velocity / (self.velocity.norm(dim=1, keepdim=True)+1e-6)
        align = (unit_velocity * self.direction).sum(dim=1)
        align_r = align * self.velocity.norm(dim=1) * 0.002

        # collision penalty
        thre_vel = 1.0
        collision_dis = self.collision_vector.norm(dim=1).clamp_min(0.)
        collision_dir = self.collision_vector / (collision_dis.unsqueeze(1)+1e-6)
        # velocity
        col_approach_velocity = (self.velocity * collision_dir.detach()).sum(dim=1).clamp_min(0.)
        col_vel_r = col_approach_velocity * (thre_vel-collision_dis.detach()).clamp(min=0, ).pow(2) * -0.003

        # position
        k = 0.02
        func = lambda x: k / (x+k)
        func2 = lambda x: -x
        col_dis_r = func(collision_dis) * -0.08

        reward = {
            "reward": base_r + pos_r + vel_r + ang_r + align_r
                    + act_change_r
                      + col_vel_r + col_dis_r
            ,
            "pos_r": dl(pos_r),
            "vel_r": dl(vel_r),
            "ang_r": dl(ang_r),
            "align_r": dl(align_r),
            "col_vel_r": dl(col_vel_r),
            "col_dis_r": dl(col_dis_r),
            # "act_r": dl(act_r),
            "act_change_r": dl(act_change_r),
        }
        return reward
