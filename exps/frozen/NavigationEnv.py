from envs.NavigationEnv import NavigationEnv, smooth_l1_loss_per_row, preprocess
import torch as th
from VisFly.utils.type import TensorDict
import numpy as np

cAd = lambda x: x.clone().detach()


class VelocityCtrlEnv(NavigationEnv):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(VelocityCtrlEnv, self).__init__(*args,**kwargs,)

    def _reset_attr(self, indices=None,reset_latent=False):
        super()._reset_attr(indices)

        if not self.pre_define_target:
            indices = np.arange(self.num_envs) if indices is None else indices
            for i in indices:
                vel = th.tensor([[30,0,5]])-cAd(self.position[i:i+1])
                vel_unit = vel / (vel.norm(dim=1, keepdim=True)+1e-6)
                self.target[i] = vel_unit * th.rand(1) * self.max_rand_velocity
    def get_observation(
            self,
            indices=None
    ):

        orientation = self.envs.dynamics._orientation.clone()
        head_velocity = orientation.world_to_head((self.velocity-0).T).T
        head_target_velocity = cAd(orientation.world_to_head(self.target.T).T)
        state = th.hstack([
            head_target_velocity / 10,
            self.orientation,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        obs = TensorDict({
            "state": state,
        })
        if "depth" in self.sensor_obs.keys():
            obs["depth"] = preprocess(self.sensor_obs["depth"])
        
        return obs

    def get_reward(self,predicted_obs=None) -> th.Tensor:
        # precise and stable target flight
        base_r = 0.1

        vel_r = (self.velocity - self.target).norm(dim=1)
        adaptive_beta = (self.velocity.norm(dim=1)/6).clamp_min(1.0)
        vel_r = smooth_l1_loss_per_row(vel_r, 1.0) * -0.03
        ang_r = (self.angular_velocity - 0).norm(dim=1) * -0.005

        acc_r = (self.envs.acceleration-0).norm(dim=1).pow(1.3) * -0.0015
        # acc_r = smooth_l1_loss_per_row(acc_r, th.zeros_like(acc_r)) * -0.003
        act_change_r = (self.envs.dynamics._pre_action[-2].to(self.device).T -
                        self._action.to(self.device)
                        ).norm(dim=-1) / self.envs.dynamics.dt * -0.006

        #  heading alignment
        unit_velocity = self.velocity / (self.velocity.norm(dim=1, keepdim=True)+1e-6)
        align = (unit_velocity * self.direction).sum(dim=1)
        align_r = align * self.velocity.norm(dim=1) * 0.004


        reward = {
            "reward": base_r + vel_r + ang_r + align_r
                    + act_change_r + acc_r
            ,
            "vel_r": cAd(vel_r),
            "ang_r": cAd(ang_r),
            "acc_r": cAd(acc_r),
            "align_r": cAd(align_r),
            "act_change_r": cAd(act_change_r),
        }
        return reward
