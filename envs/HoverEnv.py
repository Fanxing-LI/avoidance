from VisFly.envs.HoverEnv import HoverEnv2
import torch as th

from VisFly.utils.type import TensorDict


class HoverEnv(HoverEnv2):
    """
    A simple hover environment for testing purposes.
    It inherits from HoverEnv2 and sets the target position to a fixed value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([15, 0., 1.5]).reshape(1, -1)
        # self.update_cache_target()

    # def update_cache_target(self):
    #     threshold = 10.0
    #     rela_target = self.target - self.position
    #     scale = rela_target.norm(dim=1, keepdim=True).clamp(min=threshold) / threshold
    #     rela_target = self.target - self.position
    #     self.cache_target = (rela_target / scale + self.position).clone().detach()


    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1/9 / 2

        reward = (
                base_r +
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.01
        )

        return reward

    def get_observation(
            self,
            indices=None
    ):
        # self.update_cache_target()
        orientation = self.envs.dynamics._orientation.clone()
        rela = self.target - self.position
        head_target = orientation.world_to_head(rela.T).T
        head_velocity = orientation.world_to_head((self.velocity-0).T).T
        state = th.hstack([
            head_target / 10,
            self.orientation,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        return TensorDict({
            "state": state,
            # "depth": th.as_tensor(self.sensor_obs["depth"]/10).clamp(max=1)
        })
