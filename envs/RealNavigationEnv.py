from envs.NavigationEnv import NavigationEnv, dl
import torch.nn.functional as F

from VisFly.utils.type import TensorDict
import torch as th

class RealNavigationEnv(NavigationEnv):
    def __init__(self,
                 **kwargs,
                 ):
        super(RealNavigationEnv, self).__init__(**kwargs,
                                                )

    def get_success(self) -> th.Tensor:
        reach_bound = (self.position[:,0]<0) | (self.position[:,0]>=58.) | \
                        (self.position[:,1]>30.) | (self.position[:,1]<=-30.)
        return reach_bound

    def get_observation(
            self,
            indices=None,
            predicted_obs=None
    ):

        if hasattr(self, "pos_target"):
            pos_target = self.pos_target.repeat(self.num_scene, 1)
            rela_target = (pos_target - self.position)
            self.target = ((rela_target
                           / rela_target.norm(dim=1, keepdim=True))
                           * (rela_target.norm(dim=1, keepdim=True)/1).clamp_max(self.max_rand_velocity)
                           )

        orientation = self.envs.dynamics._orientation.clone()
        head_velocity = orientation.world_to_head((self.velocity-0).T).T
        head_target_velocity = dl(orientation.world_to_head(self.target.T).T)
        state = th.hstack([
            # rela / 10,
            head_target_velocity / 10,
            self.orientation,
            # self.velocity / 10,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)

        max_dis = 24.
        min_dis = 0.1
        scale = 3.

        preprocess = lambda x: 1 / (1 + th.as_tensor(x).clamp(min_dis, max_dis) / scale)
        mean_pool30 = lambda x: F.avg_pool2d(th.as_tensor(x), kernel_size=30, stride=30)
        mean_pool15 = lambda x: F.avg_pool2d(th.as_tensor(x), kernel_size=15, stride=15)
        max_pool15 = lambda x: F.max_pool2d(th.as_tensor(x), kernel_size=15, stride=15)
        mean_pool6 = lambda x: F.avg_pool2d(th.as_tensor(x), kernel_size=6, stride=6)
        max_pool5 = lambda x: F.max_pool2d(th.as_tensor(x), kernel_size=5, stride=5)

        max_pool2 = lambda x: F.max_pool2d(x, kernel_size=2, stride=2)
        avg_pool2 = lambda x: F.avg_pool2d(x, kernel_size=2, stride=2)

        flex_max_pool = lambda x, k: F.max_pool2d(th.as_tensor(x), kernel_size=k, stride=k)
        flex_avg_pool = lambda x, k: F.avg_pool2d(th.as_tensor(x), kernel_size=k, stride=k)
        flex_min_pool = lambda x, k: -F.max_pool2d(-th.as_tensor(x), kernel_size=k, stride=k)
        dim = 30
        k1=10
        obs = TensorDict({
            "state": state,
            # "depth": preprocess(mean_pool30(self.sensor_obs["depth"])),
            # "depth": max_pool5(preprocess(mean_pool6(self.sensor_obs["depth"]))),
            # "depth": flex_max_pool(preprocess(flex_max_pool(self.sensor_obs["depth"], k1)),int(dim/k1)),
            "depth": self.sensor_obs["depth"],
        })

        # if "depth2" in list(self.observation_space.keys()):
        # obs["depth2"] = self.sensor_obs["depth"]

        # obs["depth"] = mean_pool32(obs["depth"])
        #     # obs["depth"] = max_pool2(1/(1+avg_pool2(obs["depth2"]/scale)))

        return obs