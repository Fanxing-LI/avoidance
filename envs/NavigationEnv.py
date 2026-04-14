import numpy as np
import torch.nn.functional as F
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from VisFly.utils.randomization import TargetUniformRandomizer, UniformStateRandomizer
from VisFly.utils.type import TensorDict
from gymnasium import spaces
from debug.angle_trans import transform_distance_by_angle

dl = lambda x: x.clone().detach()

max_dis = 24.
min_dis = 0.1
scale = 3.

preprocess = lambda x: 1 / (1 + th.as_tensor(x).clamp(min_dis, max_dis) / scale)

preprocess_linear = lambda x: 1 - (th.as_tensor(x).clamp(min_dis, max_dis) - min_dis) / (max_dis - min_dis)

def preprocess_new(x):
    x = th.as_tensor(x).clamp(0.1, 24)
    # 对数压缩：0.1m→-2.3, 3m→1.1, 24m→3.2
    x_log = th.log(x)
    # 归一化到 [0, 1]：近处→1, 远处→0
    return 1 - (x_log - th.log(th.tensor(0.1))) / (th.log(th.tensor(24)) - th.log(th.tensor(0.1)))

def get_along_vertical_vector(base, obj):
    base_norm = base.norm(dim=1, keepdim=True)
    obj_norm = obj.norm(dim=1, keepdim=True)
    base_normal = base / (base_norm + 1e-8)
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = vertical_vector.norm(dim=1)
    return along_obj_norm.squeeze(), vertical_obj_norm, base_norm.squeeze()

def smooth_l1_loss_per_row(pred, target, beta: float = 1.0, reduction: str = "mean"):
    """
    pred, target: tensors of shape (m, n)
    beta: transition point for Smooth L1
    reduction: "mean" (default) or "sum" over dim=1, or "none" to return (m,n)
    returns: tensor of shape (m,) when reduction is "mean" or "sum"
    """
    # import torch as th
    diff = pred - target
    abs_diff = diff.abs()
    # if beta <= 0:
    #     loss = abs_diff
    # else:
    mask = abs_diff < beta
    loss = th.where(mask, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    # if reduction == "mean":
    #     return loss.mean(dim=1)
    # if reduction == "sum":
    #     return loss.sum(dim=1)
    return loss
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
            max_rand_velocity: float = 7.0,
            min_rand_velocity: float = 3.0,
            target_random: bool = True,
            pos_target: Optional[th.Tensor] = None,
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
        self.max_rand_velocity = max_rand_velocity
        self.min_rand_velocity = min_rand_velocity
        self.success_radius = 0.5
        max_depth = np.inf
        self.observation_space.spaces["depth"] = spaces.Box(
            low=0, high=max_depth, shape=[1] + [12, 16], dtype=np.float32
        )
        self.target_randomizers = [
            TargetUniformRandomizer(
            # UniformStateRandomizer(
                    position={
                    # "mean":[15,0,1.5],"half":[14, 14, 1.0]
                    "mean":[0,0,0],"half":[5, 5, 0.0]
                    # "mean":[0,0,0],"half":[max_target_dis, max_target_dis, 1.0]
                },
                min_dis=1.0,
                max_dis=6.0,
                # is_collision_func=self.envs.sceneManager.get_point_is_collision,
                # scene_id=i // self.num_agent_per_scene
            ) for i in range(self.num_envs)
        ]

        if pos_target is not None:
            self.pos_target = th.tensor(pos_target)

        self.radius = th.rand(self.num_envs) * 0.2 + 0.1


        self.alpha = 2 / (30 + 1)  # 约 0.065

        self.pre_collision_dis = th.zeros(self.num_envs, device=self.device)


    def _reset_attr(self, indices=None,reset_latent=False):
        super()._reset_attr(indices)
        indices = np.arange(self.num_envs) if indices is None else indices
        self.radius[indices] = th.rand(len(indices)) * 0.2 + 0.1
        self.pre_collision_dis[indices] = self.collision_vector[indices].norm(dim=-1)
        for i in indices:
            if not self.pre_define_target:
                # pos, _, _, _ = self.target_randomizers[i].safe_generate(1)
                # now consider pos generation as vel
                # pos, _, _, _ = self.target_randomizers[i].safe_generate(1, position=th.zeros_like(self.position[i]))
                # vel
                # self.target[i] = pos[0]
                vel = th.tensor([[30,0,2]])-dl(self.position[i:i+1])
                # vel[:,2] = 0
                vel_unit = vel / (vel.norm(dim=1, keepdim=True)+1e-6)
                vel_rand = th.sqrt(self.min_rand_velocity ** 2 + th.rand(1) *
                                  (self.max_rand_velocity ** 2 - self.min_rand_velocity ** 2))
                self.target[i] = vel_unit * vel_rand

    def detach(self):
        super().detach()
        self.pre_collision_dis = self.pre_collision_dis.detach()

    def get_observation(
            self,
            indices=None,
            predicted_obs: Optional[Dict] = None
    ) -> Dict:

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

        obs = TensorDict({
            "state": state,
            "depth": F.max_pool2d(preprocess(self.sensor_obs["depth"]), kernel_size=4, stride=4),
            # "depth2": F.max_pool2d(preprocess_linear(self.sensor_obs["depth"]), kernel_size=4, stride=4)
            # "depth": F.max_pool2d(preprocess(self.sensor_obs["depth"]), kernel_size=4, stride=4)
            # "depth": preprocess(self.sensor_obs["depth"])
        })

        # if "depth2" in list(self.observation_space.keys()):
        #     # obs["depth2"] = th.tensor(self.sensor_obs["depth2"]).clamp(min_dis, max_dis)
        #     # max_pool2 = lambda x: F.max_pool2d(x, kernel_size=2, stride=2)
        #     # avg_pool2 = lambda x: F.avg_pool2d(x, kernel_size=2, stride=2)
        #     obs["depth"] = F.max_pool2d(preprocess(self.sensor_obs["depth2"]), kernel_size=4, stride=4)
        #
        return obs

    def get_success(self) -> th.Tensor:
        reach_bound = (self.position[:,0]<1) | (self.position[:,0]>=58.) | \
                        (self.position[:,1]>29.) | (self.position[:,1]<=-29.)
        return reach_bound

    def get_reward(self,predicted_obs=None) -> th.Tensor:
        def remap_collision_point(p, col_p, v):
            pre_shape = None
            if len(p.shape) == 3:
                pre_shape = p.shape[:-1]
                p, col_p, v = p.reshape(-1,3), col_p.reshape(-1,3), th.tile(v.unsqueeze(1),(1,5,1)).reshape(-1,3)
            p,v = dl(p), dl(v)
            col_vector = col_p - p
            v_unit_vector = v / (v.norm(dim=1, keepdim=True)+1e-6)
            col_distance = col_vector.norm(dim=1, keepdim=True) + 1e-6
            col_unit_vector = col_vector / col_distance

            col_unit_vector_proj = (col_unit_vector * v_unit_vector).sum(dim=1, keepdim=True)
            col_unit_vector_align_v = col_unit_vector_proj * v_unit_vector
            col_unit_vector_vertical_v = col_unit_vector - col_unit_vector_align_v
            col_unit_vector_vertical_v = col_unit_vector_vertical_v / (col_unit_vector_vertical_v.norm(dim=1, keepdim=True)+1e-6)
            current_angle = th.arccos(col_unit_vector_proj)
            angle = transform_distance_by_angle(
                d=col_distance,
                a=current_angle,
                min_d=0.3,
                max_d=2.0,
                max_angle_increase=0.9,
            )
            # v_unit_vector_vertical = col_unit_vector_vertical_v / (col_unit_vector_vertical_v.norm(dim=1, keepdim=True)+1e-6)
            base_x, base_y = v_unit_vector, col_unit_vector_vertical_v
            # max_dis = 4
            # min_dis = 0.3
            # angle = (col_distance.clamp(min_dis,max_dis)-min_dis) / (max_dis - min_dis) * (th.pi/2)
            new_col_vector = (th.cos(angle) * base_x + th.sin(angle) * base_y) * col_distance
            new_col_p = p + new_col_vector
            if pre_shape is not None:
                new_col_p = new_col_p.reshape(*pre_shape, 3)
            return new_col_p.detach()

        # precise and stable target flight
        base_r = 0.1

        vel_r = (self.velocity - self.target).norm(dim=1)
        adaptive_beta = (self.velocity.norm(dim=1)/6).clamp_min(1.0)
        vel_r = smooth_l1_loss_per_row(vel_r, 1.0) * -0.03
        ang_r = (self.angular_velocity - 0).norm(dim=1) * -0.02

        acc_r = (self.envs.acceleration-0).norm(dim=1).pow(1.3) * -0.003
        # acc_r = smooth_l1_loss_per_row(acc_r, th.zeros_like(acc_r)) * -0.003
        # act_change_r = ((self.envs.dynamics._pre_action[-2].to(self.device).T -
        #                 self._action.to(self.device)
        #                 ).norm(dim=-1) / self.envs.dynamics.dt).pow(2) * -0.0002

        #  heading alignment
        unit_velocity = self.velocity / (self.velocity.norm(dim=1, keepdim=True)+1e-6)
        align = (unit_velocity * self.direction).sum(dim=1)
        align_r = align * self.velocity.norm(dim=1) * 0.004

        share_factor_collision = -0.03
        # share_factor_collision = 0.0
        # collision penalty
        if self.envs.sceneManager.col_refine_steps:
            dt = th.linspace(0,  self.envs.sceneManager.col_refine_dt, self.envs.sceneManager.col_refine_steps+1)[:-1]
            dp = dt.unsqueeze(0).unsqueeze(2) * self.velocity.unsqueeze(1)
            position = self.position.unsqueeze(1) + dp
            collision_point = self.envs.collision_point
            # collision_point = remap_collision_point(position, self.collision_point, self.velocity, )
            # position = (self.position-0).unsqueeze(1) - 0
            collision_vector = collision_point - position
            collision_dis = (collision_vector-0).norm(dim=-1).clamp_min(0.) - self.radius[..., None]
            # collision_dis = (collision_vector-0).norm(dim=-1).clamp_min(0.) - 0.1
        else:
            position = self.position
            collision_point = self.collision_point
            collision_vector = collision_point - self.position
            collision_dis = collision_vector.norm(dim=-1).clamp_min(0.) - self.radius
        # approaching_point = self.envs.approaching_point
        # velocity
        thre_vel = 2.0
        weight = ((thre_vel-collision_dis).clamp(min=0, )/thre_vel).pow(2)
        # weight = 1 / (1 + ((thre_vel-collision_dis) * 0.3).clamp(min=0,))
        # col_approach_velocity = (self.velocity * collision_dir.detach()).sum(dim=1).clamp_min(0.)
        if self.envs.sceneManager.col_refine_steps:
            distances = th.hstack([self.pre_collision_dis.unsqueeze(1), collision_dis])
            col_approach_velocity = (-th.diff(distances, dim=1) * 33 * 5).detach().clamp_min(0)  # dt and density
        # col_approach_velocity = (-(self.collision_vector.norm(dim=-1) - self.pre_collision_dis).detach() * 33).clamp_min(0)
        else:
            col_approach_velocity = (-(self.collision_vector.norm(dim=-1) - self.pre_collision_dis).detach() * 33).clamp_min(0)

        # position
        k = 0.01
        func = lambda x: 2.5 * k / (x+k)
        func3 = lambda x: 7.5 * th.log(1+th.exp(-32*x))
        func2 = lambda x: -x

        radius = self.radius[..., None] if self.envs.sceneManager.col_refine_steps else self.radius
        collision_dis = (self.collision_point - position).norm(dim=-1).clamp_min(0.) - radius
        col_dis_r = func3(collision_dis) * col_approach_velocity * share_factor_collision
        col_vel_r = (col_approach_velocity.detach() * weight) * share_factor_collision * 1

        if self.envs.sceneManager.col_refine_steps:
            col_dis_r = col_dis_r.mean(dim=-1)
            col_vel_r = col_vel_r.mean(dim=-1)

        reward = {
            "reward": base_r + vel_r + ang_r + align_r
                    # + act_change_r
                      + acc_r
                    # + acc_change_r
                    + col_vel_r + col_dis_r
            ,
            # "pos_r": dl(pos_r),
            "vel_r": dl(vel_r),
            "ang_r": dl(ang_r),
            "acc_r": dl(acc_r),
            "align_r": dl(align_r),
            "col_vel_r": dl(col_vel_r),
            "col_dis_r": dl(col_dis_r),
            # "act_change_r": dl(act_change_r),
        }
        if self.envs.sceneManager.col_refine_steps:
            self.pre_collision_dis = collision_vector[:,-1,:].norm(dim=-1).clone()
        else:
            self.pre_collision_dis = self.collision_vector.norm(dim=-1).clone()
        return reward
