import rosbag
import numpy as np
import os
import sys
import torch as th
import json
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from VisFly.envs.HoverEnv import HoverEnv2
from scipy import interpolate
from VisFly.utils.type import bound
from VisFly.utils.FigFashion.FigFashion import FigFon

HOVER_ACC = 9.8
FigFon.set_fashion("IEEE")


class SimplifiedBagPlot:
    def __init__(self, path, step_type=None):
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.m = 0.671
        self.bag_file = path
        self.step_type = step_type
        self.bag = rosbag.Bag(self.bag_file)
        self.topics = ['/bfctrl/cmd', '/bfctrl/local_odom', '/mavros/imu/data', '/bfctrl/traj_start_trigger']

        # 原始数据容器
        self.actions_cmd = []          # thrust, wx, wy, wz
        self.actions_time = []
        self.state_bf = []              # 13维状态
        self.state_bf_time = []
        self.imu_angle_v = []            # wx, wy, wz, acc_z
        self.imu_time = []
        self.trigger = []
        self.trigger_time = []

        # 仿真数据容器
        self.anglevel = []               # 仿真角速度
        self.accz = []                   # 仿真z轴加速度
        self.action_all = []              # 仿真所用命令
        self.state_all = []                # 仿真状态
        self.sim_t = []                    # 仿真时间

        self.env = HoverEnv2(
            num_agent_per_scene=1,
            visual=False,
            max_episode_steps=512,
            scene_kwargs={"path": self.scene_path},
            dynamics_kwargs={"ctrl_dt": 0.01, "comm_delay": 0.03, "cfg": "drone_d435i_jetson_orin_nx"}
        )
        self._parse_bag()
        self._align_data()
        self._load_config("VisFly/configs/drone/drone_d435i_jetson_orin_nx.json")
        self._run_simulation()

    def _parse_bag(self):
        """解析bag文件，提取所有话题数据"""
        for topic, msg, t in self.bag.read_messages(self.topics):
            t_sec = t.to_sec()
            if topic == '/bfctrl/cmd':
                self.actions_cmd.append([msg.thrust, msg.angularVel.x, msg.angularVel.y, msg.angularVel.z])
                self.actions_time.append(t_sec)
            elif topic == '/bfctrl/traj_start_trigger':
                self.trigger.append(msg.data)
                self.trigger_time.append(t_sec)
            elif topic == '/bfctrl/local_odom':
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                v = msg.twist.twist.linear
                a = msg.twist.twist.angular
                self.state_bf.append([p.x, p.y, p.z, o.w, o.x, o.y, o.z,
                                      v.x, v.y, v.z, a.x, a.y, a.z])
                self.state_bf_time.append(t_sec)
            elif topic == '/mavros/imu/data':
                self.imu_angle_v.append([msg.angular_velocity.x, msg.angular_velocity.y,
                                         msg.angular_velocity.z, msg.linear_acceleration.z])
                self.imu_time.append(t_sec)

        # 转换为张量/数组
        self.actions_real = th.as_tensor(np.array(self.actions_cmd, dtype=np.float32))
        self.state_bf = th.as_tensor(np.array(self.state_bf, dtype=np.float32))
        self.state_bf_time = np.array(self.state_bf_time)
        self.actions_time = np.array(self.actions_time)
        self.trigger_time = np.array(self.trigger_time)
        self.imu_angle_v = np.array(self.imu_angle_v, dtype=np.float32)
        self.imu_time = np.array(self.imu_time)

    def _load_config(self, path):
        """加载无人机配置"""
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"]))

    def _normalize(self, command, normal_range=(-1, 1)):
        """将真实命令归一化到[-1,1]区间（仿真输入）"""
        thrust_scale = HOVER_ACC
        normalized_thrust = (command[:, :1] - HOVER_ACC) / thrust_scale

        bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (normal_range[1] - normal_range[0])
        bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
        normalized_bodyrate = (command[:, 1:] - bodyrate_bias) / bodyrate_scale

        return th.hstack([normalized_thrust, normalized_bodyrate])

    def _align_data(self):
        """根据触发时间裁剪并对齐数据"""
        self.start_time, self.end_time = self.trigger_time[0], self.trigger_time[-1]

        def cut(data, times):
            mask = (times >= self.start_time) & (times <= self.end_time)
            return data[mask], times[mask]

        self.ori_action, self.ori_action_time = cut(self.actions_real, self.actions_time)
        self.imu_angle_v, self.imu_time = cut(self.imu_angle_v, self.imu_time)
        self.state_bf, self.state_bf_time = cut(self.state_bf, self.state_bf_time)

        # 时间归零
        self.ori_action_time -= self.start_time
        self.imu_time -= self.start_time
        self.state_bf_time -= self.start_time

        # 创建命令插值函数（使用 nearest 保持阶跃特性）
        interps = [interpolate.interp1d(
            self.ori_action_time, self.ori_action[:, i].numpy(),
            kind='nearest',
            fill_value=(self.ori_action[0, i].item(), self.ori_action[-1, i].item()),
            bounds_error=False
        ) for i in range(self.ori_action.shape[1])]
        self.act_f = lambda t: np.array([f(t) for f in interps]).T

    def _run_simulation(self):
        """使用仿真环境复现飞行过程"""
        self.env.reset()
        duration = self.end_time - self.start_time
        while self.env.envs.dynamics.t < duration:
            t = self.env.envs.dynamics.t
            action = self._normalize(th.as_tensor(self.act_f(t)))
            self.sim_t.append(t.clone().detach())

            self.env.envs.dynamics.step(action)
            self.action_all.append(th.from_numpy(self.act_f(t)))
            self.state_all.append(self.env.state)
            self.anglevel.append(self.env.angular_velocity.clone())
            self.accz.append(self.env.envs.dynamics.acceleration[:, 2])

        # 拼接仿真数据
        self.anglevel = th.cat(self.anglevel, dim=0)
        self.accz = th.as_tensor(self.accz, dtype=th.float32)
        self.action_all = th.cat(self.action_all, dim=0)
        self.state_all = th.cat(self.state_all, dim=0)
        self.sim_t = th.cat(self.sim_t)

    # ---------- 绘图辅助 ----------
    def _filter_to_limit(self, time, data, limit=3.0):
        """截取时间范围内的数据"""
        mask = time <= limit
        return time[mask], data[mask]

    def _filter_outliers(self, data, low=-5, high=26):
        """将异常值替换为 NaN（用于推力数据）"""
        return np.where((data >= low) & (data <= high), data, np.nan)

    # ---------- 公共绘图方法 ----------
    def plot_step_response(self, ax):
        """绘制阶跃响应（根据 step_type 自动选择）"""
        limit = 3.0

        # 定义不同轴的映射 (cmd_col, imu_col, sim_vel_col, title)
        axis_map = {
            'x': (1, 0, 0, r'$\omega_x$'),
            'y': (2, 1, 1, r'$\omega_y$'),
            'z': (3, 2, 2, r'$\omega_z$'),
        }

        if self.step_type in axis_map:
            cmd_col, imu_col, sim_col, title = axis_map[self.step_type]

            # 过滤时间
            t_cmd, cmd = self._filter_to_limit(self.ori_action_time, self.ori_action[:, cmd_col].numpy(), limit)
            t_imu, imu = self._filter_to_limit(self.imu_time, self.imu_angle_v[:, imu_col], limit)
            t_sim, sim = self._filter_to_limit(self.sim_t, self.anglevel.numpy()[:, sim_col], limit)
            _, sim_cmd = self._filter_to_limit(self.sim_t, self.action_all.numpy()[:, cmd_col], limit)

            ax.plot(t_cmd, cmd, label='Cmd')
            ax.plot(t_sim, sim_cmd, label='Sim Cmd', linestyle=':')
            ax.plot(t_imu, imu, label='Real')
            ax.plot(t_sim, sim, label='Sim', linestyle='--')
            ax.set_title(title, fontsize=20)

        elif self.step_type == 'thrust':
            t_cmd, cmd = self._filter_to_limit(self.ori_action_time, self.ori_action[:, 0].numpy(), limit)
            t_imu, imu = self._filter_to_limit(self.imu_time, self.imu_angle_v[:, 3], limit)
            t_sim, sim = self._filter_to_limit(self.sim_t, self.accz.numpy() + 9.8, limit)
            _, sim_cmd = self._filter_to_limit(self.sim_t, self.action_all.numpy()[:, 0], limit)

            # 剔除异常值
            cmd = self._filter_outliers(cmd)
            sim_cmd = self._filter_outliers(sim_cmd)
            imu = self._filter_outliers(imu)
            sim = self._filter_outliers(sim)

            ax.plot(t_cmd, cmd, label='Cmd')
            ax.plot(t_sim, sim_cmd, label='Sim Cmd', linestyle=':')
            ax.plot(t_imu, imu, label='Real')
            ax.plot(t_sim, sim, label='Sim', linestyle='--')
            ax.set_title(r'$f$', fontsize=20)

        else:
            raise ValueError(f"Unknown step_type: {self.step_type}")

        ax.grid(True, linewidth=2, alpha=0.6, zorder=0)
        ax.set_xlim(0, limit)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.legend(loc='best', fontsize=10)

    def plot_policy_response(self, axs):
        """绘制完整策略响应（1×4布局）"""
        titles = ['X-axis Angular Velocity', 'Y-axis Angular Velocity',
                  'Z-axis Angular Velocity', 'Z-axis Thrust']
        for i, ax in enumerate(axs):
            if i < 3:  # 角速度
                ax.plot(self.ori_action_time, self.ori_action[:, i+1].numpy(), label='Cmd')
                ax.plot(self.imu_time, self.imu_angle_v[:, i], label='Real')
                ax.plot(self.sim_t, self.anglevel.numpy()[:, i], label='Sim', linestyle='--')
            else:       # 推力
                real = self._filter_outliers(self.imu_angle_v[:, 3])
                sim = self._filter_outliers(self.accz.numpy() + 9.8)
                cmd = self._filter_outliers(self.ori_action[:, 0].numpy())
                ax.plot(self.ori_action_time, cmd, label='Cmd')
                ax.plot(self.imu_time, real, label='Real')
                ax.plot(self.sim_t, sim, label='Sim', linestyle='--')

            ax.set_title(titles[i])
            ax.grid(True, linewidth=2, alpha=0.6, zorder=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc='best', fontsize=10)

    def plot_z_velocity(self, ax):
        """绘制推力测试中的z轴速度对比"""
        real_vz = self.state_bf[:, 9].numpy()   # 真实z速度 (索引8)
        real_t = self.state_bf_time
        sim_vz = self.state_all[:, 9].numpy()   # 仿真z速度
        sim_t = self.sim_t.numpy()

        ax.plot(real_t, real_vz, label='Real Vel')
        ax.plot(sim_t, sim_vz, label='Sim Vel', linestyle='--')
        ax.set_title('Z Velocity during Thrust Step Test')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z Velocity (m/s)')
        ax.grid(True, linewidth=1.5, alpha=0.6)
        ax.legend(loc='best')

    def plot_z_position(self, ax):
        """绘制推力测试中的z轴位置对比"""
        real_z = self.state_bf[:, 2].numpy()    # 真实z位置 (索引2)
        real_t = self.state_bf_time
        sim_z = self.state_all[:, 2].numpy()    # 仿真z位置
        sim_t = self.sim_t.numpy()

        ax.plot(real_t, real_z, label='Real Pos')
        ax.plot(sim_t, sim_z, label='Sim Pos', linestyle='--')
        ax.set_title('Z Position during Thrust Step Test')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z Position (m)')
        ax.grid(True, linewidth=1.5, alpha=0.6)
        ax.legend(loc='best')


if __name__ == "__main__":
    # 创建 3行2列 的子图布局，每个子图独立图例
    fig, axs = FigFon.get_figure_axes((3, 2), share_legend=False, Column=1)

    # 四个阶跃响应子图
    x_step = SimplifiedBagPlot('plots/pid_test/pid_bag/x_e_0_2_t_2.bag', step_type='x')
    x_step.plot_step_response(axs[0, 0])

    y_step = SimplifiedBagPlot('plots/pid_test/pid_bag/y_e_0_2_t_2.bag', step_type='y')
    y_step.plot_step_response(axs[0, 1])

    z_step = SimplifiedBagPlot('plots/pid_test/pid_bag/z_e_0_2_t_2.bag', step_type='z')
    z_step.plot_step_response(axs[1, 0])

    thrust_step = SimplifiedBagPlot('plots/pid_test/pid_bag/t_3_5_axis.bag', step_type='thrust')
    thrust_step.plot_step_response(axs[1, 1])

    # 第五子图：z轴速度对比
    thrust_step.plot_z_velocity(axs[2, 0])

    # 第六子图：z轴位置对比
    thrust_step.plot_z_position(axs[2, 1])

    # 自动调整布局
    plt.tight_layout()
    fig.savefig('plots/step_response.png', format='png', bbox_inches='tight', dpi=400)
    print("已保存 step_response.png")