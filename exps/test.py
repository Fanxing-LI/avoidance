from typing import Optional
import torch as th
import matplotlib.pyplot as plt
from VisFly.utils.evaluate import TestBase
import numpy as np


class Test(TestBase):
    def __init__(self,
                 model,
                 name,
                 save_path: Optional[str] = None,
                 ):
        super(Test, self).__init__(model=model, name=name, save_path=save_path, )
        self.target_all = []
        self.target_dis_all = []
        self.center_all = []

    def draw(self, names=None):
        state_data = th.stack(self.state_all).cpu()
        col_dis = th.stack([collision["col_dis"] for collision in self.collision_all])
        action = th.stack([th.tensor(a) for a in self.action_all]).cpu()
        t = np.stack(self.t)[:, 0]
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(7, 4))
            plt.subplot(2, 3, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(2, 3, 2)

            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(2, 3, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            plt.legend()
            plt.subplot(2, 3, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.subplot(2, 3, 5)
            plt.plot(t[:-1], action[:, i, :], label=["a", "awx", "awy", "awz"])
            plt.legend()
            plt.subplot(2, 3, 6)
            plt.plot(t, col_dis[:, i], label="closest distance")
            plt.tight_layout()
            plt.show()

            # plt.legend()
        # fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
            # axes.plot(t, col_dis)
            # axes.set_xlabel("t/s")
            # axes.set_ylabel("closest distance/m")
        # plt.show()
        # print("rewards_sum: ", np_rewards)

        return [fig, ]

