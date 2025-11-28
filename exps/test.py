from typing import Optional
import torch as th
import matplotlib.pyplot as plt
from VisFly.utils.evaluate import TestBase
import numpy as np
import cv2

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

    def test(self, *args, **kwargs):
        super().test(*args, **kwargs)
        self._img_names.append("depth_merge")
        for obs in self.obs_all:
            depth0 = obs["depth"].permute(0, 2, 3, 1).cpu().numpy()
            merge = th.zeros(depth0.shape[0], 1, 48, 64)
            depth1 = obs["depth2"]
            # shape = depth1.shape
            # enlarge depth0 to depth1 size using torchvision
            for i, d in enumerate(depth0):
                merge[i, 0] = th.from_numpy(cv2.resize(d, (64,48), interpolation=cv2.INTER_NEAREST))*10
            # resized_img = cv2.resize(depth0, (48,64), interpolation=cv2.INTER_NEAREST)
            obs["depth_merge"] = np.concatenate([merge, depth1], axis=2)
        self.save_video()

    def draw(self, names=None):
        state_data = th.stack(self.state_all).cpu()
        col_dis = th.stack([collision["col_dis"] for collision in self.collision_all])
        action = th.stack([th.tensor(a) for a in self.action_all]).cpu()
        t = np.stack(self.t)[:, 0]
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(7, 4))
            plt.subplot(3, 3, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(3, 3, 2)

            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(3, 3, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            plt.legend()
            plt.subplot(3, 3, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.subplot(3, 3, 5)
            plt.plot(t[:-1], action[:, i, :], label=["a", "awx", "awy", "awz"])
            plt.legend()
            plt.subplot(3, 3, 6)
            plt.plot(t, col_dis[:, i], label="closest distance")
            plt.subplot(3, 3, 7)
            plt.plot(t, state_data[:,i,13:16], label=["x","y","z"])
            plt.title("acceleration")
            plt.subplot(3, 3, 8)
            plt.plot(t[:-1], (state_data[1:,i,13:16]-state_data[:-1,i,13:16])/(t[2]-t[1]), label=["ax","ay","az"])
            plt.title("jerk")
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

