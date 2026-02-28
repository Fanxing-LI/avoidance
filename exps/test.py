from typing import Optional
import torch as th
import matplotlib.pyplot as plt
from VisFly.utils.evaluate import TestBase, render_fig
import numpy as np
import cv2
import copy


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

    def test(
            self,
            policy=None,
            world=None,
            # model=None,
            is_fig: bool = False,
            is_video: bool = False,
            is_sub_video: bool = False,
            is_fig_save: bool = False,
            is_video_save: bool = False,
            render_kwargs={},

    ):
        if policy is None:
            policy = self.model.policy
        env = self.env

        # done_all = th.full((env.num_envs,), False)
        obs = env.reset(is_test=True)
        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        self.obs_all.append(obs)
        self.state_all.append(env.extend_state.clone().detach())
        self.info_all.append([{} for _ in range(env.num_envs)])
        self.t.append(env.t.clone())
        self.collision_all.append({"col_dis": env.collision_dis,
                                   "is_col": env.is_collision,
                                   "col_pt": env.collision_point})
        self.is_success = th.zeros(env.num_agent)
        record_ls = th.arange(0, env.num_agent).tolist()
        agent_index = [i for i in range(env.num_agent)]
        self.eq_r = []
        self.eq_l = []
        self.finish_t = th.zeros(env.num_agent)

        while True:
            with th.no_grad():
                action = policy.predict(obs, deterministic=True, sample=False)
                if isinstance(action, tuple):
                    action = action[0]
                # obs, reward, done, info = env.step(action, is_test=True)
                if world is not None:
                    obs, reward, done, info = env.step(action, is_test=True, latent_func=world.step)
                else:
                    obs, reward, done, info = env.step(action, is_test=True)
                # = env.get_observation(), env.reward, env.done, env.info
                col_dis, is_col, col_pt = env.collision_dis, env.is_collision, env.collision_point
                state = env.state
                self.collision_all.append({"col_dis": col_dis, "is_col": is_col, "col_pt": col_pt})

            self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(env.extend_state.clone().detach())
            self.obs_all.append(obs)
            self.info_all.append(copy.deepcopy(info))
            self.t.append(env.t.clone())
            print("current step:", env._step_count[0])
            if env.visual:
                # render_kwargs["points"] = env.target
                render_image = cv2.cvtColor(env.render(**render_kwargs)[0], cv2.COLOR_RGBA2RGB)
                self.render_image_all.append(render_image)
            # done_all[done] = True
            if done.any():
                dones_i = th.where(th.as_tensor(done))[0]
                for i in reversed(record_ls):
                    if i in dones_i:
                    # if env.position[i,0]>=56:
                        record_ls.remove(i)
                        if env.envs.once_collided[i]:
                            self.is_success[i] = 0
                        else:
                            self.is_success[i] = 1
        # for i in reversed(agent_index):
        #             if done[i]:
                        self.eq_r.append(info[i]['episode']['r'].item())
                        self.eq_l.append(info[i]['episode']['l'].item())
                        self.finish_t[i] = env.t[i].item()
                        agent_index.remove(i)

            if len(agent_index) == 0:
                break

        mean_r = th.as_tensor(self.eq_r, dtype=th.float32).mean().item()
        mean_l = th.as_tensor(self.eq_l, dtype=th.float32).mean().item()
        # print(f"Average Rewards:{mean_r}, Average Length:{mean_l}")
        # save state_all and obs_all
        figs = self.draw()

        if is_fig:
            for fig in figs:
                # plt.show fig
                render_fig(fig)

        if is_fig_save:
            for i, fig in enumerate(figs):
                self.save_fig(fig, c=i)

        if is_video:
            self.play(is_sub_video=is_sub_video)
            if is_video_save:
                self.save_video()
        render_video = th.as_tensor(np.stack(self.render_image_all, axis=0)).unsqueeze(0) if len(self.render_image_all) > 0 else None
        # return figs, render_video, mean_r, mean_l

        # of obs has key depth2
        if "depth2" in self.obs_all[0]:
            depth2_size = self.obs_all[0]["depth2"].shape[-2:]
            self._img_names.append("depth_merge")
            for obs in self.obs_all:
                depth0 = obs["depth"].permute(0, 2, 3, 1).cpu().numpy()
                merge = th.zeros(depth0.shape[0], 1, *depth2_size)
                depth1 = obs["depth2"]
                # shape = depth1.shape
                # enlarge depth0 to depth1 size using torchvision
                for i, d in enumerate(depth0):
                    merge[i, 0] = th.from_numpy(cv2.resize(d, (depth2_size[1], depth2_size[0]), interpolation=cv2.INTER_NEAREST))*10
                # resized_img = cv2.resize(depth0, (48,64), interpolation=cv2.INTER_NEAREST)
                obs["depth_merge"] = np.concatenate([merge, depth1], axis=2)
            self.save_video()
            self.save_test_info()

    def save_test_info(self):
        # average of max speed during test
        state_data = th.stack(self.state_all).cpu().numpy()
        max_speeds = np.max(np.linalg.norm(state_data[:,:,7:10], axis=2), axis=0)
        for i, max_speed in enumerate(max_speeds):
            print(f"Agent {i} max speed during test: {max_speed:.2f} m/s")

        # average of mean speed during test
        mean_speeds = 56 / self.finish_t.numpy()
        print(f"\nMean speeds during test:{mean_speeds.mean()}")

        success = np.array([info["episode"]["extra"]["collision"] for info in self.info_all[-1]]).astype(int)
        success_rate = 1-success.sum().item() / success.shape[0]
        print(f"Env {i} success rate: {success_rate*100:.2f}%")
        print("type1 col index:",th.where(th.as_tensor(success)==1)[0])

        print("success rate2:", self.is_success.sum().item()/self.model.env.num_envs)
        print("type2 col index:", th.where(self.is_success==0)[0])

        # save infos in md file
        if self.save_path is not None:
            with open(self.save_path + "/test_info.md", "w") as f:
                f.write(f"# Test Info for {self.name}\n\n")
                f.write(f"## Max Speeds (m/s)\n")
                for i, max_speed in enumerate(max_speeds):
                    f.write(f"- Agent {i}: {max_speed:.2f} m/s\n")
                f.write(f"\n## Mean Speeds (m/s)\n")
                for i, mean_speed in enumerate(mean_speeds):
                    f.write(f"- Agent {i}: {mean_speed:.2f} m/s\n")
                f.write(f"\n## Success Rates\n")
                for i in range(self.model.env.num_envs):
                    success = np.array([info["episode"]["extra"]["collision"] for info in self.info_all[-1]]).astype(int)
                    success_rate = 1 - success.sum().item() / success.shape[0]
                    f.write(f"- Env {i}: {success_rate*100:.2f}%\n")
                f.write(f"\nOverall Success Rate: {self.is_success.sum().item()/self.model.env.num_envs*100:.2f}%\n")

    def draw(self, names=None):
        state_data = th.stack(self.state_all).cpu()
        col_dis = th.stack([collision["col_dis"] for collision in self.collision_all])
        action = th.stack([th.tensor(a) for a in self.action_all]).cpu()
        t = np.stack(self.t)[:, 0]
        figs = []
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(7, 4))
            figs.append(fig)
            fig.suptitle(self.name+f"_agent_{i}")
            plt.subplot(3, 3, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(3, 3, 2)

            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(3, 3, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            v_norm = np.linalg.norm(state_data[:, i, 7:10], axis=1)
            plt.plot(t, v_norm, label="v_norm")
            plt.legend()
            plt.subplot(3, 3, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.subplot(3, 3, 5)
            plt.plot(t[:-1], action[:, i, :], label=["a", "awx", "awy", "awz"])
            plt.legend()
            plt.subplot(3, 3, 6)
            plt.plot(t, col_dis[:, i], label="closest distance")
            plt.ylim(0, 2)
            plt.subplot(3, 3, 7)
            plt.plot(t, state_data[:,i,13:16], label=["x","y","z"])
            plt.title("acceleration")
            plt.subplot(3, 3, 8)
            plt.plot(t[:-1], (state_data[1:,i,13:16]-state_data[:-1,i,13:16])/(t[2]-t[1]), label=["ax","ay","az"])
            plt.title("jerk")
            plt.tight_layout()
            # plt.show()
            # plt.legend()
        # fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
            # axes.plot(t, col_dis)
            # axes.set_xlabel("t/s")
            # axes.set_ylabel("closest distance/m")
        # plt.show()
        # print("rewards_sum: ", np_rewards)

        return figs

