import torch as th
import torch.nn as nn

import torch.nn.functional as F


def regulate(model, timesteps):

    env = model.env
    policy = model.policy

    obs = env.reset()
    for _step in range(timesteps):
        try:
            # inter_action = policy.inter_forward(obs.detach())
            inter_action, _ = policy.actor.action_and_entropy(obs.detach(),deterministic=True)
            pre_target = obs["state"][:, :3].clone().detach()
            obs["state"] = th.cat([inter_action.to(obs["state"].device), obs["state"][:,3:]],dim=-1)
            action, _ = policy.frozen_part.action_and_entropy(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            loss = F.mse_loss(pre_target.cuda(), inter_action)
            model.policy.actor.optimizer.zero_grad()
            loss.backward()
            model.policy.actor.optimizer.step()
            print("loss:",loss.item(), "  step:", _step)
        except KeyboardInterrupt:
            pass
    model.save()
    print("Regulation finished and model saved.")
