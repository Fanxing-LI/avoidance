import torch.nn as nn
import torch as th
from algorithms.BPTT_series.policy import Policy, ContinuousCritic
import zipfile
import os, sys
from algorithms.BPTT_series.BPTT import BPTT
from typing import Optional, Type, Union, Dict, Any, Tuple
import copy


class StagePolicy(Policy):
    def __init__(self, observation_space,
                 action_space,
                 lr_schedule,
                 inter_action_space,
                 features_extractor_class: Optional[Union[nn.Module]] = "EmptyExtractor",
                 features_extractor_kwargs: Optional[Dict[str, Any]] = {},
                 activation_fn: Type[nn.Module] = "relu",
                 actor: dict = {},
                 critic: dict = {},
                 share_features_extractor: bool = False,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 children_policy_cfg=None,
                 ):
        # children_cfgs = kwargs.get("children_cfgs")
        inter_action_space = inter_action_space
        super(StagePolicy, self).__init__(
            observation_space,
            inter_action_space,
            lr_schedule,
            features_extractor_class,
            features_extractor_kwargs,
            activation_fn,
            actor,
            critic,
            share_features_extractor,
            optimizer_class,
            optimizer_kwargs,
            )
        self.frozen_part = None

        activation_fn = self.act_alias[activation_fn]
        critic.update(
            {
                "optimizer_class": optimizer_class,
                "optimizer_kwargs": optimizer_kwargs,
            }
        )
        self.critic = ContinuousCritic(
            action_space=action_space,
            features_extractor=self.features_extractor if not share_features_extractor else copy.deepcopy(self.features_extractor),
            activation_fn=activation_fn,
            **critic
        )

        self.critic_target = copy.deepcopy(self.critic)

        if children_policy_cfg:
            self.load_children(children_policy_cfg)

    def load_children(self, path):
        self.frozen_part = BPTT.load_policy(path)
        self.frozen_part.eval()
        self.frozen_part.requires_grad_(False)

    def forward(self, obs):
        inter_action = self.inter_forward(obs)
        obs["state"] = th.cat(inter_action, obs["state"][:,3:])
        action = self.frozen_part(obs)
        return action

    def inter_forward(self, obs):
        return self.actor.forward(obs)

    def inter_predict(self, obs, deterministic=True):
        with th.no_grad():
            return self.actor.predict(obs, deterministic=deterministic)

    def action_and_entropy(self, obs, deterministic=False):
        action, ent = self.actor.action_and_entropy(obs, deterministic=deterministic)
        obs["state"] = th.cat([action, obs["state"][:, 3:].to(action.device)], dim=-1)
        action, _ = self.frozen_part.action_and_entropy(obs, deterministic=deterministic)
        return action, ent

    def predict(self, obs: Dict[str, th.Tensor], deterministic=False, sample=False) -> Tuple[th.Tensor, th.Tensor]:
        inter_action = self.inter_predict(obs, deterministic=deterministic)
        obs["state"] = th.cat([inter_action, obs["state"][:,3:].to(inter_action.device)],dim=-1)
        with th.no_grad():
            return self.frozen_part.predict(obs, deterministic=deterministic, sample=sample)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.frozen_part is not None:
            self.frozen_part.to(*args, **kwargs)

        return self

