from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MADDPGConfig:
    num_agents: int = 20
    obs_dim: int = 69
    max_candidates: int = 10
    sat_embed_dim: int = 64
    actor_hidden_dims: tuple = (256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.01
    noise_start: float = 0.35
    noise_final: float = 0.05
    noise_decay_steps: int = 100_000
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    grad_clip_norm: float = 1.0
    device: str = "cpu"


class MADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, handover_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.handover_head = nn.Linear(in_dim, int(handover_dim))
        self.offload_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        handover_logits = self.handover_head(features)
        offload = torch.sigmoid(self.offload_head(features)).squeeze(-1)
        return handover_logits, offload


class HANCentralizedCritic(nn.Module):
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_feature_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
    ):
        super().__init__()
        input_dim = int(num_agents) * (int(obs_dim) + int(action_feature_dim))
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        action_features: torch.Tensor,
        sat_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sat_embeddings
        if observations.dim() != 3 or action_features.dim() != 3:
            raise ValueError("MADDPG critic expects batched tensors shaped (batch, agents, dim).")
        joint_input = torch.cat(
            [
                observations.reshape(observations.shape[0], -1),
                action_features.reshape(action_features.shape[0], -1),
            ],
            dim=-1,
        )
        return self.net(joint_input).squeeze(-1)


def safe_mask_tensor(masks: torch.Tensor) -> torch.Tensor:
    safe_masks = masks.bool().clone()
    flat_masks = safe_masks.reshape(-1, safe_masks.shape[-1])
    empty_rows = ~flat_masks.any(dim=-1)
    if torch.any(empty_rows):
        flat_masks[empty_rows, 0] = True
    return safe_masks


def maddpg_actor_action_features(
    actor: MADDPGActor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    straight_through: bool = True,
) -> torch.Tensor:
    batch_size, num_agents, obs_dim = observations.shape
    handover_dim = masks.shape[-1]
    flat_obs = observations.reshape(batch_size * num_agents, obs_dim)
    flat_masks = safe_mask_tensor(masks).reshape(batch_size * num_agents, handover_dim)
    logits, offload = actor(flat_obs)
    probs = torch.softmax(logits.masked_fill(~flat_masks, -1e9), dim=-1)
    if straight_through:
        hard_indices = torch.argmax(probs, dim=-1)
        hard_handover = F.one_hot(hard_indices, num_classes=handover_dim).to(probs.dtype)
        handover_features = hard_handover + probs - probs.detach()
    else:
        handover_features = probs
    features = torch.cat([handover_features, offload.unsqueeze(-1)], dim=-1)
    return features.reshape(batch_size, num_agents, handover_dim + 1)


def maddpg_one_hot_action_features(
    handover_actions: np.ndarray,
    offload_ratios: np.ndarray,
    handover_dim: int,
) -> np.ndarray:
    handover = np.clip(np.asarray(handover_actions, dtype=np.int64), 0, handover_dim - 1)
    offload = np.clip(np.asarray(offload_ratios, dtype=np.float32), 0.0, 1.0)
    features = np.zeros((len(handover), handover_dim + 1), dtype=np.float32)
    features[np.arange(len(handover)), handover] = 1.0
    features[:, -1] = offload
    return features


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


class MADDPGAlgorithm:
    def __init__(self, config: MADDPGConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.num_agents = int(config.num_agents)
        self.obs_dim = int(config.obs_dim)
        self.handover_dim = int(config.max_candidates) + 1
        self.action_feature_dim = self.handover_dim + 1
        self.train_step = 0
        self.rng = np.random.default_rng()

        self.actor = MADDPGActor(self.obs_dim, self.handover_dim, config.actor_hidden_dims).to(self.device)
        self.target_actor = MADDPGActor(self.obs_dim, self.handover_dim, config.actor_hidden_dims).to(self.device)
        self.critic = HANCentralizedCritic(
            self.num_agents,
            self.obs_dim,
            self.action_feature_dim,
            config.critic_hidden_dims,
        ).to(self.device)
        self.target_critic = HANCentralizedCritic(
            self.num_agents,
            self.obs_dim,
            self.action_feature_dim,
            config.critic_hidden_dims,
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def _noise_std(self) -> float:
        progress = min(float(self.train_step) / max(float(self.config.noise_decay_steps), 1.0), 1.0)
        return float(self.config.noise_start + progress * (self.config.noise_final - self.config.noise_start))

    def random_actions(self, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        handover_actions = []
        for mask in np.asarray(masks, dtype=bool):
            valid = np.flatnonzero(mask)
            handover_actions.append(int(self.rng.choice(valid)) if len(valid) else 0)
        handover = np.asarray(handover_actions, dtype=np.int64)
        offload = self.rng.random(len(handover), dtype=np.float32)
        action_features = maddpg_one_hot_action_features(handover, offload, self.handover_dim)
        env_actions = np.column_stack([handover, offload]).astype(np.float32)
        return env_actions, action_features, handover

    def act(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
            logits, offload = self.actor(obs_tensor)
            logits_np = logits.detach().cpu().numpy()
            offload_np = offload.detach().cpu().numpy()

        masks_np = np.asarray(masks, dtype=bool).copy()
        empty_rows = ~masks_np.any(axis=1)
        if np.any(empty_rows):
            masks_np[empty_rows, 0] = True

        if not deterministic:
            noise_std = self._noise_std()
            logits_np = logits_np + self.rng.normal(0.0, noise_std, size=logits_np.shape)
            offload_np = offload_np + self.rng.normal(0.0, noise_std, size=offload_np.shape)

        logits_np = np.where(masks_np, logits_np, -np.inf)
        handover = np.argmax(logits_np, axis=1).astype(np.int64)
        offload_np = np.clip(offload_np, 0.0, 1.0).astype(np.float32)
        action_features = maddpg_one_hot_action_features(handover, offload_np, self.handover_dim)
        env_actions = np.column_stack([handover, offload_np]).astype(np.float32)
        self.actor.train(was_training)
        return env_actions, action_features, handover

    def update(self, replay_buffer) -> Dict[str, float]:
        if len(replay_buffer) < max(int(self.config.batch_size), 1):
            return {}

        batch = replay_buffer.sample(self.config.batch_size)
        obs_b = batch["obs"]
        action_b = batch["actions"]
        reward_b = batch["rewards"]
        next_obs_b = batch["next_obs"]
        done_b = batch["dones"]
        mask_b = batch["masks"]
        next_mask_b = batch["next_masks"]
        if reward_b.dim() == 2:
            reward_b = reward_b.mean(dim=1)

        with torch.no_grad():
            next_action_b = maddpg_actor_action_features(self.target_actor, next_obs_b, next_mask_b)
            target_q = self.target_critic(next_obs_b, next_action_b)
            target = reward_b + self.config.gamma * (1.0 - done_b) * target_q

        q_values = self.critic(obs_b, action_b)
        critic_loss = F.mse_loss(q_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad_(False)
        actor_action_b = maddpg_actor_action_features(self.actor, obs_b, mask_b)
        actor_loss = -self.critic(obs_b, actor_action_b).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        soft_update(self.actor, self.target_actor, self.config.tau)
        soft_update(self.critic, self.target_critic, self.config.tau)
        self.train_step += 1
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
        }

    def save(self, path) -> None:
        checkpoint = {
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "train_step": self.train_step,
            "python_random_state": random.getstate(),
        }
        torch.save(checkpoint, Path(path))

    def load(self, path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.train_step = int(checkpoint.get("train_step", 0))
