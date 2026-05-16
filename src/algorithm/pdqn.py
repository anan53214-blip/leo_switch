from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PDQNConfig:
    num_agents: int = 20
    obs_dim: int = 69
    max_candidates: int = 10
    q_hidden_dims: tuple = (256, 128)
    param_hidden_dims: tuple = (128, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100_000
    grad_clip_norm: float = 1.0
    param_loss_coef: float = 0.1
    bc_loss_coef: float = 0.01
    device: str = "cpu"


class PDQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_discrete_actions: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        input_dim = int(obs_dim) + int(num_discrete_actions) + 1
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        action_one_hot: torch.Tensor,
        continuous_param: torch.Tensor,
    ) -> torch.Tensor:
        if continuous_param.dim() == 1:
            continuous_param = continuous_param.unsqueeze(-1)
        x = torch.cat([obs, action_one_hot, continuous_param], dim=-1)
        return self.net(x).squeeze(-1)


class PDQNParameterNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ReLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs)).squeeze(-1)


class PDQNParameterNets(nn.Module):
    def __init__(self, obs_dim: int, num_discrete_actions: int, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        self.nets = nn.ModuleList(
            [PDQNParameterNet(obs_dim, hidden_dims) for _ in range(int(num_discrete_actions))]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(obs) for net in self.nets], dim=-1)


def _safe_masks(masks: np.ndarray) -> np.ndarray:
    masks_np = np.asarray(masks, dtype=bool).copy()
    empty_rows = ~masks_np.any(axis=1)
    if np.any(empty_rows):
        masks_np[empty_rows, 0] = True
    return masks_np


def _one_hot_action_features(
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


class PDQNAlgorithm:
    def __init__(self, config: PDQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.num_agents = int(config.num_agents)
        self.obs_dim = int(config.obs_dim)
        self.handover_dim = int(config.max_candidates) + 1
        self.action_feature_dim = self.handover_dim + 1
        self.train_step = 0
        self.rng = np.random.default_rng()

        self.q_net = PDQNNetwork(self.obs_dim, self.handover_dim, config.q_hidden_dims).to(self.device)
        self.target_q_net = PDQNNetwork(self.obs_dim, self.handover_dim, config.q_hidden_dims).to(self.device)
        self.param_nets = PDQNParameterNets(self.obs_dim, self.handover_dim, config.param_hidden_dims).to(self.device)
        self.target_param_nets = PDQNParameterNets(
            self.obs_dim,
            self.handover_dim,
            config.param_hidden_dims,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_param_nets.load_state_dict(self.param_nets.state_dict())
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.param_optimizer = torch.optim.Adam(self.param_nets.parameters(), lr=config.lr)

    def current_epsilon(self) -> float:
        progress = min(float(self.train_step) / max(float(self.config.epsilon_decay_steps), 1.0), 1.0)
        return float(self.config.epsilon_start + progress * (self.config.epsilon_final - self.config.epsilon_start))

    def _all_action_q(
        self,
        q_net: PDQNNetwork,
        param_nets: PDQNParameterNets,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = param_nets(obs)
        num_rows = obs.shape[0]
        eye = torch.eye(self.handover_dim, dtype=obs.dtype, device=obs.device)
        repeated_obs = obs.unsqueeze(1).expand(num_rows, self.handover_dim, self.obs_dim).reshape(-1, self.obs_dim)
        action_one_hot = eye.unsqueeze(0).expand(num_rows, self.handover_dim, self.handover_dim).reshape(
            -1,
            self.handover_dim,
        )
        q_values = q_net(repeated_obs, action_one_hot, params.reshape(-1)).reshape(num_rows, self.handover_dim)
        return q_values, params

    def random_actions(self, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks_np = _safe_masks(masks)
        handover_actions = []
        for mask in masks_np:
            valid = np.flatnonzero(mask)
            handover_actions.append(int(self.rng.choice(valid)) if len(valid) else 0)
        handover = np.asarray(handover_actions, dtype=np.int64)
        offload = self.rng.random(len(handover), dtype=np.float32)
        action_features = _one_hot_action_features(handover, offload, self.handover_dim)
        env_actions = np.column_stack([handover, offload]).astype(np.float32)
        return env_actions, action_features, handover

    def act(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        epsilon: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if epsilon is None:
            epsilon = self.current_epsilon()
        masks_np = _safe_masks(masks)
        was_q_training = self.q_net.training
        was_param_training = self.param_nets.training
        self.q_net.eval()
        self.param_nets.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
            q_values, params = self._all_action_q(self.q_net, self.param_nets, obs_tensor)
            q_np = q_values.detach().cpu().numpy()
            params_np = params.detach().cpu().numpy()

        q_np = np.where(masks_np, q_np, -np.inf)
        greedy_handover = np.argmax(q_np, axis=1).astype(np.int64)
        handover = greedy_handover.copy()
        offload = params_np[np.arange(len(handover)), handover].astype(np.float32)

        if float(epsilon) > 0.0:
            explore = self.rng.random(len(handover)) < float(epsilon)
            for agent_id in np.flatnonzero(explore):
                valid = np.flatnonzero(masks_np[agent_id])
                handover[agent_id] = int(self.rng.choice(valid)) if len(valid) else 0
                offload[agent_id] = float(self.rng.random())

        offload = np.clip(offload, 0.0, 1.0).astype(np.float32)
        action_features = _one_hot_action_features(handover, offload, self.handover_dim)
        env_actions = np.column_stack([handover, offload]).astype(np.float32)
        self.q_net.train(was_q_training)
        self.param_nets.train(was_param_training)
        return env_actions, action_features, handover

    def update(self, replay_buffer) -> Dict[str, float]:
        if len(replay_buffer) < max(int(self.config.batch_size), 1):
            return {}

        batch = replay_buffer.sample(self.config.batch_size)
        batch_size, num_agents, _ = batch["obs"].shape
        obs = batch["obs"].reshape(batch_size * num_agents, self.obs_dim)
        next_obs = batch["next_obs"].reshape(batch_size * num_agents, self.obs_dim)
        actions = batch["actions"].reshape(batch_size * num_agents, self.action_feature_dim)
        rewards = batch["rewards"].view(batch_size, 1).expand(batch_size, num_agents).reshape(-1)
        dones = batch["dones"].view(batch_size, 1).expand(batch_size, num_agents).reshape(-1)
        masks = batch["masks"].reshape(batch_size * num_agents, self.handover_dim)
        next_masks = batch["next_masks"].reshape(batch_size * num_agents, self.handover_dim)

        action_indices = torch.argmax(actions[:, : self.handover_dim], dim=-1)
        action_one_hot = F.one_hot(action_indices, num_classes=self.handover_dim).to(obs.dtype)
        offload = actions[:, -1]

        with torch.no_grad():
            next_q_values, _ = self._all_action_q(self.target_q_net, self.target_param_nets, next_obs)
            safe_next_masks = next_masks.bool().clone()
            empty_rows = ~safe_next_masks.any(dim=-1)
            if torch.any(empty_rows):
                safe_next_masks[empty_rows, 0] = True
            masked_next_q = next_q_values.masked_fill(~safe_next_masks, -1e9)
            max_next_q = masked_next_q.max(dim=-1).values
            target = rewards + self.config.gamma * (1.0 - dones) * max_next_q

        q_values = self.q_net(obs, action_one_hot, offload)
        q_loss = F.mse_loss(q_values, target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.q_optimizer.step()

        for param in self.q_net.parameters():
            param.requires_grad_(False)
        all_q, params = self._all_action_q(self.q_net, self.param_nets, obs)
        valid_mask = masks.bool()
        masked_q_sum = (all_q * valid_mask.to(all_q.dtype)).sum()
        valid_count = valid_mask.to(all_q.dtype).sum().clamp_min(1.0)
        q_objective = masked_q_sum / valid_count
        selected_params = params.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
        bc_loss = F.mse_loss(selected_params, offload)
        param_loss = -self.config.param_loss_coef * q_objective + self.config.bc_loss_coef * bc_loss
        self.param_optimizer.zero_grad()
        param_loss.backward()
        nn.utils.clip_grad_norm_(self.param_nets.parameters(), self.config.grad_clip_norm)
        self.param_optimizer.step()
        for param in self.q_net.parameters():
            param.requires_grad_(True)

        self.train_step += 1
        if self.train_step % max(int(self.config.target_update_interval), 1) == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_param_nets.load_state_dict(self.param_nets.state_dict())

        return {
            "q_loss": float(q_loss.detach().cpu().item()),
            "param_loss": float(param_loss.detach().cpu().item()),
            "epsilon": self.current_epsilon(),
        }

    def save(self, path) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "q_net_state_dict": self.q_net.state_dict(),
                "target_q_net_state_dict": self.target_q_net.state_dict(),
                "param_nets_state_dict": self.param_nets.state_dict(),
                "target_param_nets_state_dict": self.target_param_nets.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "param_optimizer_state_dict": self.param_optimizer.state_dict(),
                "train_step": self.train_step,
            },
            Path(path),
        )

    def load(self, path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
        self.param_nets.load_state_dict(checkpoint["param_nets_state_dict"])
        self.target_param_nets.load_state_dict(checkpoint["target_param_nets_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.param_optimizer.load_state_dict(checkpoint["param_optimizer_state_dict"])
        self.train_step = int(checkpoint.get("train_step", 0))
