"""MAPPO variant backed by load-aware candidate attention."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, "src")

from model.candidate_attention import CandidateAttentionActor, CandidateAttentionConfig
from model.critic import CentralizedCritic, CriticConfig
from .buffer import MultiAgentRolloutBuffer
from .mappo import MAPPO, MAPPOConfig


class AttentionMAPPO(MAPPO):
    """MAPPO with satellite-load self-attention and candidate cross-attention."""

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        hidden_dim = config.sat_embed_dim
        if config.actor_hidden_dims:
            hidden_dim = max(config.sat_embed_dim, int(config.actor_hidden_dims[-1] // 2))

        actor_config = CandidateAttentionConfig(
            user_obs_dim=config.obs_dim,
            sat_feature_dim=config.sat_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            max_candidates=config.max_candidates,
            dropout=0.1,
        )
        self.actor = CandidateAttentionActor(actor_config).to(self.device)

        critic_config = CriticConfig(
            input_dim=config.obs_dim,
            sat_input_dim=config.sat_embed_dim,
            num_agents=config.num_agents,
            hidden_dims=config.critic_hidden_dims,
        )
        self.critic = CentralizedCritic(critic_config).to(self.device)

        actor_lr = config.actor_lr or config.learning_rate
        critic_lr = config.critic_lr or config.learning_rate
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )
        self.buffer: Optional[MultiAgentRolloutBuffer] = None
        self.train_step = 0
        self._last_train_stats: Dict[str, float] = {}

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        candidate_masks: Optional[np.ndarray] = None,
        satellite_embeddings: Optional[np.ndarray] = None,
        candidate_sat_ids: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
        observations = np.asarray(observations, dtype=np.float32)
        expected_obs_shape = (self.config.num_agents, self.config.obs_dim)
        if observations.shape != expected_obs_shape:
            raise ValueError(
                "observations must have shape "
                f"{expected_obs_shape}, got {tuple(observations.shape)}"
            )
        if candidate_masks is None:
            candidate_masks = np.ones(
                (self.config.num_agents, self.config.max_candidates + 1),
                dtype=np.float32,
            )
        candidate_masks = np.asarray(candidate_masks, dtype=np.float32)
        expected_mask_shape = (
            self.config.num_agents,
            self.config.max_candidates + 1,
        )
        if candidate_masks.shape != expected_mask_shape:
            raise ValueError(
                "candidate_masks must have shape "
                f"{expected_mask_shape}, got {tuple(candidate_masks.shape)}"
            )
        if satellite_embeddings is None or candidate_sat_ids is None:
            raise ValueError(
                "AttentionMAPPO requires satellite_embeddings and candidate_sat_ids"
            )
        satellite_embeddings = np.asarray(satellite_embeddings, dtype=np.float32)
        if satellite_embeddings.ndim != 2:
            raise ValueError(
                "satellite_embeddings must be a rank-2 array, got "
                f"{satellite_embeddings.ndim} dimensions"
            )
        if satellite_embeddings.shape[1] != self.config.sat_embed_dim:
            raise ValueError(
                "satellite_embeddings second dimension must match "
                f"sat_embed_dim={self.config.sat_embed_dim}, got "
                f"{satellite_embeddings.shape[1]}"
            )
        candidate_sat_ids = np.asarray(candidate_sat_ids, dtype=np.int64)
        expected_ids_shape = (self.config.num_agents, self.config.max_candidates)
        if candidate_sat_ids.shape != expected_ids_shape:
            raise ValueError(
                "candidate_sat_ids must have shape "
                f"{expected_ids_shape}, got {tuple(candidate_sat_ids.shape)}"
            )

        obs_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.tensor(
            candidate_masks, dtype=torch.float32, device=self.device
        )
        sat_tensor = torch.tensor(
            satellite_embeddings, dtype=torch.float32, device=self.device
        )
        ids_tensor = torch.tensor(
            candidate_sat_ids, dtype=torch.long, device=self.device
        )

        actions = self.actor.sample_all(
            obs_tensor,
            mask_tensor,
            deterministic=deterministic,
            satellite_features=sat_tensor,
            candidate_sat_ids=ids_tensor,
        )
        value = self.critic(obs_tensor, sat_tensor)
        return {
            "handover": actions["handover"].cpu().numpy(),
            "offload": actions["offload"].cpu().numpy(),
        }, actions["log_prob"].cpu().numpy(), value.item()

    @torch.no_grad()
    def get_value(
        self,
        observations: np.ndarray,
        satellite_embeddings: Optional[np.ndarray] = None,
    ) -> float:
        if satellite_embeddings is None:
            raise ValueError("AttentionMAPPO requires satellite_embeddings")
        obs_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )
        sat_tensor = torch.tensor(
            satellite_embeddings, dtype=torch.float32, device=self.device
        )
        return self.critic(obs_tensor, sat_tensor).item()

    def update(self) -> Dict[str, float]:
        if self.buffer is None or self.buffer.pos == 0:
            return {}

        self.actor.train()
        self.critic.train()

        all_actor_losses = []
        all_critic_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        all_clip_fractions = []

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs = batch["observations"]
                actions_discrete = batch["actions_discrete"]
                actions_continuous = batch["actions_continuous"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]
                candidate_masks = batch["candidate_masks"]
                satellite_embeddings = batch["satellite_embeddings"]
                candidate_sat_ids = batch["candidate_sat_ids"]

                if satellite_embeddings is None:
                    raise ValueError(
                        "AttentionMAPPO update requires satellite_embeddings"
                    )

                if self.config.normalize_advantage:
                    advantages = (
                        (advantages - advantages.mean())
                        / (advantages.std() + 1e-8)
                    )

                new_log_probs, entropy = self.actor.evaluate_all(
                    obs,
                    actions_discrete,
                    actions_continuous,
                    candidate_masks,
                    satellite_embeddings,
                    candidate_sat_ids,
                )

                log_ratio = torch.clamp(new_log_probs - old_log_probs, -20.0, 2.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                global_states = batch["global_states"]
                gs_reshaped = global_states.view(
                    -1,
                    self.config.num_agents,
                    self.config.obs_dim,
                )
                new_values = self.critic(
                    gs_reshaped,
                    satellite_embeddings,
                ).squeeze(-1)

                if self.config.normalize_returns:
                    returns_mean = returns.mean()
                    returns_std = returns.std() + 1e-8
                    returns_for_loss = (returns - returns_mean) / returns_std
                    new_values_for_loss = (new_values - returns_mean) / returns_std
                    old_values_for_loss = (old_values - returns_mean) / returns_std
                else:
                    returns_for_loss = returns
                    new_values_for_loss = new_values
                    old_values_for_loss = old_values

                def _value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    if self.config.value_loss_type == "huber":
                        return F.smooth_l1_loss(
                            pred,
                            target,
                            beta=self.config.value_huber_beta,
                        )
                    return F.mse_loss(pred, target)

                if self.config.clip_range_vf is not None:
                    values_clipped = old_values_for_loss + torch.clamp(
                        new_values_for_loss - old_values_for_loss,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss1 = _value_loss(new_values_for_loss, returns_for_loss)
                    value_loss2 = _value_loss(values_clipped, returns_for_loss)
                    critic_loss = torch.max(value_loss1, value_loss2)
                else:
                    critic_loss = _value_loss(new_values_for_loss, returns_for_loss)

                loss = (
                    actor_loss
                    + self.config.value_loss_coef * critic_loss
                    + self._current_entropy_coef() * entropy_loss
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                with torch.no_grad():
                    raw_log_ratio = new_log_probs - old_log_probs
                    approx_kl = (
                        (torch.exp(raw_log_ratio) - 1) - raw_log_ratio
                    ).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_range)
                        .float()
                        .mean()
                        .item()
                    )

                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                all_entropy_losses.append(-entropy_loss.item())
                all_kl_divs.append(approx_kl)
                all_clip_fractions.append(clip_fraction)

            if self.config.target_kl is not None and all_kl_divs:
                if all_kl_divs[-1] > 1.5 * self.config.target_kl:
                    break

        self.train_step += 1
        stats = {
            "actor_loss": float(np.mean(all_actor_losses)),
            "critic_loss": float(np.mean(all_critic_losses)),
            "entropy": float(np.mean(all_entropy_losses)),
            "kl_divergence": float(np.mean(all_kl_divs)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "train_step": self.train_step,
        }
        self._last_train_stats = stats
        return stats
