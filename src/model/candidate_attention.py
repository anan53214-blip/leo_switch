"""Load-aware attention policy modules for candidate satellite selection."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta, Categorical


VISIBLE_SAT_OFFSET = 3 + 1 + 5
VISIBLE_SAT_FEATURE_DIM = 6


@dataclass
class CandidateAttentionConfig:
    user_obs_dim: int
    sat_feature_dim: int
    hidden_dim: int = 64
    num_heads: int = 4
    max_candidates: int = 10
    dropout: float = 0.0
    risk_feature_start: int = 8
    risk_feature_dim: int = 0
    min_offload_ratio: float = 0.05


class SatelliteLoadEncoder(nn.Module):
    """Self-attention encoder over global satellite load/status tokens."""

    def __init__(
        self,
        sat_feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(sat_feature_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, satellite_features: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if satellite_features.dim() == 2:
            satellite_features = satellite_features.unsqueeze(0)
            squeeze_batch = True
        if satellite_features.dim() != 3:
            raise ValueError(
                "satellite_features must have shape (num_sats, feat_dim) "
                "or (batch, num_sats, feat_dim)"
            )

        tokens = self.input_proj(satellite_features)
        attended, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attended)
        tokens = self.ffn_norm(tokens + self.ffn(tokens))
        return tokens.squeeze(0) if squeeze_batch else tokens


class CandidateAttentionActor(nn.Module):
    """
    Shared actor using satellite-load self-attention and user-satellite cross attention.

    The user observation keeps the environment's raw layout. The handover-candidate
    block starts at index 9 and stores ``max_candidates`` rows of six link features
    in the same order as ``candidate_sat_ids`` and handover actions 1..K.
    """

    def __init__(self, config: CandidateAttentionConfig):
        super().__init__()
        self.config = config

        self.sat_encoder = SatelliteLoadEncoder(
            sat_feature_dim=config.sat_feature_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.user_proj = nn.Sequential(
            nn.Linear(config.user_obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.link_proj = nn.Sequential(
            nn.Linear(VISIBLE_SAT_FEATURE_DIM, config.hidden_dim),
            nn.ReLU(),
        )
        self.candidate_fuse = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        if config.risk_feature_dim > 0:
            self.risk_proj = nn.Sequential(
                nn.Linear(config.risk_feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_dim),
            )
        else:
            self.risk_proj = None
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.hidden_dim)
        self.keep_head = nn.Linear(config.hidden_dim, 1)
        self.candidate_score = nn.Linear(config.hidden_dim, 1)
        self.offload_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.offload_mode_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _candidate_link_features(self, observations: torch.Tensor) -> torch.Tensor:
        end = VISIBLE_SAT_OFFSET + self.config.max_candidates * VISIBLE_SAT_FEATURE_DIM
        if observations.size(-1) < end:
            raise ValueError(
                "observations are too short to contain the visible-satellite "
                f"block ending at index {end}"
            )
        return observations[:, VISIBLE_SAT_OFFSET:end].view(
            observations.size(0),
            self.config.max_candidates,
            VISIBLE_SAT_FEATURE_DIM,
        )

    def _gather_candidate_tokens(
        self,
        encoded_sats: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
    ) -> torch.Tensor:
        ids = candidate_sat_ids.to(device=encoded_sats.device, dtype=torch.long)
        safe_ids = ids.clamp(min=0)

        if encoded_sats.dim() == 2:
            gathered = encoded_sats[safe_ids]
        elif encoded_sats.dim() == 3:
            if encoded_sats.size(0) != ids.size(0):
                raise ValueError(
                    "batched satellite_features must share the same batch "
                    "dimension as candidate_sat_ids"
                )
            batch_index = torch.arange(
                encoded_sats.size(0), device=encoded_sats.device
            ).unsqueeze(1)
            gathered = encoded_sats[batch_index, safe_ids]
        else:
            raise ValueError("encoded_sats must have rank 2 or 3")

        return gathered.masked_fill((ids < 0).unsqueeze(-1), 0.0)

    def _policy_features(
        self,
        observations: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
        satellite_features: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_context = self.user_proj(observations)
        encoded_sats = self.sat_encoder(satellite_features)
        candidate_global = self._gather_candidate_tokens(encoded_sats, candidate_sat_ids)
        candidate_link = self.link_proj(self._candidate_link_features(observations))
        candidate_tokens = self.candidate_fuse(
            torch.cat([candidate_global, candidate_link], dim=-1)
        )
        risk_end = self.config.risk_feature_start + self.config.risk_feature_dim
        if self.risk_proj is not None and satellite_features.size(-1) >= risk_end:
            gathered_raw = self._gather_candidate_tokens(
                satellite_features,
                candidate_sat_ids,
            )
            risk_features = gathered_raw[
                ...,
                self.config.risk_feature_start:risk_end,
            ]
            candidate_tokens = candidate_tokens + self.risk_proj(risk_features)

        attention_padding_mask = None
        if candidate_masks is not None:
            valid_candidates = candidate_masks[:, 1:].to(candidate_tokens.device) > 0
            candidate_tokens = candidate_tokens * valid_candidates.unsqueeze(-1)
            attention_padding_mask = ~valid_candidates
            # MultiheadAttention 在整行都被 mask 时会产生 NaN；保留一个零
            # sentinel，仅用于形成稳定的零候选上下文。
            no_valid_candidate = ~valid_candidates.any(dim=1)
            if no_valid_candidate.any():
                attention_padding_mask = attention_padding_mask.clone()
                attention_padding_mask[no_valid_candidate, 0] = False

        attended, _ = self.cross_attn(
            user_context.unsqueeze(1),
            candidate_tokens,
            candidate_tokens,
            key_padding_mask=attention_padding_mask,
            need_weights=False,
        )
        attended = self.cross_norm(user_context + attended.squeeze(1))
        return attended, candidate_tokens

    def handover_logits(
        self,
        observations: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
        satellite_features: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_context, candidate_tokens = self._policy_features(
            observations,
            candidate_masks,
            satellite_features,
            candidate_sat_ids,
        )
        return self._handover_logits_from_features(
            user_context,
            candidate_tokens,
            candidate_masks,
        )

    def _handover_logits_from_features(
        self,
        user_context: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        keep_logit = self.keep_head(user_context)
        candidate_logits = self.candidate_score(
            torch.tanh(candidate_tokens + user_context.unsqueeze(1))
        ).squeeze(-1)
        logits = torch.cat([keep_logit, candidate_logits], dim=1)

        if candidate_masks is not None:
            mask = candidate_masks.to(device=logits.device, dtype=torch.bool).clone()
            no_valid_action = ~mask.any(dim=1)
            if no_valid_action.any():
                mask[no_valid_action, 0] = True
            logits = logits.masked_fill(~mask, -1e9)
        return logits

    def _offload_distribution_from_features(
        self,
        user_context: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> Beta:
        action_features = self._offload_action_features(
            user_context,
            candidate_tokens,
        )
        alpha_beta = F.softplus(self.offload_head(action_features)) + 1.0
        return Beta(alpha_beta[..., 0], alpha_beta[..., 1])

    @staticmethod
    def _offload_action_features(
        user_context: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        keep_context = user_context.unsqueeze(1)
        action_contexts = torch.cat([keep_context, candidate_tokens], dim=1)
        expanded_user = user_context.unsqueeze(1).expand_as(action_contexts)
        return torch.cat([expanded_user, action_contexts], dim=-1)

    def _offload_mode_distribution_from_features(
        self,
        user_context: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> Bernoulli:
        action_features = self._offload_action_features(
            user_context,
            candidate_tokens,
        )
        return Bernoulli(logits=self.offload_mode_head(action_features).squeeze(-1))

    def _beta_to_env_ratio(self, beta_value: torch.Tensor) -> torch.Tensor:
        minimum = float(self.config.min_offload_ratio)
        return minimum + (1.0 - minimum) * beta_value

    def _env_ratio_to_beta(self, offload_ratio: torch.Tensor) -> torch.Tensor:
        minimum = float(self.config.min_offload_ratio)
        return ((offload_ratio - minimum) / max(1.0 - minimum, 1e-6)).clamp(
            1e-6,
            1.0 - 1e-6,
        )

    @staticmethod
    def _select_action_values(
        values: torch.Tensor,
        handover: torch.Tensor,
    ) -> torch.Tensor:
        return values.gather(1, handover.long().unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        observations: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
        satellite_features: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
    ) -> Tuple[Categorical, Beta]:
        user_context, candidate_tokens = self._policy_features(
            observations,
            candidate_masks,
            satellite_features,
            candidate_sat_ids,
        )
        logits = self._handover_logits_from_features(
            user_context,
            candidate_tokens,
            candidate_masks,
        )
        handover_dist = Categorical(logits=logits)
        offload_dist = self._offload_distribution_from_features(
            user_context,
            candidate_tokens,
        )
        return handover_dist, offload_dist

    def sample_all(
        self,
        observations: torch.Tensor,
        candidate_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        satellite_features: Optional[torch.Tensor] = None,
        candidate_sat_ids: Optional[torch.Tensor] = None,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if satellite_features is None or candidate_sat_ids is None:
            raise ValueError("satellite_features and candidate_sat_ids are required")

        user_context, candidate_tokens = self._policy_features(
            observations, candidate_masks, satellite_features, candidate_sat_ids
        )
        handover_dist = Categorical(
            logits=self._handover_logits_from_features(
                user_context,
                candidate_tokens,
                candidate_masks,
            )
        )
        offload_dist = self._offload_distribution_from_features(
            user_context,
            candidate_tokens,
        )
        offload_mode_dist = self._offload_mode_distribution_from_features(
            user_context,
            candidate_tokens,
        )
        if deterministic:
            handover = handover_dist.probs.argmax(dim=-1)
            offload_mode = (
                self._select_action_values(offload_mode_dist.probs, handover) >= 0.5
            ).to(offload_dist.mean.dtype)
            beta_action = self._select_action_values(offload_dist.mean, handover)
            expanded_beta = beta_action.unsqueeze(1).expand_as(offload_dist.mean)
            expanded_mode = offload_mode.unsqueeze(1).expand_as(offload_mode_dist.probs)
            selected_mode_log_prob = self._select_action_values(
                offload_mode_dist.log_prob(expanded_mode),
                handover,
            )
        else:
            handover = handover_dist.sample()
            offload_mode_samples = offload_mode_dist.sample()
            offload_mode = self._select_action_values(offload_mode_samples, handover)
            beta_samples = offload_dist.sample()
            beta_action = self._select_action_values(beta_samples, handover)
            expanded_beta = beta_samples
            selected_mode_log_prob = self._select_action_values(
                offload_mode_dist.log_prob(offload_mode_samples),
                handover,
            )
        offload = torch.where(
            offload_mode > 0.5,
            self._beta_to_env_ratio(beta_action),
            torch.zeros_like(beta_action),
        )
        selected_beta_log_prob = self._select_action_values(
            offload_dist.log_prob(expanded_beta),
            handover,
        ) - torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=beta_action.dtype,
                device=beta_action.device,
            )
        )
        selected_offload_log_prob = (
            selected_mode_log_prob + offload_mode * selected_beta_log_prob
        )
        if continuous_action_mask is not None:
            selected_offload_log_prob = (
                selected_offload_log_prob
                * continuous_action_mask.to(selected_offload_log_prob.dtype)
            )
        log_prob = handover_dist.log_prob(handover) + selected_offload_log_prob
        return {
            "handover": handover,
            "offload": offload,
            "log_prob": log_prob,
        }

    def evaluate_all(
        self,
        observations: torch.Tensor,
        actions_discrete: torch.Tensor,
        actions_continuous: torch.Tensor,
        candidate_masks: Optional[torch.Tensor],
        satellite_features: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_context, candidate_tokens = self._policy_features(
            observations, candidate_masks, satellite_features, candidate_sat_ids
        )
        handover_dist = Categorical(
            logits=self._handover_logits_from_features(
                user_context,
                candidate_tokens,
                candidate_masks,
            )
        )
        offload_dist = self._offload_distribution_from_features(
            user_context,
            candidate_tokens,
        )
        offload_mode_dist = self._offload_mode_distribution_from_features(
            user_context,
            candidate_tokens,
        )
        offload = actions_continuous.squeeze(-1)
        offload_mode = (
            offload >= float(self.config.min_offload_ratio)
        ).to(offload_dist.mean.dtype)
        expanded_mode = offload_mode.unsqueeze(1).expand_as(offload_mode_dist.probs)
        selected_mode_log_prob = self._select_action_values(
            offload_mode_dist.log_prob(expanded_mode),
            actions_discrete,
        )
        beta_action = self._env_ratio_to_beta(offload)
        expanded_beta = beta_action.unsqueeze(1).expand_as(offload_dist.mean)
        selected_beta_log_prob = self._select_action_values(
            offload_dist.log_prob(expanded_beta),
            actions_discrete,
        ) - torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=beta_action.dtype,
                device=beta_action.device,
            )
        )
        selected_offload_log_prob = (
            selected_mode_log_prob + offload_mode * selected_beta_log_prob
        )
        selected_mode_entropy = self._select_action_values(
            offload_mode_dist.entropy(),
            actions_discrete,
        )
        selected_beta_entropy = self._select_action_values(
            offload_dist.entropy(),
            actions_discrete,
        ) + torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=offload_dist.mean.dtype,
                device=offload_dist.mean.device,
            )
        )
        selected_offload_probability = self._select_action_values(
            offload_mode_dist.probs,
            actions_discrete,
        )
        selected_offload_entropy = (
            selected_mode_entropy
            + selected_offload_probability * selected_beta_entropy
        )
        if continuous_action_mask is not None:
            continuous_action_mask = continuous_action_mask.to(
                selected_offload_log_prob.dtype
            )
            selected_offload_log_prob = (
                selected_offload_log_prob * continuous_action_mask
            )
            selected_offload_entropy = (
                selected_offload_entropy * continuous_action_mask
            )
        log_prob = (
            handover_dist.log_prob(actions_discrete.long())
            + selected_offload_log_prob
        )
        entropy = handover_dist.entropy() + selected_offload_entropy
        return log_prob, entropy
