"""
混合动作Actor网络
=================

本模块实现支持混合动作空间的Actor网络，同时输出：
1. 离散动作：切换目标卫星选择
2. 连续动作：任务卸载比例

【混合动作空间挑战】
传统RL要么处理离散动作（DQN），要么处理连续动作（DDPG/SAC）。
本项目需要同时处理两种类型的动作。

【解决方案】
参考 "Parameterized Action Space" 和 "Hybrid Actor-Critic" 的思想：
- 使用共享的特征提取层
- 分离的动作头：
  - 离散头：输出切换目标的概率分布
  - 连续头：输出卸载比例的高斯分布参数

【动作空间设计】
对于用户i：
- 离散动作 h_i ∈ {0, 1, ..., K}
  - 0: 保持当前连接
  - 1~K: 切换到第k个可见卫星
- 连续动作 λ_i ∈ [0, 1]
  - 卸载比例

【网络结构】
```
节点嵌入 (HAN输出)
       │
       ▼
   共享MLP层
       │
   ┌───┴───┐
   ▼       ▼
离散头   连续头
   │       │
   ▼       ▼
π(h|s)  μ(λ|s), σ(λ|s)
```
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Beta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .layers import MLP


@dataclass
class ActorConfig:
    """
    Actor网络配置
    
    【参数选择指南】
    - hidden_dims: 根据问题复杂度选择，一般[256, 128]足够
    - max_candidates: 最大候选卫星数，影响离散动作维度
    - action_std_init: 连续动作的初始标准差，影响探索程度
    """
    # 输入维度
    input_dim: int = 64                # 来自HAN的嵌入维度
    
    # 网络结构
    hidden_dims: List[int] = None      # 隐藏层维度
    
    # 动作空间
    max_candidates: int = 10           # 最大候选卫星数（+1为保持）
    sat_embed_dim: int = 64            # 候选卫星嵌入维度
    
    # 连续动作参数（Beta 分布，天然支持 [0,1]）
    beta_init_scale: float = 1.0       # Beta 分布初始集中度
    min_offload_ratio: float = 0.05    # 卸载模式与纯本地模式的显式边界
    
    # 正则化
    # PPO 的 old/new log-prob 必须来自同一确定性网络；训练态 Dropout
    # 会把随机掩码噪声混入 importance ratio。
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class HybridActor(nn.Module):
    """
    混合动作Actor网络
    
    【功能】
    - 输入：用户节点嵌入 + 候选卫星信息
    - 输出：
      - 离散动作分布（Categorical）
      - 连续动作分布（Beta）
    
    【采样过程】
    1. 从离散分布采样切换决策
    2. 从连续分布采样卸载比例
    3. 计算联合log概率
    
    【训练技巧】
    - 离散动作使用Gumbel-Softmax可以实现端到端训练
    - 连续动作使用重参数化技巧
    """
    
    def __init__(self, config: ActorConfig):
        """
        初始化Actor网络
        
        Args:
            config: Actor配置
        """
        super().__init__()
        
        self.config = config
        
        # ---------- 共享特征提取层 ----------
        # 提取用户状态和候选卫星的联合特征
        self.shared_net = MLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims[:-1],
            output_dim=config.hidden_dims[-1],
            activation='relu',
            dropout=config.dropout
        )
        
        # ---------- 离散动作头（切换决策）----------
        # 输出候选卫星的logits
        # +1 是因为包含"不切换"选项
        self.handover_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1] // 2, config.max_candidates + 1)
        )
        
        # ---------- 动作条件化连续头（卸载比例，Beta 分布）----------
        # 每个 handover 动作拥有独立的 alpha/beta；候选卫星的链路、负载和
        # 表征因此可以直接影响对应目标下的卸载比例。
        self.offload_action_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1] * 2, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], 2),
        )
        # Zero-inflated gate: 0 表示精确纯本地，1 表示进入连续卸载分支。
        # gate 与 Beta 都按 handover 动作条件化。
        self.offload_mode_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1] * 2, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], 1),
        )

        # ---------- 候选卫星嵌入路径（用于增强切换决策）----------
        self.candidate_sat_proj = nn.Linear(config.sat_embed_dim, config.hidden_dims[-1])
        self.candidate_score = nn.Linear(config.hidden_dims[-1] * 2, 1)

    def handover_logits(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算切换 logits，可选地使用候选卫星嵌入增强

        Args:
            user_embedding: 用户嵌入, (batch_size, input_dim)
            candidate_mask: 候选掩码, (batch_size, max_candidates+1)
            candidate_satellite_embeddings: 候选卫星嵌入, (batch_size, max_candidates, embed_dim)

        Returns:
            logits: (batch_size, max_candidates+1)
        """
        features = self.shared_net(user_embedding)
        base_logits = self.handover_head(features)

        if candidate_satellite_embeddings is not None:
            sat_features = self.candidate_sat_proj(candidate_satellite_embeddings)
            repeated_user = features.unsqueeze(1).expand(-1, sat_features.size(1), -1)
            switch_logits = self.candidate_score(
                torch.cat([repeated_user, sat_features], dim=-1)
            ).squeeze(-1)
            logits = torch.cat([base_logits[:, :1], switch_logits], dim=-1)
        else:
            logits = base_logits

        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask.bool(), float("-inf"))

        return logits

    def _offload_distribution(
        self,
        features: torch.Tensor,
        candidate_satellite_embeddings: Optional[torch.Tensor],
    ) -> Beta:
        action_features = self._offload_action_features(
            features,
            candidate_satellite_embeddings,
        )
        alpha_beta = torch.nn.functional.softplus(
            self.offload_action_head(action_features)
        ) + 1.0
        return Beta(alpha_beta[..., 0], alpha_beta[..., 1])

    def _offload_action_features(
        self,
        features: torch.Tensor,
        candidate_satellite_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if candidate_satellite_embeddings is not None:
            candidate_contexts = self.candidate_sat_proj(
                candidate_satellite_embeddings
            )
        else:
            candidate_contexts = features.unsqueeze(1).expand(
                -1,
                self.config.max_candidates,
                -1,
            )
        action_contexts = torch.cat(
            [features.unsqueeze(1), candidate_contexts],
            dim=1,
        )
        repeated_user = features.unsqueeze(1).expand_as(action_contexts)
        return torch.cat([repeated_user, action_contexts], dim=-1)

    def _offload_mode_distribution(
        self,
        features: torch.Tensor,
        candidate_satellite_embeddings: Optional[torch.Tensor],
    ) -> Bernoulli:
        action_features = self._offload_action_features(
            features,
            candidate_satellite_embeddings,
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
        handover_action: torch.Tensor,
    ) -> torch.Tensor:
        return values.gather(
            1,
            handover_action.long().unsqueeze(-1),
        ).squeeze(-1)

    def forward(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, Beta]:
        """
        前向传播，返回动作分布

        Args:
            user_embedding: 用户嵌入, (batch_size, input_dim)
            candidate_mask: 可选的候选卫星掩码, (batch_size, max_candidates+1)
            candidate_satellite_embeddings: 候选卫星嵌入, (batch_size, max_candidates, embed_dim)

        Returns:
            - handover_dist: 切换决策的Categorical分布
            - offload_dist: 卸载比例的Beta分布
        """
        # 1. 共享特征提取
        features = self.shared_net(user_embedding)

        # 2. 离散动作分布
        handover_logits = self.handover_logits(
            user_embedding, candidate_mask, candidate_satellite_embeddings
        )

        handover_dist = Categorical(logits=handover_logits)

        # 3. 每个离散动作对应一个卸载 Beta 分布。
        offload_dist = self._offload_distribution(
            features,
            candidate_satellite_embeddings,
        )

        return handover_dist, offload_dist
    
    def sample(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            user_embedding: 用户嵌入
            candidate_mask: 候选掩码
            deterministic: 是否使用确定性策略（用于评估）
            candidate_satellite_embeddings: 候选卫星嵌入

        Returns:
            - handover_action: 切换动作, (batch_size,)
            - offload_action: 卸载比例, (batch_size, 1)
            - log_prob: 联合log概率, (batch_size,)
        """
        handover_dist, offload_dist = self.forward(
            user_embedding, candidate_mask, candidate_satellite_embeddings
        )
        features = self.shared_net(user_embedding)
        offload_mode_dist = self._offload_mode_distribution(
            features,
            candidate_satellite_embeddings,
        )
        
        if deterministic:
            # 确定性：handover 取最大概率，执行模式取 Bernoulli 众数。
            handover_action = handover_dist.probs.argmax(dim=-1)
            offload_mode = (
                self._select_action_values(
                    offload_mode_dist.probs,
                    handover_action,
                )
                >= 0.5
            ).to(offload_dist.mean.dtype)
            beta_action = self._select_action_values(
                offload_dist.mean,
                handover_action,
            )
            expanded_beta = beta_action.detach().unsqueeze(1).expand_as(offload_dist.mean)
        else:
            # 随机采样显式本地/卸载门控；只有卸载模式才使用 Beta 比例。
            handover_action = handover_dist.sample()
            offload_mode_samples = offload_mode_dist.sample()
            offload_mode = self._select_action_values(
                offload_mode_samples,
                handover_action,
            )
            beta_samples = offload_dist.rsample()
            beta_action = self._select_action_values(beta_samples, handover_action)
            expanded_beta = beta_samples
        
        offload_action = torch.where(
            offload_mode > 0.5,
            self._beta_to_env_ratio(beta_action),
            torch.zeros_like(beta_action),
        )
        
        # 联合概率 = handover × execution-mode × conditional Beta。
        handover_log_prob = handover_dist.log_prob(handover_action)
        mode_log_prob = self._select_action_values(
            offload_mode_dist.log_prob(offload_mode_samples)
            if not deterministic
            else offload_mode_dist.log_prob(
                offload_mode.unsqueeze(1).expand_as(offload_mode_dist.probs)
            ),
            handover_action,
        )
        beta_log_prob = self._select_action_values(
            offload_dist.log_prob(expanded_beta),
            handover_action,
        ) - torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=beta_action.dtype,
                device=beta_action.device,
            )
        )
        offload_log_prob = mode_log_prob + offload_mode * beta_log_prob
        
        if continuous_action_mask is not None:
            offload_log_prob = offload_log_prob * continuous_action_mask.to(
                dtype=offload_log_prob.dtype,
                device=offload_log_prob.device,
            )
        log_prob = handover_log_prob + offload_log_prob
        
        return handover_action, offload_action.unsqueeze(-1), log_prob
    
    def evaluate(
        self,
        user_embedding: torch.Tensor,
        handover_action: torch.Tensor,
        offload_action: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和熵

        用于PPO的策略更新，计算importance sampling ratio。

        Args:
            user_embedding: 用户嵌入
            handover_action: 切换动作
            offload_action: 卸载动作
            candidate_mask: 候选掩码
            candidate_satellite_embeddings: 候选卫星嵌入

        Returns:
            - log_prob: 动作log概率
            - entropy: 策略熵（用于鼓励探索）
        """
        handover_dist, offload_dist = self.forward(
            user_embedding, candidate_mask, candidate_satellite_embeddings
        )
        features = self.shared_net(user_embedding)
        offload_mode_dist = self._offload_mode_distribution(
            features,
            candidate_satellite_embeddings,
        )
        
        handover_log_prob = handover_dist.log_prob(handover_action)
        offload_action = offload_action.squeeze(-1)
        offload_mode = (
            offload_action >= float(self.config.min_offload_ratio)
        ).to(offload_dist.mean.dtype)
        expanded_mode = offload_mode.unsqueeze(1).expand_as(offload_mode_dist.probs)
        mode_log_prob = self._select_action_values(
            offload_mode_dist.log_prob(expanded_mode),
            handover_action,
        )
        beta_action = self._env_ratio_to_beta(offload_action)
        expanded_beta = beta_action.unsqueeze(1).expand_as(offload_dist.mean)
        beta_log_prob = self._select_action_values(
            offload_dist.log_prob(expanded_beta),
            handover_action,
        ) - torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=beta_action.dtype,
                device=beta_action.device,
            )
        )
        offload_log_prob = mode_log_prob + offload_mode * beta_log_prob
        if continuous_action_mask is not None:
            continuous_action_mask = continuous_action_mask.to(
                dtype=offload_log_prob.dtype,
                device=offload_log_prob.device,
            )
            offload_log_prob = offload_log_prob * continuous_action_mask
        log_prob = handover_log_prob + offload_log_prob
        
        handover_entropy = handover_dist.entropy()
        mode_entropy = self._select_action_values(
            offload_mode_dist.entropy(),
            handover_action,
        )
        beta_entropy = self._select_action_values(
            offload_dist.entropy(),
            handover_action,
        ) + torch.log(
            torch.as_tensor(
                max(1.0 - float(self.config.min_offload_ratio), 1e-6),
                dtype=offload_dist.mean.dtype,
                device=offload_dist.mean.device,
            )
        )
        selected_offload_probability = self._select_action_values(
            offload_mode_dist.probs,
            handover_action,
        )
        offload_entropy = mode_entropy + selected_offload_probability * beta_entropy
        if continuous_action_mask is not None:
            offload_entropy = offload_entropy * continuous_action_mask
        entropy = handover_entropy + offload_entropy
        
        return log_prob, entropy
    
class MultiAgentActor(nn.Module):
    """
    多智能体Actor（参数共享版本）
    
    【设计思想】
    在多智能体强化学习中，有两种参数共享策略：
    1. 完全共享：所有智能体使用同一个Actor
    2. 部分共享：共享大部分参数，仅输出层独立
    
    本实现采用完全共享，因为：
    - 用户本质上是同类智能体
    - 通过观测区分不同用户
    - 减少参数量，提高样本效率
    
    【与单智能体的区别】
    - 输入增加智能体ID/索引信息
    - 批处理所有智能体的决策
    - 支持并行采样
    """
    
    def __init__(
        self,
        config: ActorConfig,
        num_agents: int
    ):
        """
        初始化多智能体Actor
        
        Args:
            config: Actor配置
            num_agents: 智能体数量（用户数量）
        """
        super().__init__()
        
        self.config = config
        self.num_agents = num_agents
        
        # 共享的Actor网络
        self.actor = HybridActor(config)
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        candidate_masks: Optional[torch.Tensor] = None
    ) -> Tuple[Categorical, Beta]:
        """
        前向传播
        
        Args:
            user_embeddings: 所有用户的嵌入, (num_agents, input_dim)
            agent_ids: 智能体ID, (num_agents,)
            candidate_masks: 候选掩码, (num_agents, max_candidates+1)
            
        Returns:
            动作分布
        """
        del agent_ids

        return self.actor(user_embeddings, candidate_masks)
    
    def sample_all(
        self,
        user_embeddings: torch.Tensor,
        candidate_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        为所有智能体采样动作

        Args:
            user_embeddings: 所有用户嵌入, (num_agents, input_dim)
            candidate_masks: 候选掩码
            deterministic: 是否确定性策略
            candidate_satellite_embeddings: 候选卫星嵌入

        Returns:
            动作字典:
            - 'handover': (num_agents,) 切换动作
            - 'offload': (num_agents, 1) 卸载比例
            - 'log_prob': (num_agents,) log概率
        """
        handover, offload, log_prob = self.actor.sample(
            user_embeddings,
            candidate_masks,
            deterministic,
            candidate_satellite_embeddings,
            continuous_action_mask,
        )

        return {
            'handover': handover,
            'offload': offload.squeeze(-1),
            'log_prob': log_prob
        }
    
    def evaluate_all(
        self,
        user_embeddings: torch.Tensor,
        handover_actions: torch.Tensor,
        offload_actions: torch.Tensor,
        candidate_masks: Optional[torch.Tensor] = None,
        candidate_satellite_embeddings: Optional[torch.Tensor] = None,
        continuous_action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估所有智能体的动作

        Args:
            user_embeddings: 所有用户嵌入
            handover_actions: 所有切换动作
            offload_actions: 所有卸载动作
            candidate_masks: 候选掩码
            candidate_satellite_embeddings: 候选卫星嵌入

        Returns:
            - log_probs: (num_agents,)
            - entropies: (num_agents,)
        """
        return self.actor.evaluate(
            user_embeddings,
            handover_actions,
            offload_actions.unsqueeze(-1) if offload_actions.dim() == 1 else offload_actions,
            candidate_masks,
            candidate_satellite_embeddings,
            continuous_action_mask,
        )
