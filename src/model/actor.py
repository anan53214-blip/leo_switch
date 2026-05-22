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
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Beta
from typing import Dict, List, Optional, Tuple, Any
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
    
    # 连续动作参数（Beta 分布，天然支持 [0,1]）
    beta_init_scale: float = 1.0       # Beta 分布初始集中度
    
    # 正则化
    dropout: float = 0.1
    
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
        
        # ---------- 连续动作头（卸载比例，Beta 分布）----------
        # 输出 Beta 分布的 alpha 和 beta 参数
        self.offload_alpha_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1] // 2, 1),
            nn.Softplus()  # 确保 > 0
        )
        self.offload_beta_head = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1] // 2, 1),
            nn.Softplus()  # 确保 > 0
        )
    
    def forward(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Categorical, Beta]:
        """
        前向传播，返回动作分布
        
        Args:
            user_embedding: 用户嵌入, (batch_size, input_dim)
            candidate_mask: 可选的候选卫星掩码, (batch_size, max_candidates+1)
                           1表示有效候选，0表示无效（会被mask掉）
                           
        Returns:
            - handover_dist: 切换决策的Categorical分布
            - offload_dist: 卸载比例的Beta分布
        """
        # 1. 共享特征提取
        features = self.shared_net(user_embedding)
        
        # 2. 离散动作分布
        handover_logits = self.handover_head(features)
        
        # 应用候选掩码（无效候选的logit设为-inf）
        if candidate_mask is not None:
            if candidate_mask.shape != handover_logits.shape:
                raise ValueError(
                    "candidate_mask shape must match handover logits: "
                    f"got {tuple(candidate_mask.shape)} vs "
                    f"{tuple(handover_logits.shape)}"
                )
            if not torch.any(candidate_mask > 0, dim=-1).all():
                raise ValueError(
                    "candidate_mask must keep at least one valid action per "
                    "agent"
                )
            handover_logits = handover_logits.masked_fill(
                ~candidate_mask.bool(), float('-inf')
            )
        
        handover_dist = Categorical(logits=handover_logits)
        
        # 3. 连续动作分布（Beta 分布，天然支持 [0,1]）
        alpha = self.offload_alpha_head(features) + 1.0  # alpha >= 1
        beta = self.offload_beta_head(features) + 1.0    # beta >= 1
        offload_dist = Beta(alpha, beta)
        
        return handover_dist, offload_dist
    
    def sample(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            user_embedding: 用户嵌入
            candidate_mask: 候选掩码
            deterministic: 是否使用确定性策略（用于评估）
            
        Returns:
            - handover_action: 切换动作, (batch_size,)
            - offload_action: 卸载比例, (batch_size, 1)
            - log_prob: 联合log概率, (batch_size,)
        """
        handover_dist, offload_dist = self.forward(user_embedding, candidate_mask)
        
        if deterministic:
            # 确定性：选择最高概率/均值
            handover_action = handover_dist.probs.argmax(dim=-1)
            offload_action = offload_dist.mean
        else:
            # 随机采样
            handover_action = handover_dist.sample()
            offload_action = offload_dist.rsample()  # Beta 分布天然 [0,1]
        
        # Beta 分布采样值已在 (0,1)，clamp 仅做数值安全
        offload_action = torch.clamp(offload_action, 1e-6, 1.0 - 1e-6)
        
        # 计算log概率（使用 clamp 后的 action，与 evaluate 一致）
        handover_log_prob = handover_dist.log_prob(handover_action)
        offload_log_prob = offload_dist.log_prob(offload_action.detach()).sum(dim=-1)
        
        # 联合log概率（假设独立）
        log_prob = handover_log_prob + offload_log_prob
        
        return handover_action, offload_action, log_prob
    
    def evaluate(
        self,
        user_embedding: torch.Tensor,
        handover_action: torch.Tensor,
        offload_action: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和熵
        
        用于PPO的策略更新，计算importance sampling ratio。
        
        Args:
            user_embedding: 用户嵌入
            handover_action: 切换动作
            offload_action: 卸载动作
            candidate_mask: 候选掩码
            
        Returns:
            - log_prob: 动作log概率
            - entropy: 策略熵（用于鼓励探索）
        """
        handover_dist, offload_dist = self.forward(user_embedding, candidate_mask)
        
        # Log概率
        handover_log_prob = handover_dist.log_prob(handover_action)
        # 确保 offload_action 维度匹配 Beta 分布的 event_shape (batch, 1)
        if offload_action.dim() == 1:
            offload_action = offload_action.unsqueeze(-1)
        offload_action = torch.clamp(offload_action, 1e-6, 1.0 - 1e-6)
        offload_log_prob = offload_dist.log_prob(offload_action).sum(dim=-1)
        log_prob = handover_log_prob + offload_log_prob
        
        # 熵
        handover_entropy = handover_dist.entropy()
        offload_entropy = offload_dist.entropy().sum(dim=-1)
        entropy = handover_entropy + offload_entropy
        
        return log_prob, entropy
    
    def get_action_probs(
        self,
        user_embedding: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取切换动作的概率分布（用于可视化）
        
        Returns:
            概率分布, (batch_size, max_candidates+1)
        """
        handover_dist, _ = self.forward(user_embedding, candidate_mask)
        return handover_dist.probs


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
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        为所有智能体采样动作
        
        Args:
            user_embeddings: 所有用户嵌入, (num_agents, input_dim)
            candidate_masks: 候选掩码
            deterministic: 是否确定性策略
            
        Returns:
            动作字典:
            - 'handover': (num_agents,) 切换动作
            - 'offload': (num_agents, 1) 卸载比例
            - 'log_prob': (num_agents,) log概率
        """
        handover, offload, log_prob = self.actor.sample(
            user_embeddings, candidate_masks, deterministic
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
        candidate_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估所有智能体的动作
        
        Args:
            user_embeddings: 所有用户嵌入
            handover_actions: 所有切换动作
            offload_actions: 所有卸载动作
            candidate_masks: 候选掩码
            
        Returns:
            - log_probs: (num_agents,)
            - entropies: (num_agents,)
        """
        return self.actor.evaluate(
            user_embeddings,
            handover_actions,
            offload_actions.unsqueeze(-1) if offload_actions.dim() == 1 else offload_actions,
            candidate_masks
        )
