"""
MAPPO算法实现
=============

Multi-Agent Proximal Policy Optimization (MAPPO)

【PPO核心思想】
PPO通过限制策略更新步长来保证训练稳定性：
- 使用clip机制限制importance sampling ratio
- 避免策略剧烈变化导致的性能崩溃

【MAPPO特点】
1. 参数共享：所有智能体共享Actor参数
2. 集中式Critic：使用全局状态估计价值
3. 同步更新：所有智能体同时更新

【损失函数】

1. Actor损失（PPO-Clip）:
   L^CLIP = E[ min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A) ]
   其中 r(θ) = π(a|s) / π_old(a|s)

2. Critic损失（Value Loss）:
   L^VF = E[ (V(s) - R)² ]
   
3. 熵奖励（鼓励探索）:
   L^ENT = -E[ H(π(·|s)) ]

4. 总损失:
   L = L^CLIP - c1·L^VF + c2·L^ENT

【超参数推荐】(来自MAPPO论文)
- clip_range: 0.2
- value_loss_coef: 0.5
- entropy_coef: 0.01
- max_grad_norm: 0.5
- n_epochs: 10-15
- batch_size: 32-512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, 'src')

from model.actor import HybridActor, MultiAgentActor, ActorConfig
from model.critic import CentralizedCritic, CriticConfig, create_global_state
from .buffer import MultiAgentRolloutBuffer


@dataclass
class MAPPOConfig:
    """
    MAPPO配置参数
    
    【参数分类】
    1. 网络参数：Actor/Critic结构
    2. PPO参数：clip_range, epochs等
    3. 优化参数：learning_rate, grad_norm等
    4. GAE参数：gamma, gae_lambda
    """
    # ---------- 环境参数 ----------
    num_agents: int = 5                 # 智能体数量（用户数）
    obs_dim: int = 64                   # 观测维度
    global_state_dim: int = 128         # 全局状态维度
    max_candidates: int = 10            # 最大候选卫星数
    sat_embed_dim: int = 64             # 卫星嵌入维度（HAN输出维度）
    risk_feature_start: int = 8         # CandidateAttention risk feature offset
    risk_feature_dim: int = 0           # CandidateAttention risk feature width
    
    # ---------- 网络参数 ----------
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    
    # ---------- PPO参数 ----------
    clip_range: float = 0.2             # PPO clip范围
    clip_range_vf: Optional[float] = None  # 价值函数clip（None则不clip）
    target_kl: Optional[float] = 0.02   # 目标KL散度（早停），防止策略偏移过大
    value_loss_type: str = 'huber'      # 价值损失: mse / huber
    normalize_returns: bool = True      # 价值损失前是否标准化returns
    value_huber_beta: float = 10.0      # Huber beta（仅huber生效）
    
    # ---------- 损失系数 ----------
    value_loss_coef: float = 0.5        # 价值损失系数 c1
    entropy_coef: float = 0.005          # Keep exploration from collapsing too early
    entropy_schedule: str = "constant"   # Entropy schedule: constant / linear
    entropy_decay_steps: int = 300       # Decay steps for linear schedule
    entropy_min_coef_ratio: float = 0.1  # Minimum ratio for linear schedule
    
    # ---------- 优化参数 ----------
    learning_rate: float = 3e-4         # 学习率（v4: 从5e-5提升至3e-4，增大策略更新步长）
    actor_lr: Optional[float] = None    # Actor单独学习率（None则用lr）
    critic_lr: Optional[float] = None   # Critic单独学习率
    max_grad_norm: float = 0.5          # 梯度裁剪
    
    # ---------- 训练参数 ----------
    n_epochs: int = 10                  # 每次更新的epoch数（v4: 从4增至10，充分利用数据）
    batch_size: int = 64                # 批次大小
    normalize_advantage: bool = True    # 是否标准化优势
    
    # ---------- GAE参数 ----------
    gamma: float = 0.99                 # 折扣因子
    gae_lambda: float = 0.95            # GAE λ参数
    
    # ---------- 设备 ----------
    device: str = 'cpu'


class MAPPO:
    """
    MAPPO算法
    
    【使用示例】
    ```python
    config = MAPPOConfig(
        num_agents=5,
        obs_dim=64,
        global_state_dim=128
    )
    mappo = MAPPO(config)
    
    # 训练循环
    obs = env.reset()
    for step in range(total_steps):
        # 采样动作
        actions, log_probs, value = mappo.act(obs, global_state)
        
        # 环境交互
        next_obs, rewards, done, info = env.step(actions)
        
        # 存储数据
        mappo.buffer.add(obs, global_state, actions, rewards, done, value, log_probs)
        
        # 更新
        if step % update_interval == 0:
            mappo.update()
    ```
    """
    
    def __init__(self, config: MAPPOConfig):
        """
        初始化MAPPO
        
        Args:
            config: MAPPO配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # ---------- 创建Actor ----------
        actor_config = ActorConfig(
            input_dim=config.obs_dim,
            hidden_dims=config.actor_hidden_dims,
            max_candidates=config.max_candidates,
            sat_embed_dim=config.sat_embed_dim,
        )
        self.actor = MultiAgentActor(actor_config, config.num_agents).to(self.device)
        
        # ---------- 创建Critic ----------
        critic_config = CriticConfig(
            input_dim=config.obs_dim,  # 用户嵌入维度（拼接后）
            sat_input_dim=config.sat_embed_dim if hasattr(config, 'sat_embed_dim') else config.obs_dim,
            num_agents=config.num_agents,
            hidden_dims=config.critic_hidden_dims
        )
        self.critic = CentralizedCritic(critic_config).to(self.device)
        
        # ---------- 优化器 ----------
        actor_lr = config.actor_lr or config.learning_rate
        critic_lr = config.critic_lr or config.learning_rate
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )
        
        # ---------- 经验缓冲区 ----------
        # 注意：缓冲区在Runner中创建，这里只保存引用
        self.buffer = None
        
        # ---------- 训练统计 ----------
        self.train_step = 0
        self._last_train_stats = {}
    
    def _current_entropy_coef(self) -> float:
        """Return the entropy coefficient for the current update."""
        if self.config.entropy_schedule == "linear":
            decay_steps = max(int(self.config.entropy_decay_steps), 1)
            min_ratio = float(np.clip(self.config.entropy_min_coef_ratio, 0.0, 1.0))
            scale = max(1.0 - self.train_step / decay_steps, min_ratio)
            return float(self.config.entropy_coef * scale)
        return float(self.config.entropy_coef)
    
    def set_buffer(self, buffer: MultiAgentRolloutBuffer):
        """设置经验缓冲区"""
        self.buffer = buffer
    
    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        candidate_masks: Optional[np.ndarray] = None,
        satellite_embeddings: Optional[np.ndarray] = None,
        candidate_sat_ids: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
        """
        为所有智能体采样动作

        Args:
            observations: 观测, (num_agents, obs_dim)
            candidate_masks: 候选掩码, (num_agents, max_candidates+1)
            satellite_embeddings: 卫星嵌入, (num_sats, sat_embed_dim)
            candidate_sat_ids: 候选卫星ID, (num_agents, max_candidates)
            deterministic: 是否使用确定性策略

        Returns:
            - actions: {'handover': (num_agents,), 'offload': (num_agents,)}
            - log_probs: (num_agents,)
            - value: 全局状态价值（标量）
        """
        # observations 在 P0 阶段表示用户嵌入
        observations = np.asarray(observations, dtype=np.float32)
        expected_obs_shape = (self.config.num_agents, self.config.obs_dim)
        if observations.shape != expected_obs_shape:
            raise ValueError(
                "observations must have shape "
                f"{expected_obs_shape}, got {tuple(observations.shape)}"
            )

        if candidate_masks is not None:
            candidate_masks = np.asarray(candidate_masks, dtype=np.float32)
            expected_mask_shape = (
                self.config.num_agents,
                self.config.max_candidates + 1,
            )
            if candidate_masks.shape != expected_mask_shape:
                raise ValueError(
                    "candidate_masks must have shape "
                    f"{expected_mask_shape}, got "
                    f"{tuple(candidate_masks.shape)}"
                )

        if satellite_embeddings is not None:
            satellite_embeddings = np.asarray(
                satellite_embeddings,
                dtype=np.float32,
            )
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

        obs_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )

        sat_tensor = None
        if satellite_embeddings is not None:
            sat_tensor = torch.tensor(
                satellite_embeddings, dtype=torch.float32, device=self.device
            )
        
        mask_tensor = None
        if candidate_masks is not None:
            mask_tensor = torch.tensor(
                candidate_masks, dtype=torch.float32, device=self.device
            )

        candidate_satellite_embeddings = self._gather_candidate_satellite_embeddings(
            sat_tensor,
            candidate_sat_ids,
        )

        # Actor采样
        actions = self.actor.sample_all(
            obs_tensor, mask_tensor, deterministic=deterministic,
            candidate_satellite_embeddings=candidate_satellite_embeddings
        )
        
        # Critic评估
        value = self.critic(obs_tensor, sat_tensor)
        
        # 转换回numpy
        return {
            'handover': actions['handover'].cpu().numpy(),
            'offload': actions['offload'].cpu().numpy()
        }, actions['log_prob'].cpu().numpy(), value.item()
    
    def _gather_batch_candidate_satellite_embeddings(
        self,
        satellite_embeddings: torch.Tensor,
        candidate_sat_ids: torch.Tensor,
    ) -> torch.Tensor:
        ids = candidate_sat_ids.to(device=self.device, dtype=torch.long)
        safe_ids = ids.clamp(min=0)
        batch_index = torch.arange(
            satellite_embeddings.size(0),
            device=self.device,
        ).unsqueeze(1)
        gathered = satellite_embeddings[batch_index, safe_ids]
        return gathered.masked_fill((ids < 0).unsqueeze(-1), 0.0)

    def _gather_candidate_satellite_embeddings(
        self,
        satellite_embeddings: Optional[torch.Tensor],
        candidate_sat_ids: Optional[np.ndarray | torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if satellite_embeddings is None or candidate_sat_ids is None:
            return None

        ids = torch.as_tensor(candidate_sat_ids, dtype=torch.long, device=self.device)
        if ids.shape != (self.config.num_agents, self.config.max_candidates):
            raise ValueError(
                "candidate_sat_ids must have shape "
                f"{(self.config.num_agents, self.config.max_candidates)}, got {tuple(ids.shape)}"
            )

        safe_ids = ids.clamp(min=0)
        gathered = satellite_embeddings[safe_ids]
        gathered = gathered.masked_fill((ids < 0).unsqueeze(-1), 0.0)
        return gathered

    @torch.no_grad()
    def get_value(
        self,
        observations: np.ndarray,
        satellite_embeddings: Optional[np.ndarray] = None
    ) -> float:
        """
        获取状态价值（用于计算GAE）
        
        Args:
            observations: (num_agents, obs_dim)
            
        Returns:
            全局状态价值
        """
        obs_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )

        sat_tensor = None
        if satellite_embeddings is not None:
            sat_tensor = torch.tensor(
                satellite_embeddings, dtype=torch.float32, device=self.device
            )

        value = self.critic(obs_tensor, sat_tensor)
        return value.item()
    
    def update(self) -> Dict[str, float]:
        """
        执行PPO更新
        
        【更新流程】
        1. 从缓冲区获取数据
        2. 标准化优势
        3. 多个epoch更新：
           a. 计算新策略的log概率和熵
           b. 计算importance sampling ratio
           c. 计算PPO-clip损失
           d. 计算价值损失
           e. 反向传播更新
        
        Returns:
            训练统计信息
        """
        if self.buffer is None or self.buffer.pos == 0:
            return {}
        
        self.actor.train()
        self.critic.train()
        
        # 统计信息
        all_actor_losses = []
        all_critic_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        all_clip_fractions = []
        
        for epoch in range(self.config.n_epochs):
            # 获取批次数据
            for batch in self.buffer.get_batches(self.config.batch_size):
                # 解包数据
                obs = batch['observations']
                actions_discrete = batch['actions_discrete']
                actions_continuous = batch['actions_continuous']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values = batch['values']
                candidate_masks = batch['candidate_masks']
                satellite_embeddings = batch['satellite_embeddings']
                
                # ---------- 标准化优势 ----------
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ---------- 计算新策略的log概率和熵 ----------
                candidate_satellite_embeddings = None
                if satellite_embeddings is not None and 'candidate_sat_ids' in batch:
                    candidate_satellite_embeddings = self._gather_batch_candidate_satellite_embeddings(
                        satellite_embeddings,
                        batch['candidate_sat_ids'],
                    )
                new_log_probs, entropy = self.actor.actor.evaluate(
                    obs,
                    actions_discrete,
                    actions_continuous,
                    candidate_masks,
                    candidate_satellite_embeddings
                )
                
                # ---------- PPO-Clip损失 ----------
                # Importance sampling ratio（clamp log_ratio 防止数值爆炸）
                log_ratio = new_log_probs - old_log_probs
                log_ratio = torch.clamp(log_ratio, -20.0, 2.0)
                ratio = torch.exp(log_ratio)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range
                ) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 熵损失（负号因为要最大化熵）
                entropy_loss = -entropy.mean()
                
                # ---------- 价值损失 ----------
                # 使用全局状态计算价值（与act()时一致）
                # global_states 包含所有 agent 信息，reshape 为 (batch, num_agents, obs_dim)
                global_states = batch['global_states']
                num_agents = self.config.num_agents
                obs_dim = self.config.obs_dim
                gs_reshaped = global_states.view(-1, num_agents, obs_dim)
                new_values = self.critic(gs_reshaped, satellite_embeddings).squeeze(-1)  # (batch,)

                # 可选：标准化returns，降低critic loss量级和训练抖动
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
                    if self.config.value_loss_type == 'huber':
                        return F.smooth_l1_loss(pred, target, beta=self.config.value_huber_beta)
                    return F.mse_loss(pred, target)
                
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = old_values_for_loss + torch.clamp(
                        new_values_for_loss - old_values_for_loss,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss1 = _value_loss(new_values_for_loss, returns_for_loss)
                    value_loss2 = _value_loss(values_clipped, returns_for_loss)
                    critic_loss = torch.max(value_loss1, value_loss2)
                else:
                    critic_loss = _value_loss(new_values_for_loss, returns_for_loss)
                
                # ---------- Total loss ----------
                entropy_coef = self._current_entropy_coef()
                loss = (
                    actor_loss
                    + self.config.value_loss_coef * critic_loss
                    + entropy_coef * entropy_loss
                )
                
                # ---------- 反向传播 ----------
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # ---------- 记录统计 ----------
                with torch.no_grad():
                    # KL散度近似
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    
                    # Clip fraction
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_range)
                        .float().mean().item()
                    )
                
                all_actor_losses.append(actor_loss.item())
                all_critic_losses.append(critic_loss.item())
                all_entropy_losses.append(-entropy_loss.item())  # 转为正熵
                all_kl_divs.append(approx_kl)
                all_clip_fractions.append(clip_fraction)
            
            # 早停检查：当前 epoch 的平均 KL 超过阈值则停止
            if self.config.target_kl is not None and all_kl_divs:
                if all_kl_divs[-1] > 1.5 * self.config.target_kl:
                    break
        
        self.train_step += 1
        
        # 汇总统计
        stats = {
            'actor_loss': np.mean(all_actor_losses),
            'critic_loss': np.mean(all_critic_losses),
            'entropy': np.mean(all_entropy_losses),
            'kl_divergence': np.mean(all_kl_divs),
            'clip_fraction': np.mean(all_clip_fractions),
            'train_step': self.train_step
        }
        
        self._last_train_stats = stats
        return stats
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_step': self.train_step,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.train_step = checkpoint['train_step']
    
    def get_stats(self) -> Dict[str, float]:
        """获取最近的训练统计"""
        return self._last_train_stats
