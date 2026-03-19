"""
经验回放缓冲区
==============

本模块实现PPO/MAPPO所需的Rollout Buffer。

【Rollout Buffer vs Replay Buffer】
┌─────────────────────────────────────────────────────────────┐
│  Rollout Buffer (On-policy)    │  Replay Buffer (Off-policy) │
├─────────────────────────────────────────────────────────────┤
│  - 存储最近一个episode的数据    │  - 存储大量历史数据          │
│  - 每次更新后清空              │  - 随机采样历史数据          │
│  - 用于PPO, A2C等             │  - 用于DQN, SAC等           │
│  - 不需要importance sampling   │  - 需要importance sampling  │
└─────────────────────────────────────────────────────────────┘

【存储内容】
对于每个时间步，存储：
- observations: 观测（用于Actor输入）
- actions: 执行的动作
- rewards: 获得的奖励
- dones: 是否终止
- values: Critic估计的价值
- log_probs: 动作的log概率
- advantages: GAE计算的优势值（后处理）
- returns: 折扣回报（后处理）

【GAE (Generalized Advantage Estimation)】
优势函数估计：A(s,a) = Q(s,a) - V(s)

GAE使用TD(λ)方法平滑估计：
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

λ参数控制偏差-方差权衡：
- λ=0: 高偏差，低方差（1步TD）
- λ=1: 低偏差，高方差（MC）
- λ=0.95: 常用折中
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field


@dataclass
class RolloutBufferSamples:
    """
    采样数据批次
    
    存储从缓冲区采样的数据，用于训练更新。
    """
    observations: torch.Tensor      # (batch, obs_dim)
    actions_discrete: torch.Tensor  # (batch,) 离散动作
    actions_continuous: torch.Tensor  # (batch, 1) 连续动作
    old_log_probs: torch.Tensor     # (batch,) 旧策略log概率
    advantages: torch.Tensor        # (batch,) 优势值
    returns: torch.Tensor           # (batch,) 折扣回报
    values: torch.Tensor            # (batch,) 旧价值估计
    
    # 可选：候选掩码
    candidate_masks: Optional[torch.Tensor] = None


class RolloutBuffer:
    """
    单智能体Rollout Buffer
    
    【使用流程】
    ```python
    buffer = RolloutBuffer(buffer_size=2048, obs_dim=64)
    
    # 收集数据
    for step in range(buffer_size):
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done, value, log_prob)
        obs = next_obs
    
    # 计算优势和回报
    buffer.compute_returns_and_advantages(last_value, done)
    
    # 采样训练
    for batch in buffer.get_batches(batch_size=256):
        loss = compute_loss(batch)
        optimizer.step()
    
    # 清空缓冲区
    buffer.reset()
    ```
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_discrete_dim: int = 1,
        action_continuous_dim: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小（通常等于n_steps * n_envs）
            obs_dim: 观测维度
            action_discrete_dim: 离散动作维度（通常为1）
            action_continuous_dim: 连续动作维度
            gamma: 折扣因子
            gae_lambda: GAE的λ参数
            device: 设备
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.pos = 0  # 当前位置
        self.full = False  # 是否已满
        
        # 预分配存储空间
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions_discrete = np.zeros(buffer_size, dtype=np.int64)
        self.actions_continuous = np.zeros((buffer_size, action_continuous_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        # 后处理计算的值
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
    
    def reset(self):
        """清空缓冲区"""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action_discrete: int,
        action_continuous: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """
        添加一条经验
        
        Args:
            obs: 观测
            action_discrete: 离散动作
            action_continuous: 连续动作
            reward: 奖励
            done: 是否终止
            value: Critic估计的价值
            log_prob: 动作log概率
        """
        self.observations[self.pos] = obs
        self.actions_discrete[self.pos] = action_discrete
        self.actions_continuous[self.pos] = action_continuous
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool
    ):
        """
        计算GAE优势值和折扣回报
        
        【GAE公式】
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        A_t = δ_t + γλ * (1 - done_{t+1}) * A_{t+1}
        
        【Returns计算】
        R_t = A_t + V(s_t)
        
        Args:
            last_value: 最后状态的价值估计
            last_done: 最后状态是否终止
        """
        last_gae = 0
        
        # 反向遍历计算GAE
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            # TD误差
            delta = (
                self.rewards[step] 
                + self.gamma * next_value * next_non_terminal 
                - self.values[step]
            )
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        
        # Returns = Advantages + Values
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]
    
    def get_batches(
        self,
        batch_size: int
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        生成训练批次
        
        使用随机打乱后按批次返回数据。
        
        Args:
            batch_size: 批次大小
            
        Yields:
            RolloutBufferSamples批次
        """
        indices = np.random.permutation(self.pos)
        
        for start in range(0, self.pos, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield RolloutBufferSamples(
                observations=torch.tensor(
                    self.observations[batch_indices], device=self.device
                ),
                actions_discrete=torch.tensor(
                    self.actions_discrete[batch_indices], device=self.device
                ),
                actions_continuous=torch.tensor(
                    self.actions_continuous[batch_indices], device=self.device
                ),
                old_log_probs=torch.tensor(
                    self.log_probs[batch_indices], device=self.device
                ),
                advantages=torch.tensor(
                    self.advantages[batch_indices], device=self.device
                ),
                returns=torch.tensor(
                    self.returns[batch_indices], device=self.device
                ),
                values=torch.tensor(
                    self.values[batch_indices], device=self.device
                )
            )


class MultiAgentRolloutBuffer:
    """
    多智能体Rollout Buffer
    
    【设计思想】
    为MAPPO设计的缓冲区，支持多个智能体的并行存储。
    
    【数据组织】
    - 每个智能体有独立的观测、动作
    - 共享全局状态用于Critic
    - 支持按智能体或按批次采样
    
    【使用方式】
    ```python
    buffer = MultiAgentRolloutBuffer(
        buffer_size=2048,
        num_agents=5,
        obs_dim=64,
        global_state_dim=128
    )
    
    # 收集所有智能体的数据
    buffer.add(
        obs=all_obs,           # (num_agents, obs_dim)
        global_state=state,     # (global_state_dim,)
        actions=all_actions,    # (num_agents,) 或字典
        rewards=all_rewards,    # (num_agents,)
        dones=all_dones,        # (num_agents,)
        values=values,          # (1,) 全局价值
        log_probs=log_probs     # (num_agents,)
    )
    ```
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        global_state_dim: int,
        max_candidates: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        初始化多智能体缓冲区
        
        Args:
            buffer_size: 缓冲区大小（时间步数）
            num_agents: 智能体数量
            obs_dim: 单个智能体观测维度
            global_state_dim: 全局状态维度（用于Critic）
            max_candidates: 最大候选卫星数
            gamma: 折扣因子
            gae_lambda: GAE参数
            device: 设备
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.max_candidates = max_candidates
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.pos = 0
        self.full = False
        
        # ---------- 智能体级别数据 ----------
        # 观测: (buffer_size, num_agents, obs_dim)
        self.observations = np.zeros(
            (buffer_size, num_agents, obs_dim), dtype=np.float32
        )
        
        # 离散动作（切换）: (buffer_size, num_agents)
        self.actions_discrete = np.zeros(
            (buffer_size, num_agents), dtype=np.int64
        )
        
        # 连续动作（卸载比例）: (buffer_size, num_agents)
        self.actions_continuous = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        
        # 候选掩码: (buffer_size, num_agents, max_candidates+1)
        self.candidate_masks = np.ones(
            (buffer_size, num_agents, max_candidates + 1), dtype=np.float32
        )
        
        # 奖励: (buffer_size, num_agents)
        self.rewards = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        
        # 终止标志: (buffer_size,) - 通常所有智能体同时终止
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Log概率: (buffer_size, num_agents)
        self.log_probs = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        
        # ---------- 全局数据 ----------
        # 全局状态（用于Critic）: (buffer_size, global_state_dim)
        self.global_states = np.zeros(
            (buffer_size, global_state_dim), dtype=np.float32
        )

        # 卫星嵌入（用于图式Critic）: (buffer_size, num_satellites, obs_dim)
        # 这里复用 obs_dim 作为嵌入维度，P0 阶段等于 HAN 输出维度
        self.satellite_embeddings = None
        
        # 全局价值: (buffer_size,)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # ---------- 后处理数据 ----------
        # 优势值: (buffer_size, num_agents) - 每个agent独立计算
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
    
    def reset(self):
        """清空缓冲区"""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        satellite_embeddings: Optional[np.ndarray],
        actions_discrete: np.ndarray,
        actions_continuous: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        value: float,
        log_probs: np.ndarray,
        candidate_masks: Optional[np.ndarray] = None
    ):
        """
        添加一个时间步的数据
        
        Args:
            obs: 所有智能体观测, (num_agents, obs_dim)
            global_state: 全局状态, (global_state_dim,)
            actions_discrete: 离散动作, (num_agents,)
            actions_continuous: 连续动作, (num_agents,)
            rewards: 奖励, (num_agents,) 或标量
            done: 是否终止
            value: 全局价值估计
            log_probs: log概率, (num_agents,)
            candidate_masks: 候选掩码, (num_agents, max_candidates+1)
        """
        self.observations[self.pos] = obs
        self.global_states[self.pos] = global_state

        if satellite_embeddings is not None:
            if self.satellite_embeddings is None:
                num_satellites = satellite_embeddings.shape[0]
                embed_dim = satellite_embeddings.shape[1]
                self.satellite_embeddings = np.zeros(
                    (self.buffer_size, num_satellites, embed_dim), dtype=np.float32
                )
            self.satellite_embeddings[self.pos] = satellite_embeddings

        self.actions_discrete[self.pos] = actions_discrete
        self.actions_continuous[self.pos] = actions_continuous
        
        # 奖励可以是每个智能体独立或共享
        if np.isscalar(rewards):
            self.rewards[self.pos] = np.full(self.num_agents, rewards)
        else:
            self.rewards[self.pos] = rewards
        
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_probs
        
        if candidate_masks is not None:
            self.candidate_masks[self.pos] = candidate_masks
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool
    ):
        """
        计算GAE优势值和回报（per-agent）
        
        每个agent使用自己的reward计算独立的advantage，
        共享Critic的value作为baseline。
        
        Args:
            last_value: 最后状态的全局价值
            last_done: 是否终止
        """
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            # 每个agent使用自己的reward计算TD误差
            delta = (
                self.rewards[step]  # (num_agents,)
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos, np.newaxis]
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        生成训练批次
        
        【数据展平】
        将 (buffer_size, num_agents, ...) 展平为 (buffer_size * num_agents, ...)
        这样可以使用参数共享的Actor进行批量更新。
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Yields:
            批次数据字典
        """
        # 展平数据
        total_samples = self.pos * self.num_agents
        
        # 观测: (pos * num_agents, obs_dim)
        flat_obs = self.observations[:self.pos].reshape(-1, self.obs_dim)
        
        # 动作
        flat_actions_discrete = self.actions_discrete[:self.pos].reshape(-1)
        flat_actions_continuous = self.actions_continuous[:self.pos].reshape(-1)
        
        # 候选掩码
        flat_masks = self.candidate_masks[:self.pos].reshape(
            -1, self.max_candidates + 1
        )
        
        # Log概率
        flat_log_probs = self.log_probs[:self.pos].reshape(-1)
        
        # 优势是 per-agent，直接展平用于 actor loss
        flat_advantages = self.advantages[:self.pos].reshape(-1)
        # returns 取 per-timestep 均值用于 critic loss（Critic 输出全局 value）
        flat_returns = np.repeat(self.returns[:self.pos].mean(axis=1), self.num_agents)
        flat_values = np.repeat(self.values[:self.pos], self.num_agents)
        
        # 全局状态也需要复制
        flat_global_states = np.repeat(
            self.global_states[:self.pos], self.num_agents, axis=0
        )

        # 卫星嵌入也需要按智能体复制
        flat_satellite_embeddings = None
        if self.satellite_embeddings is not None:
            flat_satellite_embeddings = np.repeat(
                self.satellite_embeddings[:self.pos], self.num_agents, axis=0
            )
        
        # 打乱索引
        if shuffle:
            indices = np.random.permutation(total_samples)
        else:
            indices = np.arange(total_samples)
        
        # 生成批次
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            yield {
                'observations': torch.tensor(
                    flat_obs[batch_indices], device=self.device, dtype=torch.float32
                ),
                'global_states': torch.tensor(
                    flat_global_states[batch_indices], device=self.device, dtype=torch.float32
                ),
                'satellite_embeddings': None if flat_satellite_embeddings is None else torch.tensor(
                    flat_satellite_embeddings[batch_indices], device=self.device, dtype=torch.float32
                ),
                'actions_discrete': torch.tensor(
                    flat_actions_discrete[batch_indices], device=self.device, dtype=torch.long
                ),
                'actions_continuous': torch.tensor(
                    flat_actions_continuous[batch_indices], device=self.device, dtype=torch.float32
                ),
                'candidate_masks': torch.tensor(
                    flat_masks[batch_indices], device=self.device, dtype=torch.float32
                ),
                'old_log_probs': torch.tensor(
                    flat_log_probs[batch_indices], device=self.device, dtype=torch.float32
                ),
                'advantages': torch.tensor(
                    flat_advantages[batch_indices], device=self.device, dtype=torch.float32
                ),
                'returns': torch.tensor(
                    flat_returns[batch_indices], device=self.device, dtype=torch.float32
                ),
                'values': torch.tensor(
                    flat_values[batch_indices], device=self.device, dtype=torch.float32
                )
            }
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """获取所有数据（不打乱）"""
        return next(self.get_batches(self.pos * self.num_agents, shuffle=False))
