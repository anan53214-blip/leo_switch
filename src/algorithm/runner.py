"""
训练运行器
==========

本模块实现训练的主循环，负责：
1. 环境交互
2. 数据收集
3. 调用算法更新
4. 日志记录
5. 模型保存

【训练流程】
```
┌─────────────────────────────────────────────────────────────┐
│                      训练主循环                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for iteration in range(max_iterations):                    │
│      ┌─────────────────────────────────────────────────┐   │
│      │           1. 数据收集阶段                        │   │
│      │  for step in range(n_steps):                    │   │
│      │      action = actor.act(obs)                    │   │
│      │      next_obs, reward, done = env.step(action)  │   │
│      │      buffer.add(obs, action, reward, ...)       │   │
│      └─────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│      ┌─────────────────────────────────────────────────┐   │
│      │           2. 计算优势和回报                      │   │
│      │  buffer.compute_returns_and_advantages()        │   │
│      └─────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│      ┌─────────────────────────────────────────────────┐   │
│      │           3. 策略更新                           │   │
│      │  stats = mappo.update()                         │   │
│      └─────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│      ┌─────────────────────────────────────────────────┐   │
│      │           4. 日志和保存                          │   │
│      │  logger.log(stats)                              │   │
│      │  if save_interval: save_model()                 │   │
│      └─────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque

import torch

import sys
sys.path.insert(0, 'src')

from .mappo import MAPPO, MAPPOConfig
from .buffer import MultiAgentRolloutBuffer


@dataclass
class RunnerConfig:
    """
    训练运行器配置
    
    【关键参数】
    - n_steps: 每次更新收集的步数
    - total_timesteps: 总训练步数
    - eval_interval: 评估间隔
    """
    # ---------- 训练参数 ----------
    total_timesteps: int = 1_000_000    # 总训练步数
    n_steps: int = 2048                 # 每次更新的步数
    
    # ---------- 评估参数 ----------
    eval_interval: int = 10_000         # 评估间隔（步数）
    eval_episodes: int = 5              # 评估episode数
    
    # ---------- 日志参数 ----------
    log_interval: int = 1               # 日志间隔（更新次数）
    verbose: int = 1                    # 详细程度 (0: 无, 1: 基本, 2: 详细)
    
    # ---------- 保存参数 ----------
    save_interval: int = 50_000         # 保存间隔（步数）
    save_path: str = 'results/models'   # 保存路径
    
    # ---------- 其他 ----------
    seed: int = 42                      # 随机种子


class Runner:
    """
    训练运行器
    
    【功能】
    - 管理训练循环
    - 环境交互和数据收集
    - 调用MAPPO更新
    - 统计和日志
    - 模型保存
    
    【使用示例】
    ```python
    # 创建环境
    env = LEOSatelliteEnv(env_config)
    
    # 创建运行器
    runner_config = RunnerConfig(total_timesteps=100000)
    mappo_config = MAPPOConfig(num_agents=5, obs_dim=64)
    
    runner = Runner(env, mappo_config, runner_config)
    
    # 开始训练
    runner.train()
    
    # 评估
    runner.evaluate(n_episodes=10)
    ```
    """
    
    def __init__(
        self,
        env,
        mappo_config: MAPPOConfig,
        runner_config: RunnerConfig = None,
        graph_builder = None,
        han_encoder = None
    ):
        """
        初始化运行器
        
        Args:
            env: Gymnasium环境
            mappo_config: MAPPO配置
            runner_config: 运行器配置
            graph_builder: 图构建器（可选，用于HAN）
            han_encoder: HAN编码器（可选）
        """
        self.env = env
        self.config = runner_config or RunnerConfig()
        
        # 设置随机种子
        self._set_seed(self.config.seed)
        
        # ---------- 创建MAPPO ----------
        self.mappo = MAPPO(mappo_config)
        
        # ---------- 创建缓冲区 ----------
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=self.config.n_steps,
            num_agents=mappo_config.num_agents,
            obs_dim=mappo_config.obs_dim,
            global_state_dim=mappo_config.global_state_dim,
            max_candidates=mappo_config.max_candidates,
            gamma=mappo_config.gamma,
            gae_lambda=mappo_config.gae_lambda,
            device=mappo_config.device
        )
        self.mappo.set_buffer(self.buffer)
        
        # ---------- 可选组件 ----------
        self.graph_builder = graph_builder
        self.han_encoder = han_encoder
        
        # ---------- 训练统计 ----------
        self.total_steps = 0
        self.episodes = 0
        self.episode_rewards = deque(maxlen=100)  # 最近100个episode的奖励
        self.episode_lengths = deque(maxlen=100)
        
        # 当前episode统计
        self._episode_reward = 0
        self._episode_length = 0
        
        # 训练开始时间
        self._start_time = None
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取观测
        
        【简化版】直接使用环境的observation
        【完整版】使用图构建器+HAN获取嵌入
        
        Returns:
            - observations: (num_agents, obs_dim)
            - global_state: (global_state_dim,)
        """
        # 简化版：直接使用环境观测
        obs = self.env.get_wrapper_attr('_get_obs') if hasattr(self.env, 'get_wrapper_attr') else None
        
        if obs is None:
            # 从环境状态构建观测
            num_agents = self.mappo.config.num_agents
            obs_dim = self.mappo.config.obs_dim
            
            # 简单地使用随机观测（实际应从环境获取）
            observations = np.random.randn(num_agents, obs_dim).astype(np.float32)
            global_state = np.random.randn(self.mappo.config.global_state_dim).astype(np.float32)
        else:
            observations = obs
            global_state = observations.mean(axis=0)  # 简单聚合
        
        return observations, global_state
    
    def _get_candidate_masks(self) -> np.ndarray:
        """
        获取候选卫星掩码
        
        Returns:
            (num_agents, max_candidates+1) 的掩码
        """
        num_agents = self.mappo.config.num_agents
        max_candidates = self.mappo.config.max_candidates
        
        # 简化：所有候选都有效
        # 实际应根据可见性计算
        masks = np.ones((num_agents, max_candidates + 1), dtype=np.float32)
        
        return masks
    
    def train(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            callback: 每次更新后的回调函数
            
        Returns:
            训练统计
        """
        self._start_time = time.time()
        
        # 重置环境
        obs_raw, info = self.env.reset()
        observations, global_state = self._process_obs(obs_raw)
        
        # 训练统计
        all_stats = []
        num_updates = 0
        
        if self.config.verbose >= 1:
            print("=" * 60)
            print("开始MAPPO训练")
            print(f"  总步数: {self.config.total_timesteps:,}")
            print(f"  每次更新步数: {self.config.n_steps}")
            print(f"  智能体数量: {self.mappo.config.num_agents}")
            print("=" * 60)
        
        while self.total_steps < self.config.total_timesteps:
            # ---------- 数据收集阶段 ----------
            self.buffer.reset()
            
            for step in range(self.config.n_steps):
                # 获取候选掩码
                candidate_masks = self._get_candidate_masks()
                
                # 采样动作
                actions, log_probs, value = self.mappo.act(
                    observations, candidate_masks
                )
                
                # 执行动作
                # 将动作转换为环境期望的格式
                env_actions = self._convert_actions(actions)
                next_obs_raw, rewards, terminated, truncated, info = self.env.step(env_actions)
                done = terminated or truncated
                
                # 处理观测
                next_observations, next_global_state = self._process_obs(next_obs_raw)
                
                # 计算奖励（可能需要转换格式）
                if np.isscalar(rewards):
                    agent_rewards = np.full(self.mappo.config.num_agents, rewards)
                else:
                    agent_rewards = rewards
                
                # 存储经验
                self.buffer.add(
                    obs=observations,
                    global_state=global_state,
                    satellite_embeddings=None,
                    actions_discrete=actions['handover'],
                    actions_continuous=actions['offload'],
                    rewards=agent_rewards,
                    done=done,
                    value=value,
                    log_probs=log_probs,
                    candidate_masks=candidate_masks
                )
                
                # 更新统计
                self._episode_reward += float(np.mean(agent_rewards))
                self._episode_length += 1
                self.total_steps += 1
                
                # 处理episode结束
                if done:
                    self.episodes += 1
                    self.episode_rewards.append(self._episode_reward)
                    self.episode_lengths.append(self._episode_length)
                    
                    self._episode_reward = 0
                    self._episode_length = 0
                    
                    # 重置环境
                    next_obs_raw, info = self.env.reset()
                    next_observations, next_global_state = self._process_obs(next_obs_raw)
                
                observations = next_observations
                global_state = next_global_state
            
            # ---------- 计算优势和回报 ----------
            last_value = self.mappo.get_value(observations)
            self.buffer.compute_returns_and_advantages(last_value, done)
            
            # ---------- 策略更新 ----------
            train_stats = self.mappo.update()
            num_updates += 1
            all_stats.append(train_stats)
            
            # ---------- 日志记录 ----------
            if self.config.verbose >= 1 and num_updates % self.config.log_interval == 0:
                self._log_training(train_stats, num_updates)
            
            # ---------- 评估 ----------
            if self.config.eval_interval > 0 and self.total_steps % self.config.eval_interval < self.config.n_steps:
                eval_stats = self.evaluate(self.config.eval_episodes)
                if self.config.verbose >= 1:
                    print(f"  [评估] 平均奖励: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            
            # ---------- 保存模型 ----------
            if self.config.save_interval > 0 and self.total_steps % self.config.save_interval < self.config.n_steps:
                self._save_model(num_updates)
            
            # ---------- 回调 ----------
            if callback is not None:
                callback(self, train_stats)
        
        # 训练完成
        total_time = time.time() - self._start_time
        
        if self.config.verbose >= 1:
            print("=" * 60)
            print("训练完成!")
            print(f"  总步数: {self.total_steps:,}")
            print(f"  总episode数: {self.episodes}")
            print(f"  总时间: {total_time:.1f}s")
            print(f"  平均FPS: {self.total_steps / total_time:.1f}")
            if len(self.episode_rewards) > 0:
                print(f"  最近100 episode平均奖励: {np.mean(self.episode_rewards):.2f}")
            print("=" * 60)
        
        return {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'total_time': total_time,
            'final_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }
    
    def _process_obs(self, obs_raw) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理原始观测
        
        将环境返回的观测转换为MAPPO期望的格式
        """
        # 如果是字典，提取观测
        if isinstance(obs_raw, dict):
            obs = obs_raw.get('observation', obs_raw)
        else:
            obs = obs_raw
        
        # 确保形状正确
        obs = np.array(obs, dtype=np.float32)
        
        if obs.ndim == 1:
            # 单智能体观测，复制给所有智能体
            obs = np.tile(obs, (self.mappo.config.num_agents, 1))
        
        # 确保观测维度匹配
        if obs.shape[1] != self.mappo.config.obs_dim:
            # 填充或截断
            new_obs = np.zeros((obs.shape[0], self.mappo.config.obs_dim), dtype=np.float32)
            min_dim = min(obs.shape[1], self.mappo.config.obs_dim)
            new_obs[:, :min_dim] = obs[:, :min_dim]
            obs = new_obs
        
        # 全局状态（简单平均）
        global_state = obs.mean(axis=0)
        
        # 如果全局状态维度不匹配，调整
        if len(global_state) != self.mappo.config.global_state_dim:
            new_gs = np.zeros(self.mappo.config.global_state_dim, dtype=np.float32)
            min_dim = min(len(global_state), self.mappo.config.global_state_dim)
            new_gs[:min_dim] = global_state[:min_dim]
            global_state = new_gs
        
        return obs, global_state
    
    def _convert_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将MAPPO动作转换为环境期望的格式
        
        环境期望: (num_agents, 2) - [handover, offload]
        """
        handover = actions['handover']
        offload = actions['offload']
        
        # 组合为环境动作
        env_actions = np.stack([handover, offload], axis=1)
        
        return env_actions
    
    def _log_training(self, stats: Dict[str, float], num_updates: int):
        """记录训练日志"""
        elapsed = time.time() - self._start_time
        fps = self.total_steps / elapsed if elapsed > 0 else 0
        
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        print(f"[更新 {num_updates}] "
              f"步数: {self.total_steps:,} | "
              f"EP: {self.episodes} | "
              f"奖励: {mean_reward:.2f} | "
              f"Actor损失: {stats.get('actor_loss', 0):.4f} | "
              f"Critic损失: {stats.get('critic_loss', 0):.4f} | "
              f"熵: {stats.get('entropy', 0):.4f} | "
              f"FPS: {fps:.0f}")
    
    def _save_model(self, num_updates: int):
        """保存模型"""
        os.makedirs(self.config.save_path, exist_ok=True)
        path = os.path.join(
            self.config.save_path,
            f"mappo_step_{self.total_steps}.pt"
        )
        self.mappo.save(path)
        
        if self.config.verbose >= 2:
            print(f"  [保存] 模型已保存到 {path}")
    
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """
        评估当前策略
        
        Args:
            n_episodes: 评估episode数
            
        Returns:
            评估统计
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs_raw, info = self.env.reset()
            observations, _ = self._process_obs(obs_raw)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                candidate_masks = self._get_candidate_masks()
                
                # 使用确定性策略
                actions, _, _ = self.mappo.act(
                    observations, candidate_masks, deterministic=True
                )
                
                env_actions = self._convert_actions(actions)
                next_obs_raw, rewards, terminated, truncated, info = self.env.step(env_actions)
                done = terminated or truncated
                
                if not done:
                    observations, _ = self._process_obs(next_obs_raw)
                
                episode_reward += float(np.mean(rewards)) if not np.isscalar(rewards) else rewards
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards)
        }
