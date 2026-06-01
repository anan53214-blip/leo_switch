"""
强化学习算法模块
================

本模块实现MAPPO（Multi-Agent PPO）算法及其相关组件。

【模块结构】
- buffer.py   : 经验回放缓冲区（Rollout Buffer）
- mappo.py    : MAPPO算法实现
- runner.py   : 训练运行器（环境交互+训练循环）

【MAPPO简介】
MAPPO是PPO算法在多智能体场景的扩展，采用CTDE范式：
- 集中训练(Centralized Training): Critic访问全局信息
- 分布执行(Decentralized Execution): Actor只用本地观测

【与单智能体PPO的区别】
1. 多个Actor并行决策
2. 共享Critic评估全局状态
3. 考虑智能体间的协作/竞争

【参考论文】
"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
(Yu et al., NeurIPS 2021)
"""

from .buffer import (
    RolloutBuffer,
    MultiAgentRolloutBuffer
)

from .replay_buffer import MultiAgentReplayBuffer

from .mappo import (
    MAPPOConfig,
    MAPPO
)

from .attention_mappo import AttentionMAPPO

from .maddpg import (
    MADDPGConfig,
    MADDPGActor,
    HANCentralizedCritic,
    MADDPGAlgorithm,
)

from .pdqn import (
    PDQNConfig,
    PDQNNetwork,
    PDQNParameterNet,
    PDQNParameterNets,
    PDQNAlgorithm,
)

from .runner import (
    RunnerConfig,
    Runner
)

__all__ = [
    'RolloutBuffer',
    'MultiAgentRolloutBuffer',
    'MultiAgentReplayBuffer',
    'MAPPOConfig',
    'MAPPO',
    'AttentionMAPPO',
    'MADDPGConfig',
    'MADDPGActor',
    'HANCentralizedCritic',
    'MADDPGAlgorithm',
    'PDQNConfig',
    'PDQNNetwork',
    'PDQNParameterNet',
    'PDQNParameterNets',
    'PDQNAlgorithm',
    'RunnerConfig',
    'Runner',
]
