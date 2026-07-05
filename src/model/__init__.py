"""
神经网络模型模块
================

本模块包含用于联合切换与卸载优化的神经网络模型。

【模块结构】
- layers.py      : 基础神经网络层（注意力层、MLP等）
- hetero_gnn.py  : 异质图注意力网络(HAN)，用于编码网络状态
- actor.py       : 混合动作Actor网络（离散切换+连续卸载）
- critic.py      : 共享Critic网络（价值函数估计）

【模型架构概览】
```
环境状态 → 异质图 → HAN编码器 → 节点嵌入
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Actor网络                        Critic网络
         (每个用户一个)                      (共享/集中式)
                    ↓                               ↓
              混合动作                          状态价值
         [切换决策, 卸载比例]                    V(s)
```

【参考论文】
1. HAN: "Heterogeneous Graph Attention Network" (WWW 2019)
2. MAPPO: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2021)
"""

from .layers import (
    MLP,
    GraphAttentionLayer,
    HeterogeneousAttentionLayer,
    SemanticAttention
)

from .hetero_gnn import (
    HANConfig,
    HeterogeneousAttentionNetwork,
    HANEncoder
)

from .actor import (
    ActorConfig,
    HybridActor,
    MultiAgentActor
)

from .critic import (
    CriticConfig,
    SharedCritic,
    CentralizedCritic
)

from .candidate_attention import (
    CandidateAttentionConfig,
    SatelliteLoadEncoder,
    CandidateAttentionActor,
)

__all__ = [
    # Layers
    'MLP',
    'GraphAttentionLayer', 
    'HeterogeneousAttentionLayer',
    'SemanticAttention',
    # HAN
    'HANConfig',
    'HeterogeneousAttentionNetwork',
    'HANEncoder',
    # Actor
    'ActorConfig',
    'HybridActor',
    'MultiAgentActor',
    # Critic
    'CriticConfig',
    'SharedCritic',
    'CentralizedCritic',
    # Candidate attention
    'CandidateAttentionConfig',
    'SatelliteLoadEncoder',
    'CandidateAttentionActor',
]
