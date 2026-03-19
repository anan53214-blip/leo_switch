"""
Critic网络模块
==============

本模块实现用于价值函数估计的Critic网络。

【Critic的作用】
在Actor-Critic架构中，Critic估计状态价值V(s)或动作价值Q(s,a)。
本项目使用V(s)，即状态价值函数。

【MAPPO中的Critic设计】
MAPPO使用"集中式训练，分布式执行"(CTDE)范式：
- 训练时：Critic可以访问全局状态（所有智能体的观测）
- 执行时：Actor只使用本地观测

【两种Critic实现】
1. SharedCritic: 简单的MLP，输入为状态嵌入
2. CentralizedCritic: 聚合所有智能体信息

【与Actor的区别】
- Actor: 为每个智能体输出动作分布
- Critic: 输出全局状态价值（标量）

【网络结构】
```
全局状态嵌入
     │
     ▼
  MLP层
     │
     ▼
   V(s)
 (标量)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .layers import MLP


@dataclass
class CriticConfig:
    """
    Critic网络配置
    
    【参数说明】
    - input_dim: 状态嵌入维度，取决于HAN输出和聚合方式
    - hidden_dims: 隐藏层，一般比Actor深/宽
    - use_orthogonal_init: 正交初始化，在RL中常用
    """
    # 输入维度
    input_dim: int = 64                # 用户嵌入维度（来自HAN+拼接特征）
    sat_input_dim: int = 64            # 卫星嵌入维度（来自HAN输出）
    num_agents: int = 5                # 智能体数量（用于集中式Critic）
    
    # 网络结构
    hidden_dims: List[int] = None      # 隐藏层维度
    
    # 初始化
    use_orthogonal_init: bool = True   # 使用正交初始化
    init_gain: float = 0.01            # 最后一层的初始化增益
    
    # 正则化
    dropout: float = 0.0               # Critic一般不用dropout
    use_layer_norm: bool = True        # 使用LayerNorm稳定训练
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class SharedCritic(nn.Module):
    """
    共享Critic网络
    
    【设计思想】
    所有智能体共享同一个Critic来评估全局状态价值。
    
    输入可以是：
    1. 全局状态嵌入（如所有节点嵌入的平均/池化）
    2. 拼接所有智能体的嵌入
    
    【使用方式】
    ```python
    critic = SharedCritic(config)
    
    # 获取全局状态表示
    global_state = get_global_state(embeddings)  # (batch, input_dim)
    
    # 估计价值
    value = critic(global_state)  # (batch, 1)
    ```
    """
    
    def __init__(self, config: CriticConfig):
        """
        初始化Critic网络
        
        Args:
            config: Critic配置
        """
        super().__init__()
        
        self.config = config
        
        # 构建MLP
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        
        # 输出层（单个值）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化
        if config.use_orthogonal_init:
            self._orthogonal_init()
    
    def _orthogonal_init(self):
        """
        正交初始化
        
        【为什么使用正交初始化？】
        1. 在深度网络中保持梯度范数
        2. 避免梯度消失/爆炸
        3. 在RL中特别有效（PPO论文推荐）
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # 最后一层使用较小的增益
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=self.config.init_gain)
            nn.init.constant_(final_layer.bias, 0)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_embedding: 全局状态嵌入, (batch_size, input_dim)
            
        Returns:
            状态价值, (batch_size, 1)
        """
        return self.net(state_embedding)


class CentralizedCritic(nn.Module):
    """
    集中式Critic网络
    
    【设计思想】
    在CTDE范式中，Critic可以在训练时访问所有智能体的信息：
    - 所有用户的嵌入
    - 所有卫星的嵌入
    - 全局图结构信息
    
    【聚合方式】
    1. 简单拼接：将所有智能体嵌入拼接
    2. 注意力聚合：使用注意力机制加权聚合
    3. 图池化：对图嵌入进行池化
    
    【本实现】
    使用混合聚合：
    - 用户嵌入：平均池化
    - 卫星嵌入：平均池化
    - 拼接后输入MLP
    
    【优势】
    - 可以捕获智能体间的交互
    - 提供更准确的价值估计
    - 缓解信用分配问题
    """
    
    def __init__(self, config: CriticConfig):
        """
        初始化集中式Critic
        
        Args:
            config: Critic配置
        """
        super().__init__()
        
        self.config = config
        
        # ---------- 智能体嵌入聚合 ----------
        # 用户嵌入处理
        self.user_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0] // 2),
            nn.ReLU()
        )
        
        # 卫星嵌入处理（可选）
        self.sat_encoder = nn.Sequential(
            nn.Linear(config.sat_input_dim, config.hidden_dims[0] // 2),
            nn.ReLU()
        )
        
        # ---------- 价值网络 ----------
        # 输入维度：用户聚合 + 卫星聚合
        value_input_dim = config.hidden_dims[0]  # 两个 hidden_dims[0]//2 拼接
        
        layers = []
        prev_dim = value_input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
        
        # 初始化
        if config.use_orthogonal_init:
            self._orthogonal_init()
    
    def _orthogonal_init(self):
        """正交初始化"""
        for module in [self.user_encoder, self.sat_encoder, self.value_net]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层小增益
        final_layer = self.value_net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=self.config.init_gain)
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        satellite_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_embeddings: 用户嵌入, (batch, num_users, embed_dim) 或 (num_users, embed_dim)
            satellite_embeddings: 卫星嵌入, (batch, num_sats, embed_dim) 或 (num_sats, embed_dim)
            
        Returns:
            状态价值, (batch, 1) 或 (1,)
        """
        # 处理维度
        if user_embeddings.dim() == 2:
            user_embeddings = user_embeddings.unsqueeze(0)  # (1, N, D)
        
        # 用户嵌入聚合：平均池化
        user_enc = self.user_encoder(user_embeddings)  # (B, N, D')
        user_agg = user_enc.mean(dim=1)  # (B, D')
        
        # 卫星嵌入聚合
        if satellite_embeddings is not None:
            if satellite_embeddings.dim() == 2:
                satellite_embeddings = satellite_embeddings.unsqueeze(0)
            sat_enc = self.sat_encoder(satellite_embeddings)
            sat_agg = sat_enc.mean(dim=1)
        else:
            # 如果没有卫星嵌入，用零填充
            sat_agg = torch.zeros_like(user_agg)
        
        # 拼接
        combined = torch.cat([user_agg, sat_agg], dim=-1)
        
        # 估计价值
        value = self.value_net(combined)
        
        return value
    
    def get_value_from_graph(
        self,
        node_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        从图嵌入直接计算价值（便捷接口）
        
        Args:
            node_embeddings: HAN输出的节点嵌入字典
            
        Returns:
            状态价值
        """
        user_emb = node_embeddings.get('user')
        sat_emb = node_embeddings.get('satellite')
        
        return self.forward(user_emb, sat_emb)


class MultiAgentCritic(nn.Module):
    """
    多智能体Critic（可选实现）
    
    【与CentralizedCritic的区别】
    - CentralizedCritic: 输出单个全局价值 V(s)
    - MultiAgentCritic: 为每个智能体输出价值 V_i(s)
    
    【使用场景】
    - 异构智能体（不同类型用户）
    - 需要个体价值分解
    - 信用分配问题严重时
    
    【本项目选择】
    使用CentralizedCritic，因为：
    - 用户是同构的
    - 使用参数共享
    - MAPPO推荐共享Critic
    """
    
    def __init__(self, config: CriticConfig):
        """
        初始化多智能体Critic
        
        Args:
            config: Critic配置
        """
        super().__init__()
        
        self.config = config
        
        # 每个智能体的价值头
        self.value_heads = nn.ModuleList([
            SharedCritic(config) for _ in range(config.num_agents)
        ])
    
    def forward(
        self,
        agent_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_embeddings: 每个智能体的状态嵌入, (num_agents, input_dim)
            
        Returns:
            每个智能体的价值, (num_agents, 1)
        """
        values = []
        for i, head in enumerate(self.value_heads):
            v = head(agent_embeddings[i:i+1])
            values.append(v)
        
        return torch.cat(values, dim=0)


# ================================================================
#                    工具函数
# ================================================================

def create_global_state(
    user_embeddings: torch.Tensor,
    satellite_embeddings: torch.Tensor,
    method: str = 'concat_mean'
) -> torch.Tensor:
    """
    创建全局状态表示
    
    【聚合方法】
    - 'concat_mean': 拼接用户和卫星的平均嵌入
    - 'concat_max': 拼接用户和卫星的最大嵌入
    - 'attention': 使用注意力加权（需要额外参数）
    
    Args:
        user_embeddings: (num_users, embed_dim)
        satellite_embeddings: (num_sats, embed_dim)
        method: 聚合方法
        
    Returns:
        全局状态, (embed_dim * 2,) 或 (embed_dim,)
    """
    if method == 'concat_mean':
        user_state = user_embeddings.mean(dim=0)
        sat_state = satellite_embeddings.mean(dim=0)
        return torch.cat([user_state, sat_state], dim=-1)
    
    elif method == 'concat_max':
        user_state = user_embeddings.max(dim=0)[0]
        sat_state = satellite_embeddings.max(dim=0)[0]
        return torch.cat([user_state, sat_state], dim=-1)
    
    elif method == 'mean':
        # 所有节点平均
        all_embeddings = torch.cat([user_embeddings, satellite_embeddings], dim=0)
        return all_embeddings.mean(dim=0)
    
    else:
        raise ValueError(f"未知聚合方法: {method}")
