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

当前使用 CentralizedCritic 聚合所有智能体信息。

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
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

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
