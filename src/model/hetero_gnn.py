"""
异质图注意力网络 (Heterogeneous Attention Network, HAN)
======================================================

本模块实现HAN，用于编码LEO卫星网络的异质图结构。

【HAN架构概览】
```
                    异质图
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    元路径1        元路径2       元路径3
   (U-S-U)        (S-U-S)       (S-S-S)
        │             │             │
        ▼             ▼             ▼
    节点级注意力   节点级注意力   节点级注意力
    (GAT-like)    (GAT-like)    (GAT-like)
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                语义级注意力
                (聚合元路径)
                      │
                      ▼
                最终节点嵌入
```

【与GCN/GAT的区别】
| 模型 | 图类型 | 注意力机制 |
|------|--------|-----------|
| GCN  | 同质图 | 无（固定权重）|
| GAT  | 同质图 | 节点级注意力 |
| HAN  | 异质图 | 节点级 + 语义级 |

【本项目元路径】
1. User -> Satellite -> User (U-S-U)
   - 通过共同可见卫星连接的用户
   - 用于建模用户间的竞争/干扰关系
   
2. Satellite -> User -> Satellite (S-U-S)
   - 服务相同用户的卫星
   - 用于建模切换候选关系
   
3. Satellite -> Satellite -> Satellite (S-S-S)
   - 通过ISL连接的卫星二跳邻居
   - 用于建模网络拓扑

【参考论文】
"Heterogeneous Graph Attention Network" (Wang et al., WWW 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .layers import MLP, HeterogeneousAttentionLayer, SemanticAttention


@dataclass
class HANConfig:
    """
    HAN配置参数
    
    【参数说明】
    - hidden_dim: 隐藏层维度，影响模型容量
    - num_heads: 注意力头数，多头可以捕获不同方面的关系
    - num_layers: HAN层数，深层可以捕获高阶关系
    - dropout: 正则化，防止过拟合
    """
    # 特征维度
    satellite_in_dim: int = 10         # 卫星输入特征维度
    user_in_dim: int = 13              # 用户输入特征维度
    hidden_dim: int = 64               # 隐藏层维度
    out_dim: int = 64                  # 输出嵌入维度
    
    # 注意力参数
    num_heads: int = 4                 # 注意力头数
    num_layers: int = 2                # HAN层数
    
    # 边特征
    use_edge_features: bool = True     # 是否使用边特征
    user_sat_edge_dim: int = 5         # 用户-卫星边特征维度
    isl_edge_dim: int = 3              # 星间链路边特征维度
    
    # 正则化
    dropout: float = 0.1               # Dropout比例
    
    # 元路径配置
    metapaths: List[str] = field(default_factory=lambda: [
        'user-visible-satellite-user',
        'user-serving-satellite-user',
        'user-nearby-user',
        'satellite-visible-user-satellite',
        'satellite-isl-satellite'
    ])


class MetapathEncoder(nn.Module):
    """
    元路径编码器
    
    对单条元路径进行编码，使用节点级注意力聚合邻居。
    
    【工作流程】
    1. 沿元路径传播特征
    2. 使用注意力聚合邻居
    3. 输出目标节点的更新嵌入
    
    【示例】
    对于元路径 User -> Satellite -> User:
    - 第1跳: User特征 → Satellite (聚合连接的用户)
    - 第2跳: Satellite特征 → User (聚合可见的卫星)
    """
    
    def __init__(
        self,
        metapath: List[Tuple[str, str, str]],
        node_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_edge_features: bool,
        edge_dims: Dict[str, int]
    ):
        """
        初始化元路径编码器
        
        Args:
            metapath: 元路径，如 [('user', 'connect', 'satellite'), ('satellite', 'serve', 'user')]
            node_dims: 节点类型到特征维度的映射
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            dropout: Dropout比例
            use_edge_features: 是否使用边特征
            edge_dims: 边类型到特征维度的映射
        """
        super().__init__()
        
        self.metapath = metapath
        self.num_hops = len(metapath)
        
        # 为元路径中的每一跳创建注意力层
        self.layers = nn.ModuleList()
        
        for i, (src_type, edge_type, dst_type) in enumerate(metapath):
            # 确定输入维度
            if i == 0:
                src_dim = node_dims[src_type]
            else:
                src_dim = hidden_dim * num_heads
            
            dst_dim = node_dims[dst_type] if i == 0 else hidden_dim * num_heads
            
            # 边特征维度
            edge_key = f"{src_type}-{dst_type}"
            edge_dim = edge_dims.get(edge_key, 0) if use_edge_features else 0
            
            layer = HeterogeneousAttentionLayer(
                src_dim=src_dim,
                dst_dim=dst_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_features=use_edge_features and edge_dim > 0,
                edge_dim=edge_dim
            )
            self.layers.append(layer)
        
        # 确定输出节点类型
        self.output_node_type = metapath[-1][2]
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        edge_features: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: {node_type: features}
            edge_index: {(src, rel, dst): (src_idx, dst_idx)}
            edge_features: {(src, rel, dst): features}
            
        Returns:
            目标节点嵌入, (num_target_nodes, hidden_dim * num_heads)
        """
        # 当前特征（会在传播过程中更新）
        current_features = dict(node_features)
        
        for i, (src_type, rel, dst_type) in enumerate(self.metapath):
            edge_key = (src_type, rel, dst_type)
            
            # 检查边是否存在
            if edge_key not in edge_index:
                # 尝试反向边
                reverse_key = (dst_type, 'rev_' + rel, src_type)
                if reverse_key in edge_index:
                    edge_key = reverse_key
                    src_type, dst_type = dst_type, src_type
                else:
                    raise ValueError(f"边类型 {edge_key} 不存在")
            
            src_feat = current_features[src_type]
            dst_feat = current_features[dst_type]
            idx = edge_index[edge_key]
            
            # 边特征
            edge_feat = None
            if edge_features is not None and edge_key in edge_features:
                edge_feat = edge_features[edge_key]
            
            # 注意力传播
            new_dst_feat = self.layers[i](
                src_feat=src_feat,
                dst_feat=dst_feat,
                edge_index=idx,
                edge_feat=edge_feat
            )
            
            # 更新目标节点特征
            current_features[dst_type] = F.elu(new_dst_feat)
        
        return current_features[self.output_node_type]


class HeterogeneousAttentionNetwork(nn.Module):
    """
    异质图注意力网络 (HAN)
    
    【核心思想】
    1. 为每条元路径学习不同的节点表示
    2. 使用语义注意力聚合多条元路径的表示
    3. 输出最终的节点嵌入
    
    【本项目配置】
    - 卫星节点：通过 S-U-S 和 S-S-S 元路径聚合
    - 用户节点：通过 U-S-U 元路径聚合
    """
    
    def __init__(self, config: HANConfig):
        """
        初始化HAN
        
        Args:
            config: HAN配置
        """
        super().__init__()
        
        self.config = config
        
        # ---------- 节点特征投影 ----------
        # 将不同类型的节点投影到相同维度
        self.node_projections = nn.ModuleDict({
            'satellite': nn.Linear(config.satellite_in_dim, config.hidden_dim),
            'user': nn.Linear(config.user_in_dim, config.hidden_dim)
        })
        
        # ---------- 元路径编码器 ----------
        # 定义元路径结构
        self.metapath_structures = {
            'user-visible-satellite-user': [
                ('user', 'visible', 'satellite'),
                ('satellite', 'visible_rev', 'user')
            ],
            'user-serving-satellite-user': [
                ('user', 'serving', 'satellite'),
                ('satellite', 'serving_rev', 'user')
            ],
            'user-nearby-user': [
                ('user', 'nearby', 'user')
            ],
            'satellite-visible-user-satellite': [
                ('satellite', 'visible_rev', 'user'),
                ('user', 'visible', 'satellite')
            ],
            'satellite-isl-satellite': [
                ('satellite', 'isl', 'satellite'),
                ('satellite', 'isl', 'satellite')
            ]
        }
        
        # 节点维度映射
        node_dims = {
            'satellite': config.hidden_dim,
            'user': config.hidden_dim
        }

        # 边维度映射
        edge_dims = {
            'user-satellite': config.user_sat_edge_dim,
            'satellite-user': config.user_sat_edge_dim,
            'user-user': 1,  # nearby user edge dim
            'satellite-satellite': config.isl_edge_dim
        }
        
        # 为每条元路径创建编码器
        self.metapath_encoders = nn.ModuleDict()
        for mp_name in config.metapaths:
            if mp_name in self.metapath_structures:
                encoder = MetapathEncoder(
                    metapath=self.metapath_structures[mp_name],
                    node_dims=node_dims,
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    use_edge_features=config.use_edge_features,
                    edge_dims=edge_dims
                )
                self.metapath_encoders[mp_name] = encoder
        
        # ---------- 语义注意力 ----------
        # 用户节点的语义注意力
        self.user_semantic_attn = SemanticAttention(
            in_dim=config.hidden_dim * config.num_heads,
            hidden_dim=config.hidden_dim
        )
        
        # 卫星节点的语义注意力
        self.sat_semantic_attn = SemanticAttention(
            in_dim=config.hidden_dim * config.num_heads,
            hidden_dim=config.hidden_dim
        )
        
        # ---------- 输出投影 ----------
        self.output_proj = nn.ModuleDict({
            'satellite': nn.Linear(config.hidden_dim * config.num_heads, config.out_dim),
            'user': nn.Linear(config.hidden_dim * config.num_heads, config.out_dim)
        })
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        edge_features: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征字典
                - 'satellite': (num_sats, satellite_in_dim)
                - 'user': (num_users, user_in_dim)
            edge_index: 边索引字典
            edge_features: 边特征字典（可选）
            return_attention: 是否返回注意力权重
            
        Returns:
            节点嵌入字典
            - 'satellite': (num_sats, out_dim)
            - 'user': (num_users, out_dim)
        """
        # 1. 节点特征投影
        projected = {}
        for node_type, feat in node_features.items():
            if node_type in self.node_projections:
                projected[node_type] = F.relu(self.node_projections[node_type](feat))
            else:
                projected[node_type] = feat
        
        # 2. 元路径编码
        user_embeddings = []
        sat_embeddings = []
        
        for mp_name, encoder in self.metapath_encoders.items():
            try:
                mp_embed = encoder(projected, edge_index, edge_features)
                
                # 根据输出节点类型分类
                if encoder.output_node_type == 'user':
                    user_embeddings.append(mp_embed)
                else:
                    sat_embeddings.append(mp_embed)
            except Exception as e:
                # 如果某条元路径失败（如缺少边），跳过
                continue
        
        # 3. 语义注意力聚合
        output = {}
        attention_weights = {}
        
        # 用户嵌入
        if user_embeddings:
            user_final, user_attn = self.user_semantic_attn(
                user_embeddings, return_weights=return_attention
            )
            output['user'] = self.output_proj['user'](self.dropout(user_final))
            if return_attention:
                attention_weights['user'] = user_attn
        else:
            # 如果没有用户相关元路径，使用投影后的特征
            output['user'] = self.output_proj['user'](
                projected['user'].repeat(1, self.config.num_heads)
            )
        
        # 卫星嵌入
        if sat_embeddings:
            sat_final, sat_attn = self.sat_semantic_attn(
                sat_embeddings, return_weights=return_attention
            )
            output['satellite'] = self.output_proj['satellite'](self.dropout(sat_final))
            if return_attention:
                attention_weights['satellite'] = sat_attn
        else:
            output['satellite'] = self.output_proj['satellite'](
                projected['satellite'].repeat(1, self.config.num_heads)
            )
        
        if return_attention:
            return output, attention_weights
        return output


class HANEncoder(nn.Module):
    """
    HAN编码器（简化接口）
    
    将异质图数据转换为节点嵌入，提供更简洁的接口。
    
    【使用方式】
    ```python
    encoder = HANEncoder(config)
    
    # 从环境获取图数据
    graph = builder.build(env)
    
    # 编码
    embeddings = encoder.encode(graph)
    # embeddings['satellite']: (66, 64)
    # embeddings['user']: (5, 64)
    ```
    """
    
    def __init__(self, config: HANConfig = None):
        """
        初始化编码器
        
        Args:
            config: HAN配置，None则使用默认配置
        """
        super().__init__()
        
        self.config = config or HANConfig()
        self.han = HeterogeneousAttentionNetwork(self.config)
    
    def encode(
        self,
        graph_data,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        编码异质图
        
        Args:
            graph_data: HeteroGraphData 实例或包含以下键的字典:
                - node_features: Dict[str, np.ndarray]
                - edge_index: Dict[Tuple, Tuple[np.ndarray, np.ndarray]]
                - edge_features: Dict[Tuple, np.ndarray]
            return_attention: 是否返回注意力权重
            
        Returns:
            节点嵌入字典
        """
        # 转换为PyTorch张量
        node_features = {}
        for node_type, feat in graph_data.node_features.items():
            if isinstance(feat, np.ndarray):
                node_features[node_type] = torch.from_numpy(feat).float()
            else:
                node_features[node_type] = feat
        
        edge_index = {}
        for edge_type, (src, dst) in graph_data.edge_index.items():
            if isinstance(src, np.ndarray):
                src = torch.from_numpy(src).long()
                dst = torch.from_numpy(dst).long()
            edge_index[edge_type] = (src, dst)
        
        edge_features = {}
        if hasattr(graph_data, 'edge_features') and graph_data.edge_features:
            for edge_type, feat in graph_data.edge_features.items():
                if isinstance(feat, np.ndarray):
                    edge_features[edge_type] = torch.from_numpy(feat).float()
                else:
                    edge_features[edge_type] = feat
        
        # 前向传播
        return self.han(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            return_attention=return_attention
        )

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        edge_features: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """兼容直接传入张量化图数据的前向接口。"""
        return self.han(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            return_attention=return_attention
        )
    
    def get_output_dim(self) -> int:
        """获取输出嵌入维度"""
        return self.config.out_dim
