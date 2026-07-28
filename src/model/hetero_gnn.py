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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, replace

from .layers import HeterogeneousAttentionLayer, SemanticAttention


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
    serving_edge_dim: int = 2          # 服务关系边特征维度
    nearby_user_edge_dim: int = 1      # 用户邻接边特征维度
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
        edge_dims: Dict[Tuple[str, str, str], int]
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
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
            )
        head_dim = hidden_dim // num_heads
        
        # 为元路径中的每一跳创建注意力层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for src_type, edge_type, dst_type in metapath:
            # 边特征维度
            edge_key = (src_type, edge_type, dst_type)
            edge_dim = edge_dims.get(edge_key, 0) if use_edge_features else 0

            layer = HeterogeneousAttentionLayer(
                src_dim=node_dims[src_type],
                dst_dim=node_dims[dst_type],
                out_dim=head_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_features=use_edge_features and edge_dim > 0,
                edge_dim=edge_dim
            )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
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
            current_features[dst_type] = self.norms[i](
                dst_feat + F.elu(new_dst_feat)
            )
        
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
            ('user', 'visible', 'satellite'): config.user_sat_edge_dim,
            ('satellite', 'visible_rev', 'user'): config.user_sat_edge_dim,
            ('user', 'serving', 'satellite'): config.serving_edge_dim,
            ('satellite', 'serving_rev', 'user'): config.serving_edge_dim,
            ('user', 'nearby', 'user'): config.nearby_user_edge_dim,
            ('satellite', 'isl', 'satellite'): config.isl_edge_dim,
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
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim
        )
        
        # 卫星节点的语义注意力
        self.sat_semantic_attn = SemanticAttention(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim
        )
        
        # ---------- 输出投影 ----------
        self.output_proj = nn.ModuleDict({
            'satellite': nn.Linear(config.hidden_dim, config.out_dim),
            'user': nn.Linear(config.hidden_dim, config.out_dim)
        })
        
        self.dropout = nn.Dropout(config.dropout)
        self.last_executed_metapaths: Tuple[str, ...] = ()
    
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
        executed_metapaths = []
        
        for mp_name, encoder in self.metapath_encoders.items():
            try:
                mp_embed = encoder(projected, edge_index, edge_features)
            except Exception as exc:
                node_shapes = {
                    key: tuple(value.shape)
                    for key, value in projected.items()
                }
                relation_shapes = {
                    str(key): tuple(value.shape)
                    for key, value in (edge_features or {}).items()
                }
                raise RuntimeError(
                    f"元路径 {mp_name} 前向失败；"
                    f"node_shapes={node_shapes}, "
                    f"edge_feature_shapes={relation_shapes}"
                ) from exc
            executed_metapaths.append(mp_name)

            # 根据输出节点类型分类
            if encoder.output_node_type == 'user':
                user_embeddings.append(mp_embed)
            else:
                sat_embeddings.append(mp_embed)
        self.last_executed_metapaths = tuple(executed_metapaths)
        
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
                projected['user']
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
                projected['satellite']
            )
        
        if return_attention:
            return output, attention_weights
        return output


class HANEncoder(nn.Module):
    """
    HAN编码器（简化接口）
    
    将张量化异质图转换为卫星与用户节点嵌入。
    """
    
    def __init__(self, config: HANConfig = None):
        """
        初始化编码器
        
        Args:
            config: HAN配置，None则使用默认配置
        """
        super().__init__()
        
        self.config = config or HANConfig()
        if self.config.num_layers < 1:
            raise ValueError("HANConfig.num_layers 必须至少为 1")
        layers = []
        for layer_index in range(self.config.num_layers):
            layer_config = self.config
            if layer_index > 0:
                layer_config = replace(
                    self.config,
                    satellite_in_dim=self.config.out_dim,
                    user_in_dim=self.config.out_dim,
                    num_layers=1,
                )
            layers.append(HeterogeneousAttentionNetwork(layer_config))
        self.han_layers = nn.ModuleList(layers)

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        edge_features: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """编码张量化异质图数据。"""
        current_features = node_features
        layer_attention = {}
        for layer_index, layer in enumerate(self.han_layers):
            result = layer(
                node_features=current_features,
                edge_index=edge_index,
                edge_features=edge_features,
                return_attention=return_attention,
            )
            if return_attention:
                current_features, attention = result
                layer_attention[f"layer_{layer_index}"] = attention
            else:
                current_features = result
        if return_attention:
            return current_features, layer_attention
        return current_features
    
    def get_last_executed_metapath_count(self) -> int:
        """返回最近一次前向中全部 HAN 层成功执行的元路径数量。"""
        return sum(
            len(layer.last_executed_metapaths)
            for layer in self.han_layers
        )
