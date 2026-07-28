"""
异质图构建器模块
================

本模块负责构建用于异质图注意力网络(HAN)的异质图结构。
异质图包含多种类型的节点和边，能够更好地表示LEO卫星网络的复杂拓扑。

【异质图vs同质图】
┌──────────────────────────────────────────────────────────────────┐
│  同质图              │  异质图                                  │
├──────────────────────────────────────────────────────────────────┤
│  - 单一节点类型       │  - 多种节点类型（卫星、用户）            │
│  - 单一边类型         │  - 多种边类型（ISL、UDL）               │
│  - 统一特征维度       │  - 不同类型可有不同特征维度              │
│  - GCN, GAT          │  - HAN, HGT, RGCN等                     │
└──────────────────────────────────────────────────────────────────┘

【本项目异质图结构】
节点类型：
  1. satellite - 卫星节点（66个）
  2. user - 用户节点（动态数量）

边类型（元路径）：
  1. user-satellite (U-S) - 用户到可见卫星的上行链路
  2. satellite-user (S-U) - 卫星到用户的下行链路（U-S的反向）
  3. satellite-satellite (S-S) - 星间链路ISL

【与HAN的关系】
HAN使用元路径(meta-path)来聚合不同类型的邻居信息：
- 元路径1: User -> Satellite -> User (U-S-U)
  含义：通过共同可见卫星连接的用户
- 元路径2: Satellite -> User -> Satellite (S-U-S)
  含义：服务共同用户的卫星
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .features import FeatureExtractor, NodeFeatures


@dataclass
class HeteroGraphData:
    """
    异质图数据结构

    【数据组织】
    - node_features: 按节点类型组织的特征字典
    - edge_index: 按边类型组织的邻接关系
    - edge_features: 按边类型组织的边特征
    
    示例：
    ```python
    graph.node_features['satellite']  # shape: (66, 10)
    graph.node_features['user']       # shape: (5, 12)
    graph.edge_index[('user', 'connect', 'satellite')]  # shape: (2, num_edges)
    ```
    """
    
    # 节点特征：{node_type: feature_matrix}
    node_features: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 边索引：{(src_type, edge_type, dst_type): (src_indices, dst_indices)}
    # 使用COO格式存储边，便于转换为稀疏张量
    edge_index: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    
    # 边特征：{(src_type, edge_type, dst_type): feature_matrix}
    edge_features: Dict[Tuple[str, str, str], np.ndarray] = field(default_factory=dict)
    
    # 节点数量：{node_type: count}
    num_nodes: Dict[str, int] = field(default_factory=dict)
    
    # 元信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class HeteroGraphBuilder:
    """
    异质图构建器
    
    从LEO卫星网络环境构建异质图，供HAN等图神经网络使用。
    
    【构建流程】
    1. 提取节点特征（卫星、用户）
    2. 提取边特征（用户-卫星链路、星间链路）
    3. 构建邻接关系（COO格式）
    4. 组装为HeteroGraphData
    
    """
    
    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        add_reverse_edges: bool = True,
        add_self_loops: bool = False
    ):
        """
        初始化图构建器
        
        Args:
            feature_extractor: 特征提取器实例，None则创建默认实例
            add_reverse_edges: 是否添加反向边
                              例如：除了user->satellite，还添加satellite->user
                              这对某些GNN架构很重要
            add_self_loops: 是否添加自环边
                           自环允许节点聚合自身信息
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.add_reverse_edges = add_reverse_edges
        self.add_self_loops = add_self_loops
    
    def build(self, env) -> HeteroGraphData:
        """
        构建异质图
        
        【构建步骤详解】
        
        Step 1: 提取节点特征
        ────────────────────
        从环境中提取所有卫星和用户的特征向量
        
        Step 2: 提取边及其特征
        ────────────────────
        - 用户-卫星边：基于可见性
        - 星间链路边：基于Walker星座拓扑
        
        Step 3: 构建邻接矩阵
        ────────────────────
        将边列表转换为COO格式的稀疏表示
        
        Step 4: 组装图数据
        ────────────────────
        将所有信息打包为HeteroGraphData
        
        Args:
            env: LEOSatelliteEnv环境实例
            
        Returns:
            HeteroGraphData: 构建好的异质图
        """
        # ============== Step 1: 提取节点特征 ==============
        node_features = self.feature_extractor.extract_node_features(env)
        
        # ============== Step 2: 提取边特征 ==============
        edge_features = self.feature_extractor.extract_edge_features(env)
        
        # ============== Step 3: 构建图数据 ==============
        graph = HeteroGraphData()
        
        # ---------- 填充节点信息 ----------
        graph.node_features['satellite'] = node_features.satellite_features
        graph.node_features['user'] = node_features.user_features
        graph.num_nodes['satellite'] = node_features.satellite_features.shape[0]
        graph.num_nodes['user'] = node_features.user_features.shape[0]
        
        def _put_edges(edge_type, edges, features, feature_dim):
            if edges:
                src_indices = np.array([e[0] for e in edges], dtype=np.int64)
                dst_indices = np.array([e[1] for e in edges], dtype=np.int64)
            else:
                src_indices = np.zeros((0,), dtype=np.int64)
                dst_indices = np.zeros((0,), dtype=np.int64)
                features = np.zeros((0, feature_dim), dtype=np.float32)
            graph.edge_index[edge_type] = (src_indices, dst_indices)
            graph.edge_features[edge_type] = features

        # ---------- 填充用户-卫星可见边 ----------
        _put_edges(('user', 'visible', 'satellite'),
                   edge_features.user_satellite_edges,
                   edge_features.user_satellite_features,
                   edge_features.user_satellite_edge_dim)

        if self.add_reverse_edges:
            src_indices, dst_indices = graph.edge_index[('user', 'visible', 'satellite')]
            reverse_type = ('satellite', 'visible_rev', 'user')
            graph.edge_index[reverse_type] = (dst_indices, src_indices)
            graph.edge_features[reverse_type] = graph.edge_features[('user', 'visible', 'satellite')]

        # ---------- 填充用户-卫星服务边 ----------
        _put_edges(('user', 'serving', 'satellite'),
                   edge_features.serving_edges,
                   edge_features.serving_features,
                   edge_features.serving_edge_dim)

        if self.add_reverse_edges:
            src_indices, dst_indices = graph.edge_index[('user', 'serving', 'satellite')]
            reverse_type = ('satellite', 'serving_rev', 'user')
            graph.edge_index[reverse_type] = (dst_indices, src_indices)
            graph.edge_features[reverse_type] = graph.edge_features[('user', 'serving', 'satellite')]

        # ---------- 填充用户-用户邻居边 ----------
        _put_edges(('user', 'nearby', 'user'),
                   edge_features.nearby_user_edges,
                   edge_features.nearby_user_features,
                   edge_features.nearby_user_edge_dim)

        # ---------- 填充星间链路边 ----------
        if edge_features.inter_satellite_edges:
            src_indices = np.array([e[0] for e in edge_features.inter_satellite_edges], dtype=np.int64)
            dst_indices = np.array([e[1] for e in edge_features.inter_satellite_edges], dtype=np.int64)
            isl_features = edge_features.inter_satellite_features
            if self.add_reverse_edges:
                forward_src = src_indices
                forward_dst = dst_indices
                src_indices = np.concatenate([forward_src, forward_dst])
                dst_indices = np.concatenate([forward_dst, forward_src])
                isl_features = np.concatenate([isl_features, isl_features])
            graph.edge_index[('satellite', 'isl', 'satellite')] = (src_indices, dst_indices)
            graph.edge_features[('satellite', 'isl', 'satellite')] = isl_features
        else:
            graph.edge_index[('satellite', 'isl', 'satellite')] = (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
            graph.edge_features[('satellite', 'isl', 'satellite')] = np.zeros(
                (0, edge_features.inter_satellite_edge_dim), dtype=np.float32
            )
        
        # ---------- 添加自环（可选）----------
        if self.add_self_loops:
            self._add_self_loops(graph, node_features)
        
        # ============== Step 4: 添加元信息 ==============
        graph.metadata = {
            'timestamp': env.current_time,
            'geometry_version': getattr(env, 'geometry_version', None),
            'num_satellites': env.num_satellites,
            'num_users': env.num_users,
            'feature_dims': self.feature_extractor.get_feature_dimensions()
        }
        
        return graph
    
    def _add_self_loops(self, graph: HeteroGraphData, node_features: NodeFeatures):
        """
        添加自环边
        
        自环允许节点在消息传递时聚合自身信息：
        h_i' = aggregate(h_i, {h_j | j ∈ N(i)})
                  ↑ 自环贡献
        
        Args:
            graph: 图数据
            node_features: 节点特征
        """
        # 卫星自环
        num_sats = node_features.satellite_features.shape[0]
        sat_self_loop_type = ('satellite', 'self', 'satellite')
        graph.edge_index[sat_self_loop_type] = (
            np.arange(num_sats),
            np.arange(num_sats)
        )
        # 自环边特征可以全为1或使用节点特征
        graph.edge_features[sat_self_loop_type] = np.ones((num_sats, 1), dtype=np.float32)
        
        # 用户自环
        num_users = node_features.user_features.shape[0]
        user_self_loop_type = ('user', 'self', 'user')
        graph.edge_index[user_self_loop_type] = (
            np.arange(num_users),
            np.arange(num_users)
        )
        graph.edge_features[user_self_loop_type] = np.ones((num_users, 1), dtype=np.float32)
    
    def get_metapaths(self) -> List[List[Tuple[str, str, str]]]:
        """
        获取HAN使用的元路径
        
        【元路径(Meta-path)概念】
        元路径是异质图中连接两个节点的语义路径模式。
        
        例如元路径 User --(connect)--> Satellite --(serve)--> User
        可以找到"共同连接某颗卫星的用户对"
        
        【本项目的元路径】
        对于用户节点的聚合：
        1. U-S-U: User -> Satellite -> User
           含义：通过共同可见卫星连接的用户
           应用：用户之间的干扰/资源竞争关系
        
        对于卫星节点的聚合：
        2. S-U-S: Satellite -> User -> Satellite  
           含义：服务相同用户的卫星
           应用：切换候选卫星的关联
        
        3. S-S-S: Satellite -> Satellite -> Satellite
           含义：通过ISL连接的卫星（二跳邻居）
           应用：网络拓扑结构
        
        Returns:
            元路径列表
        """
        metapaths = [
            [('user', 'visible', 'satellite'), ('satellite', 'visible_rev', 'user')],
            [('user', 'serving', 'satellite'), ('satellite', 'serving_rev', 'user')],
            [('user', 'nearby', 'user')],
            [('satellite', 'visible_rev', 'user'), ('user', 'visible', 'satellite')],
            [('satellite', 'isl', 'satellite'), ('satellite', 'isl', 'satellite')],
        ]
        return metapaths
