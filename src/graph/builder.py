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
from .features import FeatureExtractor, NodeFeatures, EdgeFeatures


@dataclass
class HeteroGraphData:
    """
    异质图数据结构
    
    【设计说明】
    这个数据结构可以直接转换为PyTorch Geometric的HeteroData
    或DGL的DGLHeteroGraph，便于与深度学习框架集成。
    
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
    
    def get_node_types(self) -> List[str]:
        """获取所有节点类型"""
        return list(self.node_features.keys())
    
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """获取所有边类型（三元组形式）"""
        return list(self.edge_index.keys())
    
    def num_edges(self, edge_type: Tuple[str, str, str] = None) -> int:
        """获取边数量"""
        if edge_type is None:
            return sum(idx[0].shape[0] for idx in self.edge_index.values())
        return self.edge_index[edge_type][0].shape[0]
    
    def to_dict(self) -> Dict:
        """转换为字典格式（便于序列化）"""
        return {
            'node_features': {k: v.tolist() for k, v in self.node_features.items()},
            'edge_index': {str(k): (v[0].tolist(), v[1].tolist()) 
                          for k, v in self.edge_index.items()},
            'edge_features': {str(k): v.tolist() for k, v in self.edge_features.items()},
            'num_nodes': self.num_nodes,
            'metadata': self.metadata
        }


class HeteroGraphBuilder:
    """
    异质图构建器
    
    从LEO卫星网络环境构建异质图，供HAN等图神经网络使用。
    
    【构建流程】
    1. 提取节点特征（卫星、用户）
    2. 提取边特征（用户-卫星链路、星间链路）
    3. 构建邻接关系（COO格式）
    4. 组装为HeteroGraphData
    
    【使用示例】
    ```python
    builder = HeteroGraphBuilder()
    graph = builder.build(env)
    
    # 转换为PyTorch Geometric格式
    pyg_data = builder.to_pyg(graph)
    
    # 转换为DGL格式
    dgl_graph = builder.to_dgl(graph)
    ```
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
    
    # ================================================================
    #                  格式转换方法
    # ================================================================
    
    def to_pyg(self, graph: HeteroGraphData):
        """
        转换为PyTorch Geometric的HeteroData格式
        
        【PyTorch Geometric HeteroData结构】
        ```python
        from torch_geometric.data import HeteroData
        
        data = HeteroData()
        data['satellite'].x = torch.tensor(...)  # 节点特征
        data['user'].x = torch.tensor(...)
        data['user', 'connect', 'satellite'].edge_index = torch.tensor(...)  # 边
        data['user', 'connect', 'satellite'].edge_attr = torch.tensor(...)   # 边特征
        ```
        
        Args:
            graph: HeteroGraphData实例
            
        Returns:
            PyTorch Geometric的HeteroData对象
        """
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("需要安装PyTorch Geometric: pip install torch-geometric")
        
        data = HeteroData()
        
        # 添加节点特征
        for node_type, features in graph.node_features.items():
            data[node_type].x = torch.tensor(features, dtype=torch.float32)
        
        # 添加边
        for edge_type, (src, dst) in graph.edge_index.items():
            data[edge_type].edge_index = torch.tensor(
                np.stack([src, dst], axis=0),
                dtype=torch.long
            )
            
            # 添加边特征
            if edge_type in graph.edge_features:
                data[edge_type].edge_attr = torch.tensor(
                    graph.edge_features[edge_type],
                    dtype=torch.float32
                )
        
        return data
    
    def to_dgl(self, graph: HeteroGraphData):
        """
        转换为DGL的DGLHeteroGraph格式
        
        【DGL异质图结构】
        ```python
        import dgl
        
        # 创建异质图
        hetero_graph = dgl.heterograph({
            ('user', 'connect', 'satellite'): (src_tensor, dst_tensor),
            ('satellite', 'isl', 'satellite'): (src_tensor, dst_tensor),
        })
        
        # 添加节点特征
        hetero_graph.nodes['satellite'].data['feat'] = torch.tensor(...)
        
        # 添加边特征
        hetero_graph.edges['connect'].data['feat'] = torch.tensor(...)
        ```
        
        Args:
            graph: HeteroGraphData实例
            
        Returns:
            DGL的DGLHeteroGraph对象
        """
        try:
            import torch
            import dgl
        except ImportError:
            raise ImportError("需要安装DGL: pip install dgl")
        
        # 构建边字典
        graph_data = {}
        for edge_type, (src, dst) in graph.edge_index.items():
            graph_data[edge_type] = (
                torch.tensor(src, dtype=torch.long),
                torch.tensor(dst, dtype=torch.long)
            )
        
        # 创建异质图，需要指定节点数量
        hetero_graph = dgl.heterograph(
            graph_data,
            num_nodes_dict=graph.num_nodes
        )
        
        # 添加节点特征
        for node_type, features in graph.node_features.items():
            hetero_graph.nodes[node_type].data['feat'] = torch.tensor(
                features, dtype=torch.float32
            )
        
        # 添加边特征
        for edge_type, features in graph.edge_features.items():
            etype_name = edge_type[1]  # 边类型名称
            hetero_graph.edges[etype_name].data['feat'] = torch.tensor(
                features, dtype=torch.float32
            )
        
        return hetero_graph
    
    # ================================================================
    #                  元路径相关方法
    # ================================================================
    
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
    
    def compute_metapath_adjacency(
        self, 
        graph: HeteroGraphData,
        metapath: List[Tuple[str, str, str]]
    ) -> np.ndarray:
        """
        计算元路径邻接矩阵
        
        【原理】
        元路径邻接矩阵 = A1 @ A2 @ ... @ An
        其中Ai是元路径中第i条边的邻接矩阵
        
        例如U-S-U元路径：
        A_USU = A_US @ A_SU
        
        结果矩阵[i,j]=k表示用户i和用户j之间有k条U-S-U路径
        
        Args:
            graph: 异质图数据
            metapath: 元路径
            
        Returns:
            元路径邻接矩阵
        """
        adj = None
        
        for edge_type in metapath:
            if edge_type not in graph.edge_index:
                # 检查是否有反向边
                reverse_type = (edge_type[2], 'rev_' + edge_type[1], edge_type[0])
                if reverse_type in graph.edge_index:
                    edge_type = reverse_type
                else:
                    raise ValueError(f"边类型 {edge_type} 不存在于图中")
            
            src, dst = graph.edge_index[edge_type]
            src_type, _, dst_type = edge_type
            
            # 构建稀疏邻接矩阵
            num_src = graph.num_nodes[src_type]
            num_dst = graph.num_nodes[dst_type]
            
            # 创建邻接矩阵
            edge_adj = np.zeros((num_src, num_dst), dtype=np.float32)
            for s, d in zip(src, dst):
                edge_adj[s, d] = 1.0
            
            # 矩阵乘法
            if adj is None:
                adj = edge_adj
            else:
                adj = adj @ edge_adj
        
        return adj
    
    # ================================================================
    #                  调试和可视化
    # ================================================================
    
    def print_graph_summary(self, graph: HeteroGraphData):
        """
        打印图的摘要信息
        
        Args:
            graph: 异质图数据
        """
        print("=" * 60)
        print("异质图摘要")
        print("=" * 60)
        
        print("\n【节点信息】")
        for node_type, count in graph.num_nodes.items():
            feat_dim = graph.node_features[node_type].shape[1]
            print(f"  {node_type}: {count} 个节点, 特征维度 = {feat_dim}")
        
        print("\n【边信息】")
        for edge_type, (src, dst) in graph.edge_index.items():
            num_edges = src.shape[0]
            src_type, rel, dst_type = edge_type
            print(f"  {src_type} --[{rel}]--> {dst_type}: {num_edges} 条边")
            
            if edge_type in graph.edge_features:
                feat_dim = graph.edge_features[edge_type].shape[1]
                print(f"    边特征维度 = {feat_dim}")
        
        print("\n【元路径】")
        for mp in self.get_metapaths():
            path_str = " -> ".join([f"({e[0]})-[{e[1]}]->({e[2]})" for e in mp])
            print(f"  {path_str}")
        
        print("=" * 60)
