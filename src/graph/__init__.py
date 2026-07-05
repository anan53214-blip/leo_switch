"""
异质图构建模块
==============

本模块实现论文中的异质图（Heterogeneous Graph）建模方法。

【什么是异质图？】
普通图中所有节点和边都是同一类型，而异质图包含多种类型的节点和边。
在星地融合网络中：
- 节点类型：卫星节点、用户节点
- 边类型：星间链路(ISL)、用户-卫星链路(UDL)

【为什么使用异质图？】
1. 不同类型实体有不同特征（卫星有轨道参数，用户有任务需求）
2. 不同类型关系有不同语义（ISL是稳定链路，UDL是动态链路）
3. 异质图注意力网络(HAN)可以学习不同关系的重要性

【参考论文】
付一阳等《星地融合网络中基于异质图表征的多智能体协作切换方法》
"""

from .builder import HeteroGraphBuilder
from .features import FeatureExtractor, NodeFeatures, EdgeFeatures

__all__ = [
    'HeteroGraphBuilder',
    'FeatureExtractor',
    'NodeFeatures',
    'EdgeFeatures',
]
