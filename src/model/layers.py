"""
基础神经网络层
==============

本模块包含构建HAN和Actor-Critic网络所需的基础层。

【层的设计原则】
1. 模块化：每个层独立可复用
2. 灵活性：支持不同的配置参数
3. 可解释：提供注意力权重等中间结果

【主要组件】
- MLP: 多层感知机，用于特征变换
- GraphAttentionLayer: 图注意力层(GAT)，用于同质图
- HeterogeneousAttentionLayer: 异质注意力层，用于特定元路径
- SemanticAttention: 语义注意力，用于聚合多条元路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)
    
    【结构】
    Input → Linear → Activation → Dropout → Linear → ... → Output
    
    【用途】
    - 特征变换/投影
    - Actor/Critic的输出头
    - 节点/边特征编码
    
    【示例】
    ```python
    mlp = MLP(
        input_dim=64,
        hidden_dims=[128, 64],
        output_dim=32,
        activation='relu',
        dropout=0.1
    )
    output = mlp(input_tensor)  # shape: (batch, 32)
    ```
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        output_activation: bool = False
    ):
        """
        初始化MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如[128, 64]
            output_dim: 输出维度
            activation: 激活函数 ('relu', 'gelu', 'tanh', 'leaky_relu')
            dropout: Dropout比例
            use_layer_norm: 是否使用LayerNorm
            output_activation: 输出层是否使用激活函数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 选择激活函数
        self.activation = self._get_activation(activation)
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(self.activation)
        
        self.net = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),  # Swish
        }
        if name not in activations:
            raise ValueError(f"未知激活函数: {name}")
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape = (batch_size, input_dim) 或 (batch_size, num_nodes, input_dim)
            
        Returns:
            输出张量, shape = (batch_size, output_dim) 或 (batch_size, num_nodes, output_dim)
        """
        return self.net(x)


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer, GAT)
    
    【原理】
    GAT使用注意力机制来聚合邻居节点的特征：
    
    h_i' = σ( Σ_{j∈N(i)} α_ij · W · h_j )
    
    其中注意力系数 α_ij 由以下公式计算：
    
    e_ij = LeakyReLU(a^T · [W·h_i || W·h_j])
    α_ij = softmax_j(e_ij)
    
    【与传统GCN的区别】
    - GCN: 使用固定权重（基于度）
    - GAT: 使用学习的注意力权重
    
    【多头注意力】
    使用K个注意力头，然后拼接或平均：
    - 拼接: h_i' = ||_{k=1}^K σ( Σ α_ij^k · W^k · h_j )
    - 平均: h_i' = 1/K Σ_{k=1}^K σ( Σ α_ij^k · W^k · h_j )
    
    【参考】
    "Graph Attention Networks" (Veličković et al., ICLR 2018)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        初始化图注意力层
        
        Args:
            in_features: 输入特征维度
            out_features: 每个注意力头的输出维度
            num_heads: 注意力头数量
            dropout: Dropout比例
            alpha: LeakyReLU的负斜率
            concat: True则拼接多头输出，False则平均
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # 线性变换 W
        # 形状: (num_heads, in_features, out_features)
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        
        # 注意力向量 a = [a_src || a_dst]
        # 分开存储源节点和目标节点的注意力参数
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        
        # 激活函数和正则化
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 节点特征, shape = (num_nodes, in_features)
            edge_index: 边索引(COO格式), shape = (2, num_edges)
                       edge_index[0] = 源节点, edge_index[1] = 目标节点
            return_attention: 是否返回注意力权重
            
        Returns:
            - 更新后的节点特征, shape = (num_nodes, num_heads * out_features) 或 (num_nodes, out_features)
            - 注意力权重 (可选), shape = (num_edges, num_heads)
        """
        num_nodes = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # 1. 线性变换: (num_nodes, in_features) -> (num_heads, num_nodes, out_features)
        # 使用einsum: h_i' = W · h_i
        h = torch.einsum('ni,hio->hno', x, self.W)  # (H, N, out)
        
        # 2. 计算注意力分数
        # e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
        
        # 源节点分数: (num_heads, num_nodes, 1)
        attn_src = torch.einsum('hno,hok->hnk', h, self.a_src).squeeze(-1)  # (H, N)
        
        # 目标节点分数
        attn_dst = torch.einsum('hno,hok->hnk', h, self.a_dst).squeeze(-1)  # (H, N)
        
        # 边的注意力分数: e_ij = a_src · h_i + a_dst · h_j
        # 形状: (num_heads, num_edges)
        e = attn_src[:, src_idx] + attn_dst[:, dst_idx]  # (H, E)
        e = self.leaky_relu(e)
        
        # 3. Softmax归一化 (按目标节点分组)
        # 使用scatter实现按节点的softmax
        alpha = self._softmax_per_node(e, dst_idx, num_nodes)  # (H, E)
        alpha = self.dropout(alpha)
        
        # 4. 聚合邻居特征
        # h_i' = Σ_j α_ij · h_j
        h_prime = self._aggregate(h, alpha, src_idx, dst_idx, num_nodes)  # (H, N, out)
        
        # 5. 多头输出处理
        if self.concat:
            # 拼接: (N, H * out)
            out = h_prime.permute(1, 0, 2).reshape(num_nodes, -1)
        else:
            # 平均: (N, out)
            out = h_prime.mean(dim=0)
        
        if return_attention:
            return out, alpha.permute(1, 0)  # (E, H)
        return out, None
    
    def _softmax_per_node(
        self,
        e: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        按目标节点计算softmax
        
        对于每个目标节点i，计算其所有入边的softmax：
        α_ji = exp(e_ji) / Σ_k exp(e_ki)
        """
        # 数值稳定性：减去最大值
        e_max = torch.zeros(e.size(0), num_nodes, device=e.device)
        e_max.scatter_reduce_(1, dst_idx.unsqueeze(0).expand(e.size(0), -1), 
                              e, reduce='amax', include_self=False)
        e = e - e_max[:, dst_idx]
        
        # 计算exp
        exp_e = torch.exp(e)
        
        # 按目标节点求和
        sum_exp = torch.zeros(e.size(0), num_nodes, device=e.device)
        sum_exp.scatter_add_(1, dst_idx.unsqueeze(0).expand(e.size(0), -1), exp_e)
        
        # 归一化
        alpha = exp_e / (sum_exp[:, dst_idx] + 1e-10)
        
        return alpha
    
    def _aggregate(
        self,
        h: torch.Tensor,
        alpha: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        加权聚合邻居特征
        
        h_i' = Σ_{j∈N(i)} α_ij · h_j
        """
        num_heads, _, out_features = h.shape
        
        # 获取源节点特征: (H, E, out)
        h_src = h[:, src_idx, :]
        
        # 加权: (H, E, out) * (H, E, 1)
        weighted = h_src * alpha.unsqueeze(-1)
        
        # 按目标节点聚合
        h_prime = torch.zeros(num_heads, num_nodes, out_features, device=h.device)
        dst_idx_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).expand(num_heads, -1, out_features)
        h_prime.scatter_add_(1, dst_idx_expanded, weighted)
        
        return h_prime


class HeterogeneousAttentionLayer(nn.Module):
    """
    异质图注意力层
    
    【与GAT的区别】
    - GAT: 处理同质图，所有节点/边类型相同
    - 本层: 处理特定边类型，源和目标节点类型可以不同
    
    【工作原理】
    对于边类型 (src_type, edge_type, dst_type):
    1. 将源节点特征投影到公共空间
    2. 将目标节点特征投影到公共空间
    3. 计算跨类型的注意力
    4. 聚合源节点特征到目标节点
    
    【应用场景】
    - User -> Satellite 链路
    - Satellite -> Satellite 链路(ISL)
    """
    
    def __init__(
        self,
        src_dim: int,
        dst_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_dim: int = 0
    ):
        """
        初始化异质注意力层
        
        Args:
            src_dim: 源节点特征维度
            dst_dim: 目标节点特征维度
            out_dim: 输出特征维度（每个头）
            num_heads: 注意力头数
            dropout: Dropout比例
            use_edge_features: 是否使用边特征
            edge_dim: 边特征维度
        """
        super().__init__()
        
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        
        # 源节点投影: src_dim -> out_dim * num_heads
        self.W_src = nn.Linear(src_dim, out_dim * num_heads, bias=False)
        
        # 目标节点投影: dst_dim -> out_dim * num_heads
        self.W_dst = nn.Linear(dst_dim, out_dim * num_heads, bias=False)
        
        # 边特征投影（可选）
        if use_edge_features and edge_dim > 0:
            self.W_edge = nn.Linear(edge_dim, out_dim * num_heads, bias=False)
        else:
            self.W_edge = None
        
        # 注意力参数
        # 使用双线性注意力: e_ij = (W_src · h_i)^T · A · (W_dst · h_j)
        # 简化为: e_ij = a^T · [W_src · h_i || W_dst · h_j || W_edge · e_ij]
        attn_in_dim = out_dim * 2
        if use_edge_features and edge_dim > 0:
            attn_in_dim += out_dim
        self.attn = nn.Parameter(torch.empty(num_heads, attn_in_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attn)
    
    def forward(
        self,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        edge_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src_feat: 源节点特征, (num_src, src_dim)
            dst_feat: 目标节点特征, (num_dst, dst_dim)
            edge_index: (src_indices, dst_indices), 每个形状 (num_edges,)
            edge_feat: 边特征 (可选), (num_edges, edge_dim)
            
        Returns:
            更新后的目标节点特征, (num_dst, out_dim * num_heads)
        """
        src_idx, dst_idx = edge_index
        num_dst = dst_feat.size(0)
        num_edges = src_idx.size(0)
        
        # 1. 特征投影
        # (num_src, out_dim * H) -> (num_src, H, out_dim)
        h_src = self.W_src(src_feat).view(-1, self.num_heads, self.out_dim)
        h_dst = self.W_dst(dst_feat).view(-1, self.num_heads, self.out_dim)
        
        # 2. 获取边两端的特征
        # (num_edges, H, out_dim)
        h_src_edge = h_src[src_idx]
        h_dst_edge = h_dst[dst_idx]
        
        # 3. 构建注意力输入
        # 拼接源节点、目标节点（和边特征）
        if self.use_edge_features and edge_feat is not None and self.W_edge is not None:
            h_edge = self.W_edge(edge_feat).view(-1, self.num_heads, self.out_dim)
            attn_input = torch.cat([h_src_edge, h_dst_edge, h_edge], dim=-1)
        else:
            attn_input = torch.cat([h_src_edge, h_dst_edge], dim=-1)
        
        # (num_edges, H, attn_in_dim) -> (H, num_edges, attn_in_dim)
        attn_input = attn_input.permute(1, 0, 2)
        
        # 4. 计算注意力分数
        # e = attn^T · attn_input -> (H, num_edges, 1) -> (H, num_edges)
        e = torch.bmm(attn_input, self.attn).squeeze(-1)
        e = self.leaky_relu(e)
        
        # 5. Softmax归一化
        alpha = self._softmax_per_node(e, dst_idx, num_dst)
        alpha = self.dropout(alpha)
        
        # 6. 聚合
        # (H, num_edges) * (H, num_edges, out_dim) -> sum by dst
        h_src_edge_t = h_src_edge.permute(1, 0, 2)  # (H, E, out)
        weighted = h_src_edge_t * alpha.unsqueeze(-1)
        
        # 按目标节点聚合
        out = torch.zeros(self.num_heads, num_dst, self.out_dim, device=src_feat.device)
        dst_idx_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).expand(
            self.num_heads, -1, self.out_dim
        )
        out.scatter_add_(1, dst_idx_expanded, weighted)
        
        # (H, num_dst, out) -> (num_dst, H * out)
        out = out.permute(1, 0, 2).reshape(num_dst, -1)
        
        return out
    
    def _softmax_per_node(self, e, dst_idx, num_nodes):
        """按目标节点计算softmax"""
        # 数值稳定
        e_max = torch.zeros(e.size(0), num_nodes, device=e.device)
        e_max.scatter_reduce_(1, dst_idx.unsqueeze(0).expand(e.size(0), -1),
                              e, reduce='amax', include_self=False)
        e = e - e_max[:, dst_idx]
        
        exp_e = torch.exp(e)
        sum_exp = torch.zeros(e.size(0), num_nodes, device=e.device)
        sum_exp.scatter_add_(1, dst_idx.unsqueeze(0).expand(e.size(0), -1), exp_e)
        
        return exp_e / (sum_exp[:, dst_idx] + 1e-10)


class SemanticAttention(nn.Module):
    """
    语义注意力层 (Semantic-level Attention)
    
    【在HAN中的作用】
    HAN使用两级注意力：
    1. 节点级注意力：聚合同一元路径下的邻居
    2. 语义级注意力：聚合不同元路径的表示
    
    本层实现第2级，即语义级注意力。
    
    【原理】
    给定K条元路径产生的K个节点嵌入 {z_i^1, z_i^2, ..., z_i^K}
    最终嵌入为加权和：
    
    z_i = Σ_k β_k · z_i^k
    
    其中 β_k 是第k条元路径的重要性权重：
    
    w_k = 1/N Σ_i (q^T · tanh(W · z_i^k + b))
    β_k = softmax(w_k)
    
    【特点】
    - 全局权重：β_k 对所有节点相同
    - 可解释性：β_k 表示不同语义关系的重要性
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128
    ):
        """
        初始化语义注意力层
        
        Args:
            in_dim: 输入嵌入维度（每条元路径的输出维度）
            hidden_dim: 注意力隐藏维度
        """
        super().__init__()
        
        # 注意力网络: z -> score
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(
        self,
        z_list: List[torch.Tensor],
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            z_list: 元路径嵌入列表, 每个形状 (num_nodes, in_dim)
            return_weights: 是否返回注意力权重
            
        Returns:
            - 聚合后的嵌入, (num_nodes, in_dim)
            - 语义注意力权重 (可选), (num_metapaths,)
        """
        # 堆叠: (K, N, D)
        z_stack = torch.stack(z_list, dim=0)
        num_metapaths, num_nodes, embed_dim = z_stack.shape
        
        # 计算每条元路径的平均注意力分数
        # (K, N, D) -> (K, N, 1) -> mean over N -> (K, 1)
        scores = self.attention(z_stack).mean(dim=1)  # (K, 1)
        
        # Softmax得到权重
        beta = F.softmax(scores, dim=0)  # (K, 1)
        
        # 加权求和
        # (K, 1, 1) * (K, N, D) -> sum -> (N, D)
        z_final = (beta.unsqueeze(-1) * z_stack).sum(dim=0)
        
        if return_weights:
            return z_final, beta.squeeze(-1)
        return z_final, None
