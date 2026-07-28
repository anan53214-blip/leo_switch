"""
特征提取模块
============

本模块负责从环境状态中提取节点和边的特征向量。
这些特征将作为图神经网络(GNN)的输入。

【特征设计原则】
1. 归一化：所有特征归一化到相近范围，便于神经网络训练
2. 信息丰富：包含决策所需的关键信息
3. 时空相关：考虑位置、运动、时间等因素

【节点特征】
- 卫星节点：位置、速度、MEC负载、连接用户数等
- 用户节点：位置、任务信息、连接状态、RVT等

【边特征】
- 用户-卫星边：距离、仰角、SNR、传输速率、RVT等
- 星间链路边：距离、传播时延等
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class NodeFeatures:
    """
    节点特征数据类
    
    存储从环境中提取的节点特征矩阵
    
    【数据结构说明】
    - satellite_features: shape = (num_satellites, sat_feat_dim)
      每一行是一颗卫星的特征向量
    - user_features: shape = (num_users, user_feat_dim)  
      每一行是一个用户的特征向量
    """
    
    # 卫星节点特征矩阵
    satellite_features: np.ndarray = None  # (num_sats, sat_feat_dim)
    
    # 用户节点特征矩阵
    user_features: np.ndarray = None       # (num_users, user_feat_dim)
    
    # 特征维度（用于验证）
    satellite_feature_dim: int = 10
    user_feature_dim: int = 13


@dataclass
class EdgeFeatures:
    """
    边特征数据类

    存储从环境中提取的边特征

    【边的表示方式】
    边用 (源节点索引, 目标节点索引) 的列表表示
    对应的特征存储在特征矩阵中

    例如：
    user_satellite_edges = [(0, 5), (0, 10), (1, 5), ...]
    表示用户0连接卫星5和10，用户1连接卫星5，...
    """

    # 用户-卫星可见边 (User-Satellite Link, visible)
    # ---------------------------------------
    # 边索引：[(user_id, sat_id), ...]
    user_satellite_edges: List[Tuple[int, int]] = field(default_factory=list)
    # 边特征矩阵：(num_edges, edge_feat_dim)
    user_satellite_features: np.ndarray = None

    # 用户-卫星服务边 (User-Satellite Link, serving)
    # ---------------------------------------
    serving_edges: List[Tuple[int, int]] = field(default_factory=list)
    serving_features: np.ndarray = None

    # 用户-用户邻居边 (User-User nearby)
    # ---------------------------------------
    nearby_user_edges: List[Tuple[int, int]] = field(default_factory=list)
    nearby_user_features: np.ndarray = None

    # 星间链路边 (Inter-Satellite Link)
    # ---------------------------------------
    # 边索引：[(sat_id_1, sat_id_2), ...]
    inter_satellite_edges: List[Tuple[int, int]] = field(default_factory=list)
    # 边特征矩阵
    inter_satellite_features: np.ndarray = None

    # 特征维度
    user_satellite_edge_dim: int = 5
    serving_edge_dim: int = 2
    nearby_user_edge_dim: int = 1
    inter_satellite_edge_dim: int = 3


class FeatureExtractor:
    """
    特征提取器
    
    从LEO卫星网络环境中提取节点和边的特征。
    
    【使用方式】
    ```python
    extractor = FeatureExtractor()
    node_features = extractor.extract_node_features(env)
    edge_features = extractor.extract_edge_features(env)
    ```
    
    【特征归一化策略】
    - 位置：除以最大值（如经度/180，纬度/90）
    - 距离：除以典型最大值（如2000km）
    - 角度：除以90度
    - 比例值：已经在[0,1]范围内
    - ID：除以总数
    """
    
    def __init__(
        self,
        normalize: bool = True,
        include_velocity: bool = True
    ):
        """
        初始化特征提取器
        
        Args:
            normalize: 是否归一化特征（强烈建议True）
            include_velocity: 是否包含速度特征
        """
        self.normalize = normalize
        self.include_velocity = include_velocity
        
        # ---------- 归一化参数 ----------
        # 这些参数用于将原始值映射到[0,1]或[-1,1]范围
        
        # 位置归一化
        self.max_latitude = 90.0           # 最大纬度
        self.max_longitude = 180.0         # 最大经度
        self.max_altitude = 600.0          # 最大高度(km)，LEO卫星约550km
        
        # 距离归一化
        self.max_distance = 2500.0         # 最大星地距离(km)
        
        # 速度归一化
        self.max_velocity = 8.0            # 最大轨道速度(km/s)
        
        # 其他归一化
        self.max_rvt = 600.0               # 最大RVT(秒)，约10分钟
        self.max_snr = 50.0                # 最大SNR(dB)
        self.max_data_rate = 500.0         # 最大传输速率(Mbps)
        self.max_queue_length = 100        # 最大队列长度
        self.max_users_per_sat = 50        # 每卫星最大用户数
        
        # 任务参数归一化
        self.max_data_size = 50e6          # 最大数据量(bits)
        self.max_computation = 10e9        # 最大计算量(cycles)
        self.max_delay = 10.0              # 最大时延要求(s)
        self._isl_topology_cache: Dict[
            Tuple[int, int],
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]
        ] = {}

    def _position_norm_factor(self, env) -> float:
        constellation = getattr(env, 'constellation', None)
        for attr in ('semi_major_axis', '_a'):
            value = getattr(constellation, attr, None)
            if value is not None:
                return max(float(value), 1e-6)

        all_pos = getattr(constellation, '_all_pos_ecef', None)
        if all_pos is not None and len(all_pos) > 0:
            return max(float(np.linalg.norm(all_pos, axis=1).max()), 1e-6)

        return 7000.0
    
    # ================================================================
    #                       节点特征提取
    # ================================================================
    
    def extract_node_features(self, env) -> NodeFeatures:
        """
        提取所有节点的特征
        
        Args:
            env: LEOSatelliteEnv 环境实例
            
        Returns:
            NodeFeatures 包含卫星和用户特征矩阵
        """
        # 提取卫星特征
        sat_features = self._extract_satellite_features(env)
        
        # 提取用户特征
        user_features = self._extract_user_features(env)
        
        return NodeFeatures(
            satellite_features=sat_features,
            user_features=user_features,
            satellite_feature_dim=sat_features.shape[1],
            user_feature_dim=user_features.shape[1]
        )
    
    def _extract_satellite_features(self, env) -> np.ndarray:
        """
        提取卫星节点特征
        
        【卫星特征向量组成】(共10维)
        ┌─────────────────────────────────────────────────────────┐
        │  索引  │  特征名称      │  物理含义                      │
        ├─────────────────────────────────────────────────────────┤
        │  0-2   │  position      │  ECEF位置(x,y,z)归一化         │
        │  3-5   │  velocity      │  速度(vx,vy,vz)归一化          │
        │  6     │  cpu_util      │  CPU利用率 [0,1]               │
        │  7     │  queue_len     │  任务队列长度(归一化)           │
        │  8     │  num_users     │  已连接用户数(归一化)           │
        │  9     │  avail_freq    │  可用计算资源(归一化)           │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            env: 环境实例
            
        Returns:
            特征矩阵 shape=(num_satellites, 10)
        """
        num_sats = env.num_satellites
        feat_dim = 10 if self.include_velocity else 7
        features = np.zeros((num_sats, feat_dim), dtype=np.float32)

        # 向量化位置特征
        features[:, 0:3] = (
            env.constellation._all_pos_ecef / self._position_norm_factor(env)
        )
        
        idx = 3
        if self.include_velocity:
            features[:, 3:6] = env.constellation._all_vel_eci / self.max_velocity
            idx = 6
        
        # MEC状态特征（需要逐个访问server对象，但计算量小）
        for sat_id in range(num_sats):
            server = env.mec_manager.get_server(sat_id)
            if server:
                features[sat_id, idx] = server.utilization
                features[sat_id, idx+1] = server.queue_length / self.max_queue_length
                features[sat_id, idx+2] = len(server.connected_users) / self.max_users_per_sat
                features[sat_id, idx+3] = server.available_freq_ghz / server.config.satellite_max_cpu_freq_ghz
        
        return features
    
    def _extract_user_features(self, env) -> np.ndarray:
        """
        提取用户节点特征
        
        【用户特征向量组成】(共13维)
        ┌─────────────────────────────────────────────────────────┐
        │  索引  │  特征名称          │  物理含义                  │
        ├─────────────────────────────────────────────────────────┤
        │  0-2   │  position          │  ECEF位置(x,y,z)归一化     │
        │  3     │  state             │  连接状态(编码)             │
        │  4     │  serving_sat       │  服务卫星ID(归一化)         │
        │  5     │  connection_time   │  已连接时长(归一化)         │
        │  6     │  task_data         │  当前任务数据量             │
        │  7     │  task_compute      │  当前任务计算量             │
        │  8     │  task_deadline     │  当前任务时延要求           │
        │  9     │  task_type         │  任务类型(编码)             │
        │  10    │  handover_count    │  切换次数(归一化)           │
        │  11    │  service_quality   │  服务质量指标               │
        │  12    │  rvt_warning       │  RVT预警信号(0/1)          │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            env: 环境实例
            
        Returns:
            特征矩阵 shape=(num_users, 13)
        """
        num_users = env.num_users
        feat_dim = 13
        features = np.zeros((num_users, feat_dim), dtype=np.float32)
        
        for user_id, user in enumerate(env.user_manager.users):
            idx = 0
            
            # ------ 1. 位置特征 (3维) ------
            pos_ecef = env._user_pos_ecef[user_id]
            norm_factor = self._position_norm_factor(env)
            features[user_id, idx:idx+3] = pos_ecef / norm_factor
            idx += 3
            
            # ------ 2. 连接状态 (1维) ------
            # UserState: IDLE=0, CONNECTED=1, HANDOVER=2, BLOCKED=3
            # 归一化到[0, 1]
            features[user_id, idx] = user.state.value / 3.0
            idx += 1
            
            # ------ 3. 服务卫星 (1维) ------
            # -1表示未连接，归一化到[-1/N, 1]
            if user.serving_satellite >= 0:
                features[user_id, idx] = user.serving_satellite / env.num_satellites
            else:
                features[user_id, idx] = -0.1  # 特殊值表示未连接
            idx += 1
            
            # ------ 4. 连接时长 (1维) ------
            # 使用服务时间作为连接时长的度量
            max_connection_time = 600.0  # 假设最大10分钟
            features[user_id, idx] = min(user.total_service_time, max_connection_time) / max_connection_time
            idx += 1
            
            # ------ 5. 当前任务信息 (4维) ------
            task = env.user_tasks.get(user_id)
            if task is not None:
                # 任务数据量
                features[user_id, idx] = task.data_size / self.max_data_size
                idx += 1
                # 任务计算量
                features[user_id, idx] = task.computation / self.max_computation
                idx += 1
                # 时延要求
                features[user_id, idx] = task.max_delay / self.max_delay
                idx += 1
                # 任务类型 (LIGHT=0, MEDIUM=1, HEAVY=2)
                features[user_id, idx] = task.task_type.value / 2.0
                idx += 1
            else:
                # 无任务时填充0
                idx += 4
            
            # ------ 6. 历史统计 (2维) ------
            # 切换次数
            max_handovers = 20
            features[user_id, idx] = min(user.handover_count, max_handovers) / max_handovers
            idx += 1
            
            # 服务质量：成功切换率
            if user.handover_count > 0:
                features[user_id, idx] = user.successful_handovers / user.handover_count
            else:
                features[user_id, idx] = 1.0  # 无切换时认为质量完美
            idx += 1
            
            # ------ 7. RVT预警信号 (1维) ------
            rvt_threshold = getattr(env.config, 'rvt_threshold_sec', 60.0)
            if user.serving_satellite >= 0:
                vis_info = env._get_satellite_visibility(user, user.serving_satellite)
                if vis_info is not None and vis_info.is_visible:
                    features[user_id, idx] = 1.0 if vis_info.rvt_seconds < rvt_threshold else 0.0
                else:
                    features[user_id, idx] = 1.0  # 不可见则标记预警
            else:
                features[user_id, idx] = 1.0  # 未连接则标记预警
            idx += 1
        
        return features
    
    # ================================================================
    #                       边特征提取
    # ================================================================
    
    def extract_edge_features(self, env) -> EdgeFeatures:
        """
        提取所有边的特征

        【边类型说明】
        1. 用户-卫星可见边 (User-Satellite Link, visible)
           - 连接条件：卫星对用户可见（仰角 > 阈值）
           - 特征：距离、仰角、SNR、传输速率、RVT、是否当前服务

        2. 用户-卫星服务边 (User-Satellite Link, serving)
           - 连接条件：用户当前连接的卫星
           - 特征：is_serving, service_time

        3. 用户-用户邻居边 (User-User nearby)
           - 连接条件：地理距离 < 500km
           - 特征：distance_normalized

        4. 星间链路边 (Inter-Satellite Link, ISL)
           - 连接条件：同轨道相邻 或 跨轨道相邻
           - 特征：距离、传播时延、链路状态

        Args:
            env: 环境实例

        Returns:
            EdgeFeatures 包含边索引和特征
        """
        # 提取用户-卫星可见边
        us_edges, us_features = self._extract_user_satellite_edges(env)

        # 提取用户-卫星服务边
        serving_edges, serving_features = self._extract_serving_edges(env)

        # 提取用户-用户邻居边
        nearby_edges, nearby_features = self._extract_nearby_user_edges(env)

        # 提取星间链路边
        isl_edges, isl_features = self._extract_inter_satellite_edges(env)

        return EdgeFeatures(
            user_satellite_edges=us_edges,
            user_satellite_features=us_features,
            serving_edges=serving_edges,
            serving_features=serving_features,
            nearby_user_edges=nearby_edges,
            nearby_user_features=nearby_features,
            inter_satellite_edges=isl_edges,
            inter_satellite_features=isl_features,
            user_satellite_edge_dim=us_features.shape[1] if len(us_edges) > 0 else 5,
            serving_edge_dim=serving_features.shape[1] if len(serving_edges) > 0 else 2,
            nearby_user_edge_dim=nearby_features.shape[1] if len(nearby_edges) > 0 else 1,
            inter_satellite_edge_dim=isl_features.shape[1] if len(isl_edges) > 0 else 3
        )
    
    def _extract_user_satellite_edges(self, env) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        提取用户-卫星边及其特征
        
        【用户-卫星边特征】(共5维)
        ┌─────────────────────────────────────────────────────────┐
        │  索引  │  特征名称       │  物理含义                     │
        ├─────────────────────────────────────────────────────────┤
        │  0     │  distance       │  星地距离(归一化)              │
        │  1     │  elevation      │  仰角(归一化到[0,1])           │
        │  2     │  snr            │  信噪比(归一化)                │
        │  3     │  data_rate      │  可达传输速率(归一化)          │
        │  4     │  rvt            │  剩余可见时间(归一化)          │
        └─────────────────────────────────────────────────────────┘
        
        Returns:
            (边列表, 特征矩阵)
        """
        edges = []
        features_list = []
        
        for user in env.user_manager.users:
            user_id = user.user_id
            
            # 获取该用户的可见卫星
            visible_sats = env._get_visible_satellites(user)
            
            for sat_info in visible_sats:
                sat_id = sat_info.sat_id
                
                # 添加边 (user_id, sat_id)
                edges.append((user_id, sat_id))
                
                # 提取边特征
                edge_feat = np.zeros(5, dtype=np.float32)

                # 距离（归一化）
                edge_feat[0] = sat_info.distance_km / self.max_distance

                # 仰角（归一化到[0,1]，因为已经过滤了负仰角）
                edge_feat[1] = sat_info.elevation_deg / 90.0

                # SNR（从信道模型计算）
                snr_db = env.channel.compute_snr_db(
                    sat_info.distance_km,
                    sat_info.elevation_deg,
                    'uplink'
                )
                edge_feat[2] = np.clip(snr_db / self.max_snr, -1, 1)

                # 传输速率
                data_rate = env.channel.compute_data_rate_mbps(
                    sat_info.distance_km,
                    sat_info.elevation_deg,
                    'uplink'
                )
                edge_feat[3] = data_rate / self.max_data_rate

                # 剩余可见时间(RVT)
                edge_feat[4] = sat_info.rvt_seconds / self.max_rvt

                features_list.append(edge_feat)
        
        # 转换为numpy数组
        if features_list:
            features = np.stack(features_list, axis=0)
        else:
            features = np.zeros((0, 5), dtype=np.float32)

        return edges, features

    def _extract_serving_edges(self, env) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        提取用户-卫星服务边

        只包含用户当前连接的卫星（每个用户最多一条边）。
        特征：[is_serving=1.0, service_time_normalized]
        """
        edges = []
        features = []
        for user in env.user_manager.users:
            if user.serving_satellite >= 0:
                edges.append((user.user_id, user.serving_satellite))
                service_time = getattr(user, 'total_service_time', 0.0)
                features.append(np.array([1.0, min(float(service_time), 600.0) / 600.0], dtype=np.float32))
        return edges, np.stack(features, axis=0) if features else np.zeros((0, 2), dtype=np.float32)

    def _extract_nearby_user_edges(self, env) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        提取用户-用户邻居边

        连接地理距离在 500km 内的用户对。
        特征：[distance_normalized]
        """
        edges = []
        features = []
        max_distance_km = 500.0
        positions = env._user_pos_ecef
        for i in range(env.num_users):
            for j in range(env.num_users):
                if i == j:
                    continue
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance <= max_distance_km:
                    edges.append((i, j))
                    features.append(np.array([distance / max_distance_km], dtype=np.float32))
        return edges, np.stack(features, axis=0) if features else np.zeros((0, 1), dtype=np.float32)

    def _extract_inter_satellite_edges(self, env) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        提取星间链路(ISL)边及其特征
        
        【星间链路拓扑】
        Walker星座中，每颗卫星有4条ISL：
        - 2条同轨道链路：前后相邻卫星
        - 2条跨轨道链路：左右相邻轨道的最近卫星
        
        【ISL边特征】(共3维)
        ┌─────────────────────────────────────────────────────────┐
        │  索引  │  特征名称       │  物理含义                     │
        ├─────────────────────────────────────────────────────────┤
        │  0     │  distance       │  卫星间距离(归一化)           │
        │  1     │  prop_delay     │  传播时延(归一化)             │
        │  2     │  link_type      │  链路类型(同轨=0, 跨轨=1)     │
        └─────────────────────────────────────────────────────────┘
        
        Returns:
            (边列表, 特征矩阵)
        """
        num_planes = env.constellation.num_planes
        sats_per_plane = env.constellation.sats_per_plane
        all_pos = env.constellation._all_pos_ecef  # (N, 3)
        topology_key = (num_planes, sats_per_plane)
        cached_topology = self._isl_topology_cache.get(topology_key)
        if cached_topology is not None:
            src_arr, dst_arr, link_type_arr, edges = cached_topology
            diffs = all_pos[src_arr] - all_pos[dst_arr]  # (E, 3)
            distances = np.linalg.norm(diffs, axis=1)     # (E,)
            features = np.column_stack([
                distances / 5000.0,
                distances / 300.0 / 20.0,
                link_type_arr
            ]).astype(np.float32)
            return edges, features
        
        # 预计算所有ISL边索引（拓扑固定，只需算一次距离）
        src_list = []
        dst_list = []
        link_types = []
        
        for sat_id in range(env.num_satellites):
            plane_id = sat_id // sats_per_plane
            sat_idx = sat_id % sats_per_plane
            
            # 同轨道前向邻居
            next_sat_id = plane_id * sats_per_plane + (sat_idx + 1) % sats_per_plane
            if sat_id < next_sat_id:
                src_list.append(sat_id)
                dst_list.append(next_sat_id)
                link_types.append(0.0)
            
            # 跨轨道邻居
            right_sat_id = ((plane_id + 1) % num_planes) * sats_per_plane + sat_idx
            if sat_id < right_sat_id:
                src_list.append(sat_id)
                dst_list.append(right_sat_id)
                link_types.append(1.0)
        
        if not src_list:
            return [], np.zeros((0, 3), dtype=np.float32)
        
        src_arr = np.array(src_list)
        dst_arr = np.array(dst_list)
        
        # 向量化距离计算
        diffs = all_pos[src_arr] - all_pos[dst_arr]  # (E, 3)
        distances = np.linalg.norm(diffs, axis=1)     # (E,)
        
        link_type_arr = np.asarray(link_types, dtype=np.float32)
        features = np.column_stack([
            distances / 5000.0,
            distances / 300.0 / 20.0,
            link_type_arr
        ]).astype(np.float32)
        
        edges = list(zip(src_list, dst_list))
        self._isl_topology_cache[topology_key] = (
            np.asarray(src_arr, dtype=np.int32),
            np.asarray(dst_arr, dtype=np.int32),
            link_type_arr,
            edges
        )
        return edges, features
    
    # ================================================================
    #                       辅助方法
    # ================================================================
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        获取各类特征的维度
        
        Returns:
            特征维度字典
        """
        return {
            'satellite_node': 10 if self.include_velocity else 7,
            'user_node': 13,
            'user_satellite_edge': 5,
            'inter_satellite_edge': 3
        }
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        获取特征名称（用于可视化和调试）
        
        Returns:
            特征名称字典
        """
        sat_names = ['pos_x', 'pos_y', 'pos_z']
        if self.include_velocity:
            sat_names += ['vel_x', 'vel_y', 'vel_z']
        sat_names += ['cpu_util', 'queue_len', 'num_users', 'avail_freq']
        
        return {
            'satellite_node': sat_names,
            'user_node': [
                'pos_x', 'pos_y', 'pos_z',
                'state', 'serving_sat', 'conn_time',
                'task_data', 'task_compute', 'task_deadline', 'task_type',
                'handover_count', 'service_quality', 'rvt_warning'
            ],
            'user_satellite_edge': [
                'distance', 'elevation', 'snr', 'data_rate', 'rvt'
            ],
            'inter_satellite_edge': [
                'distance', 'prop_delay', 'link_type'
            ]
        }
