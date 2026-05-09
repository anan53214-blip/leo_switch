"""
Gymnasium强化学习环境
整合星座、用户、信道、MEC等模块，实现联合切换与卸载优化环境

主要功能：
1. 多智能体环境（每个用户是一个智能体）
2. 混合动作空间（离散切换 + 连续卸载比例）
3. 综合奖励函数（时延、能耗、切换成功率）

参考论文：
- 付一阳等《星地融合网络中基于异质图表征的多智能体协作切换方法》
- 宋晓勤等《基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法》
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .constellation import WalkerConstellation
from .visibility import VisibilityCalculator, VisibilityInfo
from .user import User, UserPosition, UserState, UserGenerator, UserManager
from .task import Task, TaskType, TaskGenerator, TaskManager, TaskConfig
from .channel import SatelliteChannel, ChannelConfig, MultiUserChannel
from .mec import MECServer, MECConfig, MECManager, OffloadingCalculator


@dataclass
class EnvConfig:
    """
    环境配置参数
    """
    # 星座参数
    num_planes: int = 6
    sats_per_plane: int = 11
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    
    # 用户参数
    num_users: int = 10
    user_center_lat: float = 39.9      # 北京
    user_center_lon: float = 116.4
    user_radius_deg: float = 5.0       # 扩大用户分布范围，增加可见卫星差异
    
    # 仿真参数
    time_step_sec: float = 1.0         # 时间步长 (秒)
    max_steps: int = 3600              # 最大步数 (1小时)
    
    # 可见性参数
    min_elevation_deg: float = 10.0    # More moderate visibility threshold
    
    # 切换参数
    handover_delay_sec: float = 0.6    # Moderate signaling cost
    rvt_threshold_sec: float = 60.0    # Encourage proactive switching
    
    # 任务参数
    task_arrival_prob: float = 0.45    # Avoid pathological queue saturation
    min_effective_offload_ratio: float = 0.05  # Treat tiny noisy offload actions as local execution
    task_arrival_seed_offset: int = 7919
    
    # 奖励权重（增大正向奖励系数，平衡奖惩信号）
    reward_delay_weight: float = 1.4
    reward_energy_weight: float = 0.4
    reward_handover_weight: float = 0.3
    reward_load_balance_weight: float = 0.1
    reward_qos_weight: float = 0.4
    reward_enqueue_bonus: float = 0.02
    reward_invalid_action_penalty: float = 0.5
    reward_blocked_penalty: float = 1.0
    reward_queue_full_penalty: float = 0.3
    reward_failed_handover_penalty: float = 0.6
    reward_deadline_penalty: float = 1.0
    reward_energy_reference: float = 10.0
    
    # 随机种子
    seed: Optional[int] = None


def summarize_env_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """Derive reliability and QoS metrics from raw environment counters."""
    resolved_tasks = int(stats.get('completed_tasks', 0) + stats.get('deadline_violations', 0))
    total_tasks = int(stats.get('total_tasks', 0))
    pending_tasks = max(total_tasks - resolved_tasks, 0)

    total_handovers = int(stats.get('total_handovers', 0))
    successful_handovers = int(stats.get('successful_handovers', 0))
    failed_handovers = int(stats.get('failed_handovers', 0))
    forced_disconnects = int(stats.get('forced_disconnects', 0))
    continuity_events = total_handovers + forced_disconnects

    total_user_seconds = float(stats.get('total_user_seconds', 0.0))
    blocked_user_seconds = float(stats.get('blocked_user_seconds', 0.0))
    handover_interruption_seconds = float(stats.get('handover_interruption_seconds', 0.0))
    service_interruption_seconds = float(
        stats.get(
            'service_interruption_seconds',
            blocked_user_seconds + handover_interruption_seconds,
        )
    )
    has_time_based_reliability = (
        total_user_seconds > 0.0 and (
            'blocked_user_seconds' in stats or
            'handover_interruption_seconds' in stats or
            'service_interruption_seconds' in stats
        )
    )
    legacy_continuity_rate = (
        1.0 - float(forced_disconnects) / max(continuity_events, 1)
        if continuity_events > 0 else 1.0
    )
    service_availability_rate = (
        max(0.0, 1.0 - blocked_user_seconds / total_user_seconds)
        if has_time_based_reliability else legacy_continuity_rate
    )
    service_continuity_rate = (
        max(0.0, 1.0 - service_interruption_seconds / total_user_seconds)
        if has_time_based_reliability else legacy_continuity_rate
    )

    avg_delay = float(stats.get('total_delay', 0.0)) / max(resolved_tasks, 1)
    completed_count = float(stats.get('completed_tasks', 0))
    deadline_count = float(stats.get('deadline_violations', 0))
    task_completion_rate = completed_count / max(resolved_tasks, 1)
    task_success_rate = completed_count / max(total_tasks, 1)
    task_failure_rate = deadline_count / max(total_tasks, 1)
    task_settlement_rate = float(resolved_tasks) / max(total_tasks, 1)
    task_resolution_rate = task_settlement_rate
    delay_score = 1.0 / (1.0 + max(avg_delay, 0.0))
    effective_latency_score = (
        delay_score *
        np.clip(service_continuity_rate, 0.0, 1.0) *
        np.clip(task_success_rate, 0.0, 1.0)
    )

    summary = stats.copy()
    summary.update({
        'resolved_tasks': resolved_tasks,
        'pending_tasks': pending_tasks,
        'handover_success_rate': (
            float(successful_handovers) / max(total_handovers, 1)
        ),
        'handover_failure_rate': (
            float(failed_handovers) / max(total_handovers, 1)
        ),
        'forced_termination_rate': (
            float(forced_disconnects) / max(continuity_events, 1)
        ),
        'service_availability_rate': service_availability_rate,
        # Continuity is modeled as uninterrupted service time ratio, which
        # penalizes both handover-induced interruptions and blocked periods.
        'service_continuity_rate': service_continuity_rate,
        'task_completion_rate': task_completion_rate,
        'task_success_rate': task_success_rate,
        'task_failure_rate': task_failure_rate,
        'task_settlement_rate': task_settlement_rate,
        'task_resolution_rate': task_resolution_rate,
        'pending_task_rate': (
            float(pending_tasks) / max(total_tasks, 1)
        ),
        'deadline_violation_rate': (
            float(stats.get('deadline_violations', 0)) / max(total_tasks, 1)
        ),
        'avg_delay': avg_delay,
        'effective_latency_score': effective_latency_score,
        'avg_load_balance_score': (
            float(stats.get('load_balance_sum', 0.0)) /
            max(int(stats.get('load_balance_samples', 0)), 1)
        ),
    })
    return summary


class LEOSatelliteEnv(gym.Env):
    """
    LEO卫星网络切换与任务卸载联合优化环境
    
    这是一个多智能体环境，每个用户是一个智能体。
    
    观测空间 (每个用户):
        - 用户位置 (3)
        - 用户状态 (1)
        - 当前服务卫星信息 (5)
        - 可见卫星信息 (K * 6)，K为最大可见卫星数
        - 当前任务信息 (4)
        
    动作空间 (每个用户):
        - 离散: 切换决策 (0=不切换, 1~K=切换到第k个可见卫星)
        - 连续: 卸载比例 λ ∈ [0, 1]
    
    奖励:
        综合考虑时延、能耗、切换成功率、QoS满足率
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        初始化环境
        
        Args:
            config: 环境配置
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # 随机数生成器
        self.rng = np.random.default_rng(self.config.seed)
        self.task_arrival_rng = self._make_task_arrival_rng(self.config.seed)
        
        # 初始化各模块
        self._init_constellation()
        self._init_users()
        self._init_channel()
        self._init_mec()
        self._init_task_generator()
        
        # 定义动作和观测空间
        self._define_spaces()
        
        # 状态变量
        self.current_step = 0
        self.current_time = 0.0
        self.episode_rewards = []
        self.pending_rewards: Dict[int, float] = {}
        self._offload_task_meta: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        # RVT估算用的仰角历史
        self._prev_elevations: Dict[tuple, float] = {}
        
        # 每步可见性缓存（step开始时清空，避免重复计算）
        self._visibility_cache_step = -1
        self._visibility_cache: Dict[int, List[VisibilityInfo]] = {}
        
        # 预计算用户ECEF位置和ENU基向量（用户不移动，只需算一次）
        self._precompute_user_geometry()
        
        # 统计信息
        self.stats = self._build_stats()
        self._last_load_balance_score = 1.0

    def _make_task_arrival_rng(self, seed: Optional[int]) -> np.random.Generator:
        """Build a task-arrival RNG that is independent of action outcomes."""
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(seed) + int(self.config.task_arrival_seed_offset))
    
    def _init_constellation(self):
        """初始化星座"""
        self.constellation = WalkerConstellation(
            num_planes=self.config.num_planes,
            sats_per_plane=self.config.sats_per_plane,
            altitude_km=self.config.altitude_km,
            inclination_deg=self.config.inclination_deg
        )
        self.num_satellites = self.constellation.total_sats
        
        # 可见性计算器
        self.visibility_calc = VisibilityCalculator(
            min_elevation_deg=self.config.min_elevation_deg
        )
    
    def _init_users(self):
        """初始化用户"""
        generator = UserGenerator(seed=self.config.seed)
        users = generator.generate_users_in_circle(
            center_lat=self.config.user_center_lat,
            center_lon=self.config.user_center_lon,
            radius_deg=self.config.user_radius_deg,
            num_users=self.config.num_users
        )
        self.user_manager = UserManager(users)
        self.num_users = self.config.num_users
    
    def _init_channel(self):
        """初始化信道"""
        self.channel = SatelliteChannel()
        self.multi_user_channel = MultiUserChannel()
    
    def _init_mec(self):
        """初始化MEC"""
        self.mec_manager = MECManager(num_satellites=self.num_satellites)
        self.offload_calc = OffloadingCalculator()
    
    def _init_task_generator(self):
        """初始化任务生成器"""
        self.task_generator = TaskGenerator(seed=self.config.seed)
        self.task_manager = TaskManager()
        
        # 当前每个用户的任务
        self.user_task_queues: Dict[int, List[Task]] = {
            i: [] for i in range(self.num_users)
        }
        self.user_tasks: Dict[int, Optional[Task]] = {
            i: None for i in range(self.num_users)
        }
    
    def _precompute_user_geometry(self):
        """预计算用户的ECEF位置和ENU基向量（用户不移动，只需算一次）"""
        N = self.num_users
        self._user_pos_ecef = np.zeros((N, 3), dtype=np.float64)
        self._user_e_up = np.zeros((N, 3), dtype=np.float64)
        self._user_e_east = np.zeros((N, 3), dtype=np.float64)
        self._user_e_north = np.zeros((N, 3), dtype=np.float64)
        
        for i, user in enumerate(self.user_manager.users):
            self._user_pos_ecef[i] = user.get_ecef_position()
            lat_rad = np.radians(user.position.latitude)
            lon_rad = np.radians(user.position.longitude)
            cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
            cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
            self._user_e_east[i] = [-sin_lon, cos_lon, 0]
            self._user_e_north[i] = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]
            self._user_e_up[i] = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]

    def _invalidate_visibility_cache(self):
        """使可见性缓存失效（每步调用一次）"""
        self._visibility_cache.clear()

    @staticmethod
    def _build_stats() -> Dict[str, float]:
        """Initialize environment statistics and reward breakdown terms."""
        return {
            'total_handovers': 0,
            'successful_handovers': 0,
            'failed_handovers': 0,
            'forced_disconnects': 0,
            'total_user_seconds': 0.0,
            'blocked_user_seconds': 0.0,
            'handover_interruption_seconds': 0.0,
            'service_interruption_seconds': 0.0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'deadline_violations': 0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'reward_delay': 0.0,
            'reward_energy': 0.0,
            'reward_qos': 0.0,
            'reward_handover': 0.0,
            'reward_load_balance': 0.0,
            'reward_enqueue': 0.0,
            'penalty_deadline': 0.0,
            'penalty_queue_full': 0.0,
            'penalty_invalid_action': 0.0,
            'penalty_blocked': 0.0,
            'penalty_failed_handover': 0.0,
            'penalty_handover_cost': 0.0,
            'load_balance_sum': 0.0,
            'load_balance_samples': 0,
        }

    def _record_reward_terms(self, **terms: float) -> None:
        """Accumulate signed reward terms into the environment statistics."""
        for key, value in terms.items():
            if key in self.stats:
                self.stats[key] += float(value)

    @staticmethod
    def _summarize_stats(stats: Dict[str, float]) -> Dict[str, float]:
        """Add derived QoS and reliability metrics on top of raw counters."""
        return summarize_env_stats(stats)

    def _compute_visibility_batch(self, user_id: int) -> List[VisibilityInfo]:
        """向量化计算单个用户对所有卫星的可见性"""
        # 获取所有卫星ECEF位置 (N_sat, 3)
        all_sat_pos = self.constellation._all_pos_ecef  # 直接引用，无拷贝
        
        # 用户位置和ENU基向量
        user_pos = self._user_pos_ecef[user_id]  # (3,)
        e_east = self._user_e_east[user_id]
        e_north = self._user_e_north[user_id]
        e_up = self._user_e_up[user_id]
        
        # 向量化计算: vec_to_sat (N_sat, 3)
        vec = all_sat_pos - user_pos[None, :]
        distances = np.linalg.norm(vec, axis=1)  # (N_sat,)
        
        # ENU投影 (向量化点积)
        east_comp = vec @ e_east
        north_comp = vec @ e_north
        up_comp = vec @ e_up
        
        # 仰角
        horiz_dist = np.sqrt(east_comp**2 + north_comp**2)
        elevations = np.degrees(np.arctan2(up_comp, horiz_dist))
        
        # 筛选可见卫星
        min_elev = self.config.min_elevation_deg
        visible_mask = elevations >= min_elev
        visible_ids = np.where(visible_mask)[0]
        
        if len(visible_ids) == 0:
            return []
        
        # 方位角（仅对可见卫星计算）
        azimuths = np.degrees(np.arctan2(east_comp[visible_ids], north_comp[visible_ids]))
        azimuths[azimuths < 0] += 360.0
        
        # 构建结果
        user = self.user_manager.users[user_id]
        visible_sats = []
        for j, sat_id in enumerate(visible_ids):
            elev = float(elevations[sat_id])
            rvt = self._estimate_rvt(user, int(sat_id), elev)
            visible_sats.append(VisibilityInfo(
                sat_id=int(sat_id), is_visible=True,
                elevation_deg=elev, azimuth_deg=float(azimuths[j]),
                distance_km=float(distances[sat_id]), rvt_seconds=rvt
            ))
        
        visible_sats.sort(key=lambda x: (-x.rvt_seconds, -x.elevation_deg, x.distance_km))
        return visible_sats[:self.max_visible_sats]

    def _define_spaces(self):
        """定义动作和观测空间"""
        # 最大可见卫星数
        self.max_visible_sats = 10
        
        # ========== 观测空间 ==========
        # 每个用户的观测维度
        user_obs_dim = (
            3 +                              # 用户位置 (lat, lon, alt)
            1 +                              # 用户状态
            5 +                              # 当前服务卫星 (id, dist, elev, snr, rvt)
            self.max_visible_sats * 6 +      # 可见卫星 (id, dist, elev, snr, rvt, load)
            4                                # 当前任务 (data, compute, deadline, priority)
        )
        self.user_obs_dim = user_obs_dim
        
        # 单用户观测空间
        self.single_observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(user_obs_dim,),
            dtype=np.float32
        )
        
        # 全局观测空间（所有用户）
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.num_users, user_obs_dim),
            dtype=np.float32
        )
        
        # ========== 动作空间 ==========
        # 每个用户的动作: [切换决策, 卸载比例]
        # 切换决策: 0=不切换, 1~K=切换到第k个可见卫星
        self.handover_action_dim = self.max_visible_sats + 1
        
        # 混合动作空间
        self.single_action_space = spaces.Dict({
            'handover': spaces.Discrete(self.handover_action_dim),
            'offload_ratio': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # 简化版：使用Box空间表示混合动作
        # [0]: 切换动作 (离散，但用连续表示，取整后使用)
        # [1]: 卸载比例
        self.action_space = spaces.Box(
            low=np.array([[0.0, 0.0]] * self.num_users, dtype=np.float32),
            high=np.array([[self.handover_action_dim - 1, 1.0]] * self.num_users, dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            (观测, 信息)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.task_arrival_rng = self._make_task_arrival_rng(seed)
            self.task_generator = TaskGenerator(seed=seed)
        
        # 重置星座时间
        self.constellation.reset()
        
        # 重置用户状态
        for user in self.user_manager.users:
            user.state = UserState.IDLE
            user.serving_satellite = -1
            user.handover_count = 0
            user.successful_handovers = 0
            user.failed_handovers = 0
        
        # 重置MEC
        self.mec_manager.reset_all()
        
        # 重置任务
        self.task_manager = TaskManager()
        self.user_task_queues = {i: [] for i in range(self.num_users)}
        self.user_tasks = {i: None for i in range(self.num_users)}
        
        # 重置统计
        self.current_step = 0
        self.current_time = 0.0
        self.episode_rewards = []
        self.pending_rewards: Dict[int, float] = {}  # 待发放奖励池（用户ID -> 累积奖励）
        self._offload_task_meta = {}
        self._prev_elevations = {}  # 重置RVT仰角历史
        self._invalidate_visibility_cache()
        self.stats = self._build_stats()
        self._last_load_balance_score = 1.0
        
        # 初始连接：为每个用户选择最佳卫星
        self._initial_connection()
        
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _initial_connection(self):
        """初始连接，为每个用户选择最佳卫星"""
        for user in self.user_manager.users:
            # 计算可见卫星
            visible_sats = self._get_visible_satellites(user)
            
            if visible_sats:
                # 选择仰角最高的卫星
                best_sat = max(visible_sats, key=lambda x: x.elevation_deg)
                user.connect_to_satellite(best_sat.sat_id, self.current_time)
                
                # 更新MEC连接
                server = self.mec_manager.get_server(best_sat.sat_id)
                if server:
                    server.add_user(user.user_id)
            else:
                user.state = UserState.BLOCKED
    
    def step(
        self,
        actions: np.ndarray,
        *,
        return_observation: bool = True,
        return_info: bool = True
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict]:
        """
        执行一步
        
        Args:
            actions: 所有用户的动作 shape=(num_users, 2)
                    actions[i, 0]: 切换动作 (离散)
                    actions[i, 1]: 卸载比例 (连续)
        
        Returns:
            (观测, 奖励, terminated, truncated, 信息)
        """
        # 0. 使可见性缓存失效
        self._invalidate_visibility_cache()
        self._expire_pending_user_tasks()
        
        # 1. 生成新任务
        self._generate_tasks()
        previous_total_handovers = int(self.stats.get('total_handovers', 0))
        
        # 2. 执行每个用户的动作
        user_rewards = []
        for user_id in range(self.num_users):
            user = self.user_manager.users[user_id]
            action = actions[user_id]
            
            # 解析动作
            handover_action = int(np.clip(np.round(action[0]), 0, self.handover_action_dim - 1))
            offload_ratio = float(np.clip(action[1], 0.0, 1.0))
            
            # 执行动作并计算奖励
            reward = self._execute_user_action(user, handover_action, offload_ratio)
            
            # 加入待发放奖励（来自上一步完成的卸载任务）
            if user_id in self.pending_rewards:
                reward += self.pending_rewards[user_id]
                self.pending_rewards[user_id] = 0.0
            
            user_rewards.append(reward)
        
        # 3. 更新环境状态（处理 MEC 队列，累积新的 pending_rewards）
        self._update_environment()
        
        # 4. 计算全局奖励
        total_reward = np.mean(user_rewards)
        self.episode_rewards.append(total_reward)
        load_balance_score = self._compute_load_balance_score()
        self._last_load_balance_score = load_balance_score
        self.stats['load_balance_sum'] += load_balance_score
        self.stats['load_balance_samples'] += 1

        step_user_seconds = float(self.num_users) * float(self.config.time_step_sec)
        blocked_users = sum(1 for user in self.user_manager.users if user.state == UserState.BLOCKED)
        handovers_this_step = max(int(self.stats.get('total_handovers', 0)) - previous_total_handovers, 0)
        blocked_seconds = float(blocked_users) * float(self.config.time_step_sec)
        handover_seconds = float(handovers_this_step) * float(self.config.handover_delay_sec)
        interruption_seconds = min(step_user_seconds, blocked_seconds + handover_seconds)

        self.stats['total_user_seconds'] += step_user_seconds
        self.stats['blocked_user_seconds'] += blocked_seconds
        self.stats['handover_interruption_seconds'] += handover_seconds
        self.stats['service_interruption_seconds'] += interruption_seconds
        
        # 5. 检查终止条件
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.config.max_steps
        
        # 6. 获取新观测
        observation = self._get_observation() if return_observation else None
        info = self._get_info() if return_info else {}
        
        return observation, total_reward, terminated, truncated, info
    
    def _refresh_user_task_head(self, user_id: int) -> None:
        queue = self.user_task_queues[user_id]
        self.user_tasks[user_id] = queue[0] if queue else None

    def _pop_user_task(self, user_id: int) -> Optional[Task]:
        queue = self.user_task_queues[user_id]
        task = queue.pop(0) if queue else None
        self._refresh_user_task_head(user_id)
        return task

    def _compute_unserved_deadline_penalty(
        self,
        elapsed: float,
        max_delay: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Penalty for a generated task that misses its deadline unserved."""
        max_delay = max(float(max_delay), 1e-6)
        delay_ratio = float(elapsed) / max_delay
        reward_qos = -self.config.reward_qos_weight
        penalty_deadline = -self.config.reward_deadline_penalty * min(
            max(delay_ratio - 1.0, 0.0),
            2.0,
        )
        return reward_qos + penalty_deadline, {
            'reward_qos': reward_qos,
            'penalty_deadline': penalty_deadline,
        }

    def _expire_pending_user_tasks(self) -> None:
        """Resolve pending user tasks that already missed their deadline."""
        for user_id in range(self.num_users):
            queue = self.user_task_queues[user_id]
            while queue:
                task = queue[0]
                elapsed = max(float(self.current_time) - float(task.creation_time), 0.0)
                if elapsed <= float(task.max_delay):
                    break
                queue.pop(0)
                self.stats['deadline_violations'] += 1
                self.stats['total_delay'] += elapsed
                task_reward, reward_terms = self._compute_unserved_deadline_penalty(
                    elapsed=elapsed,
                    max_delay=task.max_delay,
                )
                self._record_reward_terms(**reward_terms)
                if user_id not in self.pending_rewards:
                    self.pending_rewards[user_id] = 0.0
                self.pending_rewards[user_id] += task_reward
                self.task_manager.fail_task(task.task_id, self.current_time)
            self._refresh_user_task_head(user_id)

    def _generate_tasks(self):
        """Generate exogenous user tasks independent of service connectivity."""
        for user_id in range(self.num_users):
            if self.task_arrival_rng.random() < self.config.task_arrival_prob:
                task = self.task_generator.generate_task(
                    user_id=user_id,
                    current_time=self.current_time
                )
                self.user_task_queues[user_id].append(task)
                self._refresh_user_task_head(user_id)
                self.task_manager.add_task(task)
                self.stats['total_tasks'] += 1

    def _execute_user_action(
        self,
        user: User,
        handover_action: int,
        offload_ratio: float
    ) -> float:
        """
        执行单个用户的动作
        
        Args:
            user: 用户对象
            handover_action: 切换动作
            offload_ratio: 卸载比例
            
        Returns:
            该用户的奖励
        """
        reward = 0.0
        
        # ========== 处理切换 ==========
        visible_sats = self._get_visible_satellites(user)
        
        if handover_action > 0:
            # 执行切换
            if len(visible_sats) >= handover_action:
                target_sat = visible_sats[handover_action - 1]
                reward += self._execute_handover(user, target_sat)
            else:
                self.stats['total_handovers'] += 1
                self.stats['failed_handovers'] += 1
                invalid_penalty = -self.config.reward_invalid_action_penalty
                reward += invalid_penalty
                self._record_reward_terms(penalty_invalid_action=invalid_penalty)
        else:
            # 不切换，检查当前连接是否有效
            if user.serving_satellite >= 0:
                current_visible = self._is_satellite_visible(user, user.serving_satellite)
                if not current_visible:
                    # 当前卫星不可见，强制切换
                    if visible_sats:
                        best_sat = max(visible_sats, key=lambda x: x.elevation_deg)
                        reward += self._execute_handover(user, best_sat)
                    else:
                        stale_server = self.mec_manager.get_server(user.serving_satellite)
                        if stale_server:
                            stale_server.remove_user(user.user_id)
                        user.serving_satellite = -1
                        user.state = UserState.BLOCKED
                        blocked_penalty = -self.config.reward_blocked_penalty
                        reward += blocked_penalty
                        self._record_reward_terms(penalty_blocked=blocked_penalty)
        
        # ========== 处理任务卸载 ==========
        task = self.user_tasks[user.user_id]
        if task is not None and user.state == UserState.CONNECTED:
            reward += self._execute_offloading(user, task, offload_ratio)
        
            self._pop_user_task(user.user_id)
        return reward

    def _compute_load_balance_score(self) -> float:
        """Use the spread of active satellite load as a cooperative signal."""
        active_loads = []
        for server in self.mec_manager.servers.values():
            aggregate_load = server.queue_length + len(server.connected_users)
            if aggregate_load > 0:
                active_loads.append(float(aggregate_load))

        if len(active_loads) <= 1:
            return 1.0

        return 1.0 / (1.0 + float(np.std(active_loads)))

    def _compute_task_reward(
        self,
        total_delay: float,
        total_energy: float,
        max_delay: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Convert optimization targets into a reward signal.

        Song Xiaoqin et al. optimize end-to-end service delay under constraints,
        while Fu Yiyang et al. emphasize service continuity and collaboration.
        This helper keeps delay/QoS dominant and adds energy as a secondary term.
        """
        max_delay = max(float(max_delay), 1e-6)
        delay_ratio = float(total_delay) / max_delay
        delay_reward = max(1.0 - min(delay_ratio, 1.0), 0.0)

        energy_ref = max(float(self.config.reward_energy_reference), 1e-6)
        energy_reward = max(1.0 - min(float(total_energy) / energy_ref, 1.0), 0.0)
        qos_reward = 1.0 if delay_ratio <= 1.0 else -1.0

        reward_delay = self.config.reward_delay_weight * delay_reward
        reward_energy = self.config.reward_energy_weight * energy_reward
        reward_qos = self.config.reward_qos_weight * qos_reward
        penalty_deadline = 0.0

        if delay_ratio > 1.0:
            penalty_deadline = -self.config.reward_deadline_penalty * min(delay_ratio - 1.0, 2.0)

        reward = reward_delay + reward_energy + reward_qos + penalty_deadline

        return reward, {
            'reward_delay': reward_delay,
            'reward_energy': reward_energy,
            'reward_qos': reward_qos,
            'penalty_deadline': penalty_deadline,
        }

    def _compute_handover_success_probability(
        self,
        elevation_deg: float,
        rvt_seconds: float,
        snr_db: float,
        utilization: float,
        queue_ratio: float,
        migration_load: int = 0,
    ) -> float:
        """
        Estimate handover success from channel quality and target load.

        This makes the success signal sensitive to the chosen target satellite,
        which is closer to the cooperative handover setting in the reference
        papers than using a fixed success probability.
        """
        elevation_score = np.clip(
            (elevation_deg - self.config.min_elevation_deg) /
            max(90.0 - self.config.min_elevation_deg, 1.0),
            0.0,
            1.0,
        )
        rvt_score = np.clip(
            rvt_seconds / max(2.0 * self.config.rvt_threshold_sec, 1.0),
            0.0,
            1.0,
        )
        snr_score = np.clip((snr_db + 5.0) / 30.0, 0.0, 1.0)
        load_headroom = 1.0 - np.clip(utilization, 0.0, 1.0)
        queue_headroom = 1.0 - np.clip(queue_ratio, 0.0, 1.0)
        migration_penalty = np.clip(migration_load / 5.0, 0.0, 1.0)

        success_prob = (
            0.35
            + 0.20 * elevation_score
            + 0.15 * rvt_score
            + 0.15 * snr_score
            + 0.10 * load_headroom
            + 0.10 * queue_headroom
            - 0.10 * migration_penalty
        )
        return float(np.clip(success_prob, 0.1, 0.995))
    
    def _execute_handover(self, user: User, target_sat: VisibilityInfo) -> float:
        """
        执行切换
        
        Args:
            user: 用户
            target_sat: 目标卫星信息
            
        Returns:
            切换奖励
        """
        reward = 0.0
        self.stats['total_handovers'] += 1
        balance_before = self._compute_load_balance_score()
        
        old_sat_id = user.serving_satellite
        old_server = self.mec_manager.get_server(old_sat_id) if old_sat_id >= 0 else None
        new_server = self.mec_manager.get_server(target_sat.sat_id)
        migration_load = 0
        if old_server is not None:
            migration_load = sum(
                1 for task in old_server.task_queue if task['user_id'] == user.user_id
            )
        
        # 从旧卫星断开
        if old_sat_id >= 0:
            if old_server:
                old_server.remove_user(user.user_id)
        
        # 切换时延惩罚（信令开销、链路重建等）
        delay_penalty = min(self.config.handover_delay_sec / 2.0, 1.0)
        
        # 模拟切换过程
        user.start_handover(target_sat.sat_id, self.current_time)
        
        snr_db = self.channel.compute_snr_db(
            target_sat.distance_km,
            target_sat.elevation_deg,
        )
        queue_ratio = 1.0
        utilization = 1.0
        if new_server is not None:
            queue_ratio = new_server.queue_length / max(new_server.config.max_queue_size, 1)
            utilization = new_server.utilization
        success_prob = self._compute_handover_success_probability(
            elevation_deg=target_sat.elevation_deg,
            rvt_seconds=target_sat.rvt_seconds,
            snr_db=snr_db,
            utilization=utilization,
            queue_ratio=queue_ratio,
            migration_load=migration_load,
        )
        handover_success = self.rng.random() < success_prob
        
        if handover_success:
            user.complete_handover(target_sat.sat_id, self.current_time, success=True)
            
            # 连接到新卫星的MEC
            if new_server:
                new_server.add_user(user.user_id)
            
            # 迁移旧卫星上该用户的排队任务到新卫星
            if old_sat_id >= 0:
                migration_result = self.mec_manager.migrate_user_tasks(
                    user_id=user.user_id,
                    old_sat_id=old_sat_id,
                    new_sat_id=target_sat.sat_id,
                    handover_delay=self.config.handover_delay_sec
                )
                migration_penalty = 0.05 * migration_result['migrated']
                migration_penalty += 0.1 * migration_result['failed']
            else:
                migration_penalty = 0.0
            
            self.stats['successful_handovers'] += 1
            
            elevation_score = np.clip(target_sat.elevation_deg / 90.0, 0.0, 1.0)
            rvt_score = np.clip(
                target_sat.rvt_seconds / max(self.config.rvt_threshold_sec, 1.0),
                0.0,
                1.0,
            )
            balance_after = self._compute_load_balance_score()
            balance_gain = balance_after - balance_before

            handover_score = 0.5 * elevation_score + 0.5 * rvt_score
            reward_handover = self.config.reward_handover_weight * handover_score
            penalty_handover_cost = -self.config.reward_handover_weight * (
                delay_penalty + migration_penalty
            )
            reward_load_balance = self.config.reward_load_balance_weight * balance_gain

            reward += reward_handover + penalty_handover_cost + reward_load_balance
            self._record_reward_terms(
                reward_handover=reward_handover,
                penalty_handover_cost=penalty_handover_cost,
                reward_load_balance=reward_load_balance,
            )
        else:
            user.complete_handover(-1, self.current_time, success=False)
            self.stats['failed_handovers'] += 1
            failed_penalty = -self.config.reward_failed_handover_penalty
            reward += failed_penalty
            self._record_reward_terms(penalty_failed_handover=failed_penalty)
        
        return reward
    
    def _execute_offloading(
        self,
        user: User,
        task: Task,
        offload_ratio: float
    ) -> float:
        """
        执行任务卸载（支持卫星资源竞争与排队）
        
        改进版：
        - 本地计算部分立即计算时延/能耗
        - 卸载部分将任务入队到目标卫星的 MEC 服务器
        - 任务完成后的奖励延迟发放（通过 pending_rewards）
        
        Args:
            user: 用户
            task: 任务
            offload_ratio: 卸载比例
            
        Returns:
            本步可立即发放的奖励（本地计算部分 + 入队状态）
        """
        reward = 0.0
        offload_ratio = float(np.clip(offload_ratio, 0.0, 1.0))
        if offload_ratio < self.config.min_effective_offload_ratio:
            offload_ratio = 0.0
        wait_delay = max(float(self.current_time) - float(task.creation_time), 0.0)
        
        # 获取卫星信息
        sat_id = user.serving_satellite
        if sat_id < 0:
            return -0.5  # 无连接惩罚
        
        # 获取链路信息
        vis_info = self._get_satellite_visibility(user, sat_id)
        if vis_info is None or not vis_info.is_visible:
            return -0.5
        
        server = self.mec_manager.get_server(sat_id)
        if server is None:
            return -0.5
        
        # ========== 本地计算部分（立即处理） ==========
        local_ratio = 1.0 - offload_ratio
        local_cycles = local_ratio * task.computation
        local_delay = 0.0
        local_energy = 0.0
        
        if local_cycles > 0:
            local_delay = self.offload_calc.compute_local_delay(local_cycles)
            local_energy = self.offload_calc.compute_local_energy(local_cycles)
            self.stats['total_energy'] += local_energy
        
        # ========== 卸载部分（入队处理） ==========
        if offload_ratio > 0:
            offload_cycles = offload_ratio * task.computation
            offload_data_bits = offload_ratio * task.data_size
            
            # 计算上传/下载时延（固定，与排队无关）
            upload_delay, download_delay = self.offload_calc.compute_transmission_delay(
                offload_data_bits, vis_info.distance_km, vis_info.elevation_deg
            )
            
            # 传输能耗
            upload_energy = self.offload_calc.compute_transmission_energy(
                offload_data_bits, vis_info.distance_km, vis_info.elevation_deg
            )
            self.stats['total_energy'] += upload_energy
            
            # 尝试将任务入队
            enqueued = server.enqueue_task(
                user_id=user.user_id,
                task_id=task.task_id,
                offload_cycles=offload_cycles,
                offload_data_bits=offload_data_bits,
                max_delay=task.max_delay,
                arrival_time=task.creation_time,
                upload_delay=upload_delay,
                download_delay=download_delay,
                offload_ratio=offload_ratio,
                upload_energy=upload_energy,
            )
            
            if enqueued:
                # 入队成功，记录任务元信息用于后续奖励计算
                task.offload_ratio = offload_ratio
                task.local_delay = local_delay
                task.local_energy = local_energy
                task.transmission_energy = upload_energy
                self._offload_task_meta[(user.user_id, task.task_id)] = {
                    'local_delay': wait_delay + local_delay,
                    'local_energy': local_energy,
                }
                queue_margin = 1.0 - (
                    server.queue_length / max(server.config.max_queue_size, 1)
                )
                enqueue_bonus = self.config.reward_enqueue_bonus * max(queue_margin, 0.0)
                reward += enqueue_bonus
                self._record_reward_terms(reward_enqueue=enqueue_bonus)
            else:
                # 队列已满，任务被拒绝 -> 强制本地执行或丢弃
                # 这里选择：退化为完全本地执行
                fallback_cycles = offload_cycles
                fallback_delay = self.offload_calc.compute_local_delay(fallback_cycles)
                fallback_energy = self.offload_calc.compute_local_energy(fallback_cycles)
                
                local_delay += fallback_delay
                local_energy += fallback_energy
                total_delay = wait_delay + local_delay
                self.stats['total_delay'] += total_delay
                self.stats['total_energy'] += fallback_energy
                
                # 惩罚：队列满导致无法卸载
                queue_penalty = -self.config.reward_queue_full_penalty
                reward += queue_penalty
                self._record_reward_terms(penalty_queue_full=queue_penalty)

                if total_delay <= task.max_delay:
                    self.stats['completed_tasks'] += 1
                else:
                    self.stats['deadline_violations'] += 1
                task_reward, reward_terms = self._compute_task_reward(
                    total_delay=total_delay,
                    total_energy=local_energy + upload_energy,
                    max_delay=task.max_delay,
                )
                reward += task_reward
                self._record_reward_terms(**reward_terms)
                
                task.total_delay = total_delay
                task.total_energy = local_energy + upload_energy
                task.offload_ratio = 0.0  # 实际退化为本地
        else:
            # 完全本地执行
            total_delay = wait_delay + local_delay
            self.stats['total_delay'] += total_delay
            task.total_delay = total_delay
            task.total_energy = local_energy
            task.offload_ratio = 0.0
            
            if total_delay <= task.max_delay:
                self.stats['completed_tasks'] += 1
            else:
                self.stats['deadline_violations'] += 1
            task_reward, reward_terms = self._compute_task_reward(
                total_delay=total_delay,
                total_energy=local_energy,
                max_delay=task.max_delay,
            )
            reward += task_reward
            self._record_reward_terms(**reward_terms)
        
        return reward
    
    def _update_environment(self):
        """更新环境状态（包含 MEC 队列处理）"""
        # 更新时间
        self.current_time += self.config.time_step_sec
        self._expire_pending_user_tasks()
        
        # 传播星座
        self.constellation.propagate(self.config.time_step_sec)
        
        # ========== 处理所有卫星的任务队列 ==========
        completed_tasks = self.mec_manager.process_all_queues(
            self.current_time, self.config.time_step_sec
        )
        
        # 累积完成任务的奖励到 pending_rewards
        for task_info in completed_tasks:
            user_id = task_info['user_id']
            task_meta = self._offload_task_meta.pop(
                (user_id, task_info['task_id']),
                {},
            )
            total_delay = max(
                float(task_info['total_delay']),
                float(task_meta.get('local_delay', 0.0)),
            )
            total_energy = float(task_info.get('upload_energy', 0.0))
            total_energy += float(task_meta.get('local_energy', 0.0))
            deadline_met = total_delay <= task_info['max_delay']

            if deadline_met:
                self.stats['completed_tasks'] += 1
            else:
                self.stats['deadline_violations'] += 1
            task_reward, reward_terms = self._compute_task_reward(
                total_delay=total_delay,
                total_energy=total_energy,
                max_delay=task_info['max_delay'],
            )
            self._record_reward_terms(**reward_terms)
            
            # Split-task delay follows the max(local, offloaded) model.
            self.stats['total_delay'] += total_delay
            
            # 将奖励加入待发放池
            if user_id not in self.pending_rewards:
                self.pending_rewards[user_id] = 0.0
            self.pending_rewards[user_id] += task_reward
        
        # 检查用户连接状态
        for user in self.user_manager.users:
            if user.state == UserState.CONNECTED:
                # 检查当前卫星是否仍可见
                if not self._is_satellite_visible(user, user.serving_satellite):
                    stale_sat_id = user.serving_satellite
                    if stale_sat_id >= 0:
                        stale_server = self.mec_manager.get_server(stale_sat_id)
                        if stale_server:
                            stale_server.remove_user(user.user_id)
                    user.serving_satellite = -1
                    user.state = UserState.BLOCKED
                    self.stats['forced_disconnects'] += 1
    
    def _get_visible_satellites(self, user: User) -> List[VisibilityInfo]:
        """获取用户可见的卫星列表（带缓存）"""
        uid = user.user_id
        if uid in self._visibility_cache:
            return self._visibility_cache[uid]
        result = self._compute_visibility_batch(uid)
        self._visibility_cache[uid] = result
        return result
    
    def _is_satellite_visible(self, user: User, sat_id: int) -> bool:
        """检查卫星是否对用户可见（利用缓存）"""
        visible = self._get_visible_satellites(user)
        return any(v.sat_id == sat_id for v in visible)
    
    def _get_satellite_visibility(self, user: User, sat_id: int) -> Optional[VisibilityInfo]:
        """获取特定卫星的可见性信息（利用缓存或快速单点计算）"""
        # 先查缓存
        visible = self._get_visible_satellites(user)
        for v in visible:
            if v.sat_id == sat_id:
                return v
        # 不在可见列表中，快速计算仰角判断
        user_pos = self._user_pos_ecef[user.user_id]
        sat_pos = self.constellation._all_pos_ecef[sat_id]
        vec = sat_pos - user_pos
        up_comp = np.dot(vec, self._user_e_up[user.user_id])
        east_comp = np.dot(vec, self._user_e_east[user.user_id])
        north_comp = np.dot(vec, self._user_e_north[user.user_id])
        horiz = np.sqrt(east_comp**2 + north_comp**2)
        elev = np.degrees(np.arctan2(up_comp, horiz))
        dist = np.linalg.norm(vec)
        azim = np.degrees(np.arctan2(east_comp, north_comp))
        if azim < 0:
            azim += 360.0
        is_vis = elev >= self.config.min_elevation_deg
        rvt = self._estimate_rvt(user, sat_id, elev) if is_vis else 0.0
        return VisibilityInfo(sat_id=sat_id, is_visible=is_vis,
                              elevation_deg=elev, azimuth_deg=azim,
                              distance_km=dist, rvt_seconds=rvt)
    
    def _estimate_rvt(self, user: User, sat_id: int, current_elevation: float) -> float:
        """
        估算剩余可见时间 (RVT)
        
        改进模型：利用仰角变化率判断上升/下降阶段，
        结合正弦近似过境曲线估算剩余可见时间
        """
        orbital_period = self.constellation.orbital_period
        max_visible_time = orbital_period / 5  # ~1146s for 550km
        min_elev = self.config.min_elevation_deg
        
        # 跟踪仰角变化以判断升降阶段
        key = (user.user_id, sat_id)
        prev_elev = self._prev_elevations.get(key, None)
        self._prev_elevations[key] = current_elevation
        
        # 判断是否在下降阶段
        if prev_elev is not None:
            is_descending = current_elevation < prev_elev - 0.1  # 0.1度容差
        else:
            # 首次观测，根据仰角高低猜测
            is_descending = current_elevation > 45.0
        
        # 正弦近似：elevation(t) ≈ max_elev * sin(π * t / T_visible)
        # 归一化仰角到 [0, 1] 范围 (min_elev ~ 90)
        norm_elev = max((current_elevation - min_elev) / (90.0 - min_elev), 0.01)
        norm_elev = min(norm_elev, 1.0)
        
        # 反正弦求当前在过境弧上的相位
        phase = math.asin(math.sqrt(norm_elev))  # [0, π/2]
        
        if is_descending:
            # 下降阶段：已过顶点，phase 映射到 [π/2, π]
            remaining_fraction = (math.pi / 2 - phase) / math.pi
        else:
            # 上升阶段：还未过顶，剩余包含上升+下降
            remaining_fraction = (math.pi - phase) / math.pi
        
        rvt = max_visible_time * remaining_fraction
        
        # 安全边界：至少留 5 秒余量
        safety_margin = 5.0
        rvt = max(rvt - safety_margin, 0.0)
        
        return rvt
    
    def _get_observation(self) -> np.ndarray:
        """获取所有用户的观测"""
        observations = np.zeros((self.num_users, self.user_obs_dim), dtype=np.float32)
        
        for user_id, user in enumerate(self.user_manager.users):
            observations[user_id] = self._get_user_observation(user)
        
        return observations
    
    def _get_user_observation(self, user: User) -> np.ndarray:
        """获取单个用户的观测"""
        obs = np.zeros(self.user_obs_dim, dtype=np.float32)
        idx = 0
        
        # 1. 用户位置 (归一化)
        obs[idx:idx+3] = [
            user.position.latitude / 90.0,
            user.position.longitude / 180.0,
            user.position.altitude / 100.0
        ]
        idx += 3
        
        # 2. 用户状态
        obs[idx] = user.state.value / 3.0
        idx += 1
        
        # 3. 当前服务卫星信息
        if user.serving_satellite >= 0:
            vis_info = self._get_satellite_visibility(user, user.serving_satellite)
            if vis_info and vis_info.is_visible:
                server = self.mec_manager.get_server(user.serving_satellite)
                obs[idx:idx+5] = [
                    user.serving_satellite / self.num_satellites,
                    vis_info.distance_km / 2000.0,
                    vis_info.elevation_deg / 90.0,
                    self.channel.compute_snr_db(vis_info.distance_km, vis_info.elevation_deg) / 50.0,
                    vis_info.rvt_seconds / 600.0
                ]
        idx += 5
        
        # 4. 可见卫星信息
        visible_sats = self._get_visible_satellites(user)
        for i in range(self.max_visible_sats):
            if i < len(visible_sats):
                sat = visible_sats[i]
                server = self.mec_manager.get_server(sat.sat_id)
                load = server.utilization if server else 0.0
                
                obs[idx:idx+6] = [
                    sat.sat_id / self.num_satellites,
                    sat.distance_km / 2000.0,
                    sat.elevation_deg / 90.0,
                    self.channel.compute_snr_db(sat.distance_km, sat.elevation_deg) / 50.0,
                    sat.rvt_seconds / 600.0,
                    load
                ]
            idx += 6
        
        # 5. 当前任务信息
        task = self.user_tasks[user.user_id]
        if task is not None:
            obs[idx:idx+4] = [
                task.data_size / 1e8,           # 归一化
                task.computation / 1e10,         # 归一化
                task.max_delay / 10.0,           # 归一化
                task.task_type.value / 2.0       # 归一化
            ]
        idx += 4
        
        return obs
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        stats_summary = self.get_stats_summary()
        return {
            'step': self.current_step,
            'time': self.current_time,
            'stats': stats_summary,
            'load_balance_score': self._last_load_balance_score,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
        }

    def get_stats_summary(self) -> Dict[str, float]:
        """Return a snapshot of derived environment statistics."""
        return self._summarize_stats(self.stats.copy())
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self._render_human()
        return None
    
    def _render_human(self):
        """人类可读的渲染"""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}, Time: {self.current_time:.1f}s")
        print(f"{'='*60}")
        
        # 用户状态
        connected = sum(1 for u in self.user_manager.users if u.state == UserState.CONNECTED)
        blocked = sum(1 for u in self.user_manager.users if u.state == UserState.BLOCKED)
        print(f"Users: {connected} connected, {blocked} blocked")
        
        # 统计信息
        summary = self._summarize_stats(self.stats.copy())
        print(f"Handovers: {self.stats['successful_handovers']}/{self.stats['total_handovers']}")
        print(
            f"Tasks: {self.stats['completed_tasks']}/{summary['resolved_tasks']} resolved, "
            f"{summary['pending_tasks']} pending"
        )
        if summary['resolved_tasks'] > 0:
            print(f"Avg Delay: {summary['avg_delay']*1000:.2f}ms")
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_state_for_graph(self) -> Dict:
        """
        获取用于构建异质图的状态信息
        
        Returns:
            包含卫星、用户、边信息的字典
        """
        state = {
            'satellites': [],
            'users': [],
            'edges': []
        }
        
        # 卫星状态
        for sat_id in range(self.num_satellites):
            sat_pos = self.constellation.get_satellite_position(sat_id)
            server = self.mec_manager.get_server(sat_id)
            
            state['satellites'].append({
                'id': sat_id,
                'position': sat_pos,
                'utilization': server.utilization if server else 0.0,
                'queue_length': server.queue_length if server else 0,
                'connected_users': len(server.connected_users) if server else 0,
            })
        
        # 用户状态
        for user in self.user_manager.users:
            task = self.user_tasks[user.user_id]
            state['users'].append({
                'id': user.user_id,
                'position': {
                    'latitude': user.position.latitude,
                    'longitude': user.position.longitude,
                },
                'state': user.state.value,
                'serving_satellite': user.serving_satellite,
                'task': {
                    'data_size': task.data_size if task else 0,
                    'computation': task.computation if task else 0,
                    'max_delay': task.max_delay if task else 0,
                } if task else None,
            })
        
        # 用户-卫星边
        for user in self.user_manager.users:
            visible_sats = self._get_visible_satellites(user)
            for sat in visible_sats:
                state['edges'].append({
                    'type': 'user_satellite',
                    'user_id': user.user_id,
                    'satellite_id': sat.sat_id,
                    'distance': sat.distance_km,
                    'elevation': sat.elevation_deg,
                    'rvt': sat.rvt_seconds,
                })
        
        return state


# ==================== 便捷函数 ====================

def make_env(config: Optional[EnvConfig] = None, **kwargs) -> LEOSatelliteEnv:
    """创建环境的便捷函数"""
    if config is None:
        config = EnvConfig(**kwargs)
    return LEOSatelliteEnv(config)


def make_vec_env(
    num_envs: int,
    config: Optional[EnvConfig] = None,
    **kwargs
) -> List[LEOSatelliteEnv]:
    """创建多个并行环境"""
    envs = []
    for i in range(num_envs):
        env_config = config or EnvConfig(**kwargs)
        env_config.seed = (config.seed or 0) + i if config else i
        envs.append(LEOSatelliteEnv(env_config))
    return envs
