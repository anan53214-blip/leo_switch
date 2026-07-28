"""
Gymnasium强化学习环境
整合星座、用户、信道、MEC等模块，实现联合切换与卸载优化环境

主要功能：
1. 多智能体环境（每个用户是一个智能体）
2. 混合动作空间（离散切换 + 连续卸载比例）
3. QoS 门控奖励函数（任务结果、时延、能耗和服务中断）

参考论文：
- 付一阳等《星地融合网络中基于异质图表征的多智能体协作切换方法》
- 宋晓勤等《基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法》
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, fields
from .constellation import WalkerConstellation
from .visibility import VisibilityCalculator, VisibilityInfo
from .user import User, UserState, UserGenerator, UserManager
from .task import Task, TaskGenerator, TaskManager
from .channel import SatelliteChannel
from .mec import MECManager, OffloadingCalculator


TASK_SUCCESS_REWARD = 1.0
TASK_FAILURE_PENALTY = 1.0
REWARD_ENERGY_REFERENCE_J = 10.0
REWARD_BREAKDOWN_KEYS = (
    'reward_task_success',
    'penalty_delay',
    'penalty_energy',
    'penalty_task_failure',
    'penalty_service_interruption',
    'penalty_failed_handover',
)


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
    num_users: int = 20
    user_center_lat: float = 39.9      # 北京
    user_center_lon: float = 116.4
    user_radius_deg: float = 5.0       # 扩大用户分布范围，增加可见卫星差异
    
    # 仿真参数
    time_step_sec: float = 1.0         # 时间步长 (秒)
    max_steps: int = 600               # 每个 episode 的最大步数（10 分钟）

    # Episode 起始时间随机化
    randomize_episode_start: bool = True
    episode_start_time_jitter_sec: float = 5730.127  # 一个 550km 轨道周期
    
    # 可见性参数
    min_elevation_deg: float = 10.0    # More moderate visibility threshold
    
    # 切换参数
    handover_delay_sec: float = 0.6    # Moderate signaling cost
    rvt_threshold_sec: float = 60.0    # Encourage proactive switching
    pre_handover_rvt_sec: float = 30.0  # Pre-handover RVT 阈值
    handover_min_snr_db: float = -5.0  # 目标链路硬准入阈值
    
    # 任务参数
    task_arrival_prob: float = 0.35    # Moderate competition without pathological saturation
    min_effective_offload_ratio: float = 0.05  # Treat tiny noisy offload actions as local execution
    task_arrival_seed_offset: int = 7919
    
    # QoS 门控奖励：成功/失败固定为 +1/-1，仅保留四个可调系数
    reward_delay_weight: float = 0.60
    reward_energy_weight: float = 0.10
    reward_interruption_weight: float = 0.30
    reward_failed_handover_penalty: float = 0.20
    
    # 随机种子
    seed: Optional[int] = None


def build_env_config(source: Any = None, **overrides: Any) -> EnvConfig:
    """从配置对象或字典提取 EnvConfig 字段，避免各入口重复维护字段清单。"""
    field_names = {item.name for item in fields(EnvConfig)}
    values: Dict[str, Any] = {}
    if isinstance(source, Mapping):
        values.update({
            name: source[name]
            for name in field_names
            if name in source
        })
    elif source is not None:
        values.update({
            name: getattr(source, name)
            for name in field_names
            if hasattr(source, name)
        })
    unknown = set(overrides) - field_names
    if unknown:
        raise TypeError(f"未知环境配置字段: {sorted(unknown)}")
    values.update(overrides)
    return EnvConfig(**values)


def _finite_load_variance_samples(values) -> np.ndarray:
    samples = np.asarray(list(values or []), dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    return np.clip(samples, 0.0, 0.25)


def _finite_nonnegative_samples(values) -> np.ndarray:
    samples = np.asarray(list(values or []), dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    return samples[samples >= 0.0]


def summarize_env_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """Derive reliability and QoS metrics from raw environment counters."""
    failed_tasks = int(stats.get('failed_tasks', 0))
    resolved_tasks = int(
        stats.get('completed_tasks', 0)
        + stats.get('deadline_violations', 0)
        + failed_tasks
    )
    total_tasks = int(stats.get('total_tasks', 0))
    pending_tasks = max(total_tasks - resolved_tasks, 0)

    total_handovers = int(stats.get('total_handovers', 0))
    handover_attempts = int(stats.get('handover_attempts', total_handovers))
    handover_committed = int(stats.get('handover_committed', total_handovers))
    # 兼容旧日志以及只写入 total_handovers 的调用方。
    if handover_attempts != total_handovers:
        handover_attempts = total_handovers
        handover_committed = total_handovers
    successful_handovers = int(stats.get('successful_handovers', 0))
    failed_handovers = int(stats.get('failed_handovers', 0))
    forced_disconnects = int(stats.get('forced_disconnects', 0))
    continuity_events = handover_committed + forced_disconnects

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
    failed_count = deadline_count + float(failed_tasks)
    task_completion_rate = completed_count / max(resolved_tasks, 1)
    task_success_rate = completed_count / max(total_tasks, 1)
    task_failure_rate = failed_count / max(total_tasks, 1)
    task_settlement_rate = float(resolved_tasks) / max(total_tasks, 1)
    task_resolution_rate = task_settlement_rate
    handover_frequency = (
        float(handover_committed) / total_user_seconds
        if total_user_seconds > 0.0 else 0.0
    )
    blocked_time_ratio = (
        blocked_user_seconds / total_user_seconds
        if total_user_seconds > 0.0 else 0.0
    )
    handovers_per_user_minute = 60.0 * handover_frequency

    summary = stats.copy()
    mec_load_fairness = (
        float(stats.get('load_balance_sum', 0.0)) /
        max(int(stats.get('load_balance_samples', 0)), 1)
    )
    total_energy = float(stats.get('total_energy', 0.0))
    energy_per_successful_task = total_energy / max(completed_count, 1.0)
    successful_task_delay_samples = _finite_nonnegative_samples(
        stats.get('successful_task_delay_samples', [])
    )
    avg_success_delay = (
        float(np.mean(successful_task_delay_samples))
        if successful_task_delay_samples.size > 0
        else 0.0
    )
    p95_success_delay = (
        float(np.percentile(successful_task_delay_samples, 95))
        if successful_task_delay_samples.size > 0
        else 0.0
    )
    jain_mec_load_fairness = (
        float(stats.get('jain_load_fairness_sum', 0.0)) /
        max(int(stats.get('jain_load_fairness_samples', 0)), 1)
    )
    load_variance_samples = _finite_load_variance_samples(
        stats.get('load_variance_samples', [])
    )
    load_balance_variance = (
        float(np.mean(load_variance_samples))
        if load_variance_samples.size > 0 else 0.0
    )
    load_balance_coefficient = float(
        (1.0 - 4.0 * load_balance_variance) /
        (1.0 + 4.0 * load_balance_variance)
    )
    load_variance_cdf = [
        {
            'x': float(value),
            'cdf': float((index + 1) / max(int(load_variance_samples.size), 1)),
        }
        for index, value in enumerate(np.sort(load_variance_samples))
    ]

    summary.update({
        'resolved_tasks': resolved_tasks,
        'pending_tasks': pending_tasks,
        'handover_success_rate': (
            float(successful_handovers) / max(handover_attempts, 1)
        ),
        'handover_failure_rate': (
            float(failed_handovers) / max(handover_attempts, 1)
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
        'handover_frequency': handover_frequency,
        'handovers_per_user_minute': handovers_per_user_minute,
        'blocked_time_ratio': blocked_time_ratio,
        'avg_delay': avg_delay,
        'avg_success_delay': avg_success_delay,
        'p95_success_delay': p95_success_delay,
        'successful_task_delay_samples': [
            float(value) for value in successful_task_delay_samples
        ],
        'energy_per_successful_task': energy_per_successful_task,
        'load_balance_variance': load_balance_variance,
        'load_balance_coefficient': load_balance_coefficient,
        'load_variance_sample_count': int(load_variance_samples.size),
        'load_variance_samples': [float(value) for value in load_variance_samples],
        'load_variance_cdf': load_variance_cdf,
        'mec_load_fairness': mec_load_fairness,
        'jain_mec_load_fairness': jain_mec_load_fairness,
        'active_load_balance_score': mec_load_fairness,
        'avg_load_balance_score': mec_load_fairness,
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
        self.last_user_rewards = np.zeros(self.config.num_users, dtype=np.float32)
        self._step_handover_interruption_seconds = np.zeros(
            self.config.num_users,
            dtype=np.float32,
        )
        self._offload_task_meta: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        # 可见性和 RVT 缓存绑定明确的几何版本。
        self.geometry_version = 0
        self._visibility_cache_version = self.geometry_version
        self._visibility_cache: Dict[int, List[VisibilityInfo]] = {}
        self._rvt_prediction_version = -1
        self._rvt_prediction_time = None
        self._rvt_prediction_offsets = np.zeros(0, dtype=np.float64)
        self._rvt_future_positions = np.zeros(
            (0, self.num_satellites, 3),
            dtype=np.float64,
        )
        
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
        """使可见性缓存失效，并将缓存绑定到当前几何版本。"""
        self._visibility_cache.clear()
        self._visibility_cache_version = self.geometry_version

    @staticmethod
    def _build_stats() -> Dict[str, float]:
        """Initialize environment statistics and reward breakdown terms."""
        return {
            'total_handovers': 0,
            'handover_attempts': 0,
            'handover_committed': 0,
            'handover_aborted': 0,
            'handover_radio_failures': 0,
            'migration_rejections': 0,
            'reconnection_attempts': 0,
            'reconnections': 0,
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
            'failed_tasks': 0,
            'total_delay': 0.0,
            'successful_task_delay_samples': [],
            'total_energy': 0.0,
            'reward_task_success': 0.0,
            'penalty_delay': 0.0,
            'penalty_energy': 0.0,
            'penalty_task_failure': 0.0,
            'penalty_service_interruption': 0.0,
            'penalty_failed_handover': 0.0,
            'load_balance_sum': 0.0,
            'load_balance_samples': 0,
            'jain_load_fairness_sum': 0.0,
            'jain_load_fairness_samples': 0,
            'load_variance_samples': [],
        }

    def _record_reward_terms(
        self,
        *,
        average_over_users: bool = True,
        **terms: float,
    ) -> None:
        """累计各项对环境全局平均 reward 的实际贡献。"""
        scale = 1.0 / max(self.num_users, 1) if average_over_users else 1.0
        for key, value in terms.items():
            if key in self.stats:
                self.stats[key] += float(value) * scale

    def _compute_service_interruption_penalties(
        self,
        interruption_seconds: np.ndarray,
    ) -> np.ndarray:
        """按用户计算本时隙服务中断惩罚。"""
        slot_duration = max(float(self.config.time_step_sec), 1e-6)
        interruption_ratio = np.clip(
            np.asarray(interruption_seconds, dtype=np.float32) / slot_duration,
            0.0,
            1.0,
        )
        return (
            -float(self.config.reward_interruption_weight) * interruption_ratio
        ).astype(np.float32)

    @staticmethod
    def _summarize_stats(stats: Dict[str, float]) -> Dict[str, float]:
        """Add derived QoS and reliability metrics on top of raw counters."""
        return summarize_env_stats(stats)

    def _get_rvt_prediction_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回绑定当前几何版本的未来卫星位置预测网格。"""
        if (
            self._rvt_prediction_version != self.geometry_version
            or self._rvt_prediction_time != self.constellation.current_time
        ):
            prediction_step = 5.0
            max_prediction = max(
                float(self.constellation.orbital_period) / 2.0,
                prediction_step,
            )
            offsets = np.arange(
                0.0,
                max_prediction + prediction_step,
                prediction_step,
                dtype=np.float64,
            )
            self._rvt_prediction_offsets = offsets
            self._rvt_future_positions = (
                self.constellation.get_all_positions_ecef_at_offsets(offsets)
            )
            self._rvt_prediction_version = self.geometry_version
            self._rvt_prediction_time = self.constellation.current_time
        return self._rvt_prediction_offsets, self._rvt_future_positions

    def _estimate_rvt_batch(
        self,
        user_id: int,
        satellite_ids: np.ndarray,
        current_elevations: np.ndarray,
    ) -> np.ndarray:
        """批量预测可见卫星首次下降穿过最低仰角的时间。"""
        sat_ids = np.asarray(satellite_ids, dtype=np.int64).reshape(-1)
        current = np.asarray(current_elevations, dtype=np.float64).reshape(-1)
        if sat_ids.size == 0:
            return np.zeros(0, dtype=np.float64)

        offsets, future_positions = self._get_rvt_prediction_grid()
        vectors = (
            future_positions[:, sat_ids, :]
            - self._user_pos_ecef[user_id][None, None, :]
        )
        up = vectors @ self._user_e_up[user_id]
        east = vectors @ self._user_e_east[user_id]
        north = vectors @ self._user_e_north[user_id]
        elevations = np.degrees(
            np.arctan2(up, np.sqrt(east**2 + north**2))
        )

        min_elevation = float(self.config.min_elevation_deg)
        below = elevations[1:] < min_elevation
        has_crossing = np.any(below, axis=0)
        first_below = np.argmax(below, axis=0) + 1
        rvt = np.full(
            sat_ids.shape,
            float(offsets[-1]),
            dtype=np.float64,
        )

        for column in np.where(has_crossing)[0]:
            upper_index = int(first_below[column])
            lower_index = upper_index - 1
            lower_elevation = float(elevations[lower_index, column])
            upper_elevation = float(elevations[upper_index, column])
            denominator = lower_elevation - upper_elevation
            if abs(denominator) <= 1e-9:
                fraction = 1.0
            else:
                fraction = np.clip(
                    (lower_elevation - min_elevation) / denominator,
                    0.0,
                    1.0,
                )
            rvt[column] = (
                float(offsets[lower_index])
                + fraction
                * float(offsets[upper_index] - offsets[lower_index])
            )

        rvt[current < min_elevation] = 0.0
        return np.maximum(rvt, 0.0)

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
        
        rvts = self._estimate_rvt_batch(
            user_id,
            visible_ids,
            elevations[visible_ids],
        )

        # 构建结果
        visible_sats = []
        for j, sat_id in enumerate(visible_ids):
            elev = float(elevations[sat_id])
            visible_sats.append(VisibilityInfo(
                sat_id=int(sat_id), is_visible=True,
                elevation_deg=elev, azimuth_deg=float(azimuths[j]),
                distance_km=float(distances[sat_id]),
                rvt_seconds=float(rvts[j]),
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
        
        # 重置星座时间（支持 episode 起始时间随机化）
        offset_sec = 0.0
        if self.config.randomize_episode_start:
            max_offset = max(float(self.config.episode_start_time_jitter_sec), 0.0)
            offset_sec = float(self.rng.uniform(0.0, max_offset)) if max_offset > 0 else 0.0
        self.constellation.reset(time_offset_sec=offset_sec)
        self.geometry_version = 0
        self._rvt_prediction_version = -1
        
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
        self.last_user_rewards = np.zeros(self.config.num_users, dtype=np.float32)
        self._step_handover_interruption_seconds = np.zeros(
            self.config.num_users,
            dtype=np.float32,
        )
        self._offload_task_meta = {}
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
        # 当前动作继续使用生成该动作观测时的同一份候选卫星映射。
        # 可见性缓存只在几何状态推进后失效。
        self._expire_pending_user_tasks()
        
        # 1. 生成新任务
        self._generate_tasks()
        self._step_handover_interruption_seconds.fill(0.0)
        
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
            
            user_rewards.append(reward)
        
        # 3. 更新环境状态（处理 MEC 队列，产生本步任务结算奖励）
        self._update_environment()

        # 本步产生的任务奖励在本步发放，避免 terminal/truncated 后丢失。
        for user_id in range(self.num_users):
            pending_reward = float(self.pending_rewards.get(user_id, 0.0))
            if pending_reward != 0.0:
                user_rewards[user_id] += pending_reward
                self.pending_rewards[user_id] = 0.0
        
        # 4. 计算用户级服务中断惩罚和全局平均奖励
        self._record_load_balance_metrics()

        slot_duration = float(self.config.time_step_sec)
        blocked_seconds_by_user = np.asarray(
            [
                slot_duration if user.state == UserState.BLOCKED else 0.0
                for user in self.user_manager.users
            ],
            dtype=np.float32,
        )
        interruption_seconds_by_user = np.minimum(
            blocked_seconds_by_user + self._step_handover_interruption_seconds,
            slot_duration,
        )
        interruption_penalties = self._compute_service_interruption_penalties(
            interruption_seconds_by_user
        )
        self.last_user_rewards = (
            np.asarray(user_rewards, dtype=np.float32) + interruption_penalties
        )
        total_reward = float(np.mean(self.last_user_rewards))

        step_user_seconds = float(self.num_users) * slot_duration
        blocked_seconds = float(np.sum(blocked_seconds_by_user))
        handover_seconds = float(np.sum(self._step_handover_interruption_seconds))
        interruption_seconds = float(np.sum(interruption_seconds_by_user))
        self.stats['total_user_seconds'] += step_user_seconds
        self.stats['blocked_user_seconds'] += blocked_seconds
        self.stats['handover_interruption_seconds'] += handover_seconds
        self.stats['service_interruption_seconds'] += interruption_seconds
        self._record_reward_terms(
            average_over_users=False,
            penalty_service_interruption=float(np.mean(interruption_penalties))
        )
        self.episode_rewards.append(total_reward)
        
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

    @staticmethod
    def _compute_task_failure_reward() -> Tuple[float, Dict[str, float]]:
        """所有终态任务失败统一记为 -1，避免重复处罚 deadline。"""
        penalty = -TASK_FAILURE_PENALTY
        return penalty, {
            'reward_task_success': 0.0,
            'penalty_delay': 0.0,
            'penalty_energy': 0.0,
            'penalty_task_failure': penalty,
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
                task_reward, reward_terms = self._compute_task_failure_reward()
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
        visible_sats = self._get_handover_candidates(user)
        
        if handover_action > 0:
            if len(visible_sats) >= handover_action:
                target_sat = visible_sats[handover_action - 1]
                reward += self._execute_handover(user, target_sat)
            else:
                # 策略入口已有 action mask；防御性地将越界动作退化为 stay。
                handover_action = 0

        if handover_action == 0:
            if user.serving_satellite >= 0:
                current_visible = self._is_satellite_visible(user, user.serving_satellite)
                if not current_visible:
                    # stay 动作不替策略自动选择目标；旧链路失效后进入阻塞。
                    self._block_user(user, user.serving_satellite)
        
        # ========== 处理任务卸载 ==========
        task = self.user_tasks[user.user_id]
        if task is not None and user.state == UserState.CONNECTED:
            reward += self._execute_offloading(user, task, offload_ratio)
        
            self._pop_user_task(user.user_id)
        return reward

    def _compute_mec_loads(self) -> np.ndarray:
        mec_loads = []
        for server in self.mec_manager.servers.values():
            queue_ratio = float(server.queue_length) / max(
                float(server.config.max_queue_size),
                1.0,
            )
            cpu_utilization = float(np.clip(server.utilization, 0.0, 1.0))
            mec_loads.append(
                float(np.clip(0.5 * queue_ratio + 0.5 * cpu_utilization, 0.0, 1.0))
            )
        return np.asarray(mec_loads, dtype=np.float32)

    def _compute_mec_load_fairness(self) -> float:
        """Measure load fairness among MEC servers with nonzero work."""
        loads = self._compute_mec_loads()
        if len(loads) == 0:
            return 0.0

        active_loads = loads[loads > 1e-6]
        if len(active_loads) < 2:
            return 0.0

        total_load = float(np.sum(active_loads))
        if total_load <= 1e-9:
            return 0.0

        squared_load = float(np.sum(np.square(active_loads)))
        fairness = (total_load * total_load) / (
            float(len(active_loads)) * max(squared_load, 1e-9)
        )
        return float(np.clip(fairness, 0.0, 1.0))

    @staticmethod
    def _jain_fairness(loads: np.ndarray) -> Optional[float]:
        """Compute Jain fairness over all available MEC nodes with nonzero total load."""
        load_array = np.asarray(loads, dtype=np.float64).reshape(-1)
        load_array = load_array[np.isfinite(load_array)]
        if load_array.size == 0:
            return None
        load_array = np.clip(load_array, 0.0, None)
        total_load = float(np.sum(load_array))
        if total_load <= 1e-9:
            return None
        squared_load = float(np.sum(np.square(load_array)))
        fairness = (total_load * total_load) / (
            float(load_array.size) * max(squared_load, 1e-9)
        )
        return float(np.clip(fairness, 0.0, 1.0))

    def _compute_systemwide_jain_mec_load_fairness(self) -> Optional[float]:
        """Return paper metric Jain fairness, including idle available MEC nodes."""
        return self._jain_fairness(self._compute_mec_loads())

    def _compute_load_balance_score(self) -> float:
        """Backward-compatible reward hook for MEC load fairness."""
        return self._compute_mec_load_fairness()

    @staticmethod
    def _load_variance_from_loads(loads: np.ndarray) -> float:
        load_array = np.asarray(loads, dtype=np.float64).reshape(-1)
        load_array = load_array[np.isfinite(load_array)]
        if len(load_array) == 0:
            return 0.0
        mean_load = float(np.mean(load_array))
        return float(np.clip(np.mean((load_array - mean_load) ** 2), 0.0, 0.25))

    def _compute_load_balance_variance(self) -> float:
        """Return the time-point variance of system-wide normalized MEC loads."""
        return self._load_variance_from_loads(self._compute_mec_loads())

    def _record_load_balance_metrics(self) -> None:
        """Record per-step load balance score and active-load variance sample."""
        load_balance_score = self._compute_load_balance_score()
        loads = self._compute_mec_loads()
        self._last_load_balance_score = load_balance_score
        self.stats['load_balance_sum'] += load_balance_score
        self.stats['load_balance_samples'] += 1
        jain_fairness = self._jain_fairness(loads)
        if jain_fairness is not None:
            self.stats['jain_load_fairness_sum'] += jain_fairness
            self.stats['jain_load_fairness_samples'] += 1
        if np.any(loads > 1e-6):
            self.stats['load_variance_samples'].append(
                self._load_variance_from_loads(loads)
            )

    def _compute_task_reward(
        self,
        total_delay: float,
        total_energy: float,
        max_delay: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        QoS 门控任务奖励。

        按 deadline 完成时，以 +1 为基础奖励，再扣归一化时延和能耗；
        未按 deadline 完成时统一返回 -1，不再叠加时延、裕量和超时惩罚。
        """
        max_delay = max(float(max_delay), 1e-6)
        delay_ratio = np.clip(float(total_delay) / max_delay, 0.0, 1.0)

        if float(total_delay) > max_delay:
            return self._compute_task_failure_reward()

        energy_ratio = np.clip(
            float(total_energy) / REWARD_ENERGY_REFERENCE_J,
            0.0,
            1.0,
        )
        penalty_delay = -float(self.config.reward_delay_weight) * float(delay_ratio)
        penalty_energy = -float(self.config.reward_energy_weight) * float(energy_ratio)
        reward = TASK_SUCCESS_REWARD + penalty_delay + penalty_energy

        return reward, {
            'reward_task_success': TASK_SUCCESS_REWARD,
            'penalty_delay': penalty_delay,
            'penalty_energy': penalty_energy,
            'penalty_task_failure': 0.0,
        }

    def _check_handover_link_feasibility(
        self,
        target_sat: VisibilityInfo,
    ) -> Tuple[bool, Optional[str]]:
        """Apply deterministic radio admission checks before any state change."""
        if not target_sat.is_visible:
            return False, 'target_not_visible'
        if target_sat.elevation_deg < self.config.min_elevation_deg:
            return False, 'elevation_below_threshold'
        if target_sat.rvt_seconds < self.config.pre_handover_rvt_sec:
            return False, 'rvt_below_threshold'
        if self.mec_manager.get_server(target_sat.sat_id) is None:
            return False, 'target_server_unavailable'
        snr_db = self.channel.compute_snr_db(
            target_sat.distance_km,
            target_sat.elevation_deg,
        )
        if snr_db < self.config.handover_min_snr_db:
            return False, 'snr_below_threshold'
        return True, None

    def _settle_stranded_tasks(self, user_id: int, sat_id: int) -> None:
        """Fail tasks that can no longer be reached after a forced disconnect."""
        removed_tasks = self.mec_manager.remove_user_tasks(sat_id, user_id)
        for task_info in removed_tasks:
            task_id = int(task_info['task_id'])
            self._offload_task_meta.pop((user_id, task_id), None)
            elapsed = max(
                float(self.current_time) - float(task_info['arrival_time'])
                + float(task_info.get('upload_delay', 0.0)),
                0.0,
            )
            task_reward, reward_terms = self._compute_task_failure_reward()
            self.stats['failed_tasks'] += 1
            self.stats['total_delay'] += elapsed
            self._record_reward_terms(**reward_terms)
            self.pending_rewards[user_id] = (
                self.pending_rewards.get(user_id, 0.0) + task_reward
            )
            self.task_manager.fail_task(task_id, self.current_time)

    def _block_user(self, user: User, old_sat_id: int) -> None:
        """Enter BLOCKED only after the old service link is no longer usable."""
        was_attached = old_sat_id >= 0
        if was_attached:
            old_server = self.mec_manager.get_server(old_sat_id)
            if old_server is not None:
                old_server.remove_user(user.user_id)
            self._settle_stranded_tasks(user.user_id, old_sat_id)
        if user.state != UserState.BLOCKED or user.serving_satellite >= 0:
            self.stats['forced_disconnects'] += 1
        user.serving_satellite = -1
        user.state = UserState.BLOCKED
        user.handover_start_time = -1.0
        user.last_update_time = self.current_time

    def _handover_failure_reward(self) -> float:
        failed_penalty = -self.config.reward_failed_handover_penalty
        self._record_reward_terms(penalty_failed_handover=failed_penalty)
        return failed_penalty
    
    def _execute_handover(self, user: User, target_sat: VisibilityInfo) -> float:
        """
        执行切换
        
        Args:
            user: 用户
            target_sat: 目标卫星信息
            
        Returns:
            切换奖励
        """
        # 防御性语义：目标就是当前服务卫星时等价于 stay，不计切换事件。
        if target_sat.sat_id == user.serving_satellite:
            return 0.0

        old_sat_id = int(user.serving_satellite)
        is_reconnection = old_sat_id < 0 or user.state == UserState.BLOCKED
        old_link_valid = (
            not is_reconnection
            and self._is_satellite_visible(user, old_sat_id)
        )
        if is_reconnection:
            self.stats['reconnection_attempts'] += 1
        else:
            self.stats['total_handovers'] += 1
            self.stats['handover_attempts'] += 1

        link_feasible, failure_reason = self._check_handover_link_feasibility(
            target_sat
        )
        if not link_feasible:
            if is_reconnection:
                self.stats['handover_radio_failures'] += 1
            else:
                self.stats['failed_handovers'] += 1
                self.stats['handover_aborted'] += 1
                self.stats['handover_radio_failures'] += 1
                user.handover_count += 1
                user.failed_handovers += 1
            if not old_link_valid and old_sat_id >= 0:
                self._block_user(user, old_sat_id)
            return self._handover_failure_reward()

        migration_plan = self.mec_manager.prepare_user_task_migration(
            user_id=user.user_id,
            old_sat_id=old_sat_id,
            new_sat_id=target_sat.sat_id,
        )
        if not migration_plan.feasible:
            self.stats['migration_rejections'] += 1
            if not is_reconnection:
                self.stats['failed_handovers'] += 1
                self.stats['handover_aborted'] += 1
                user.handover_count += 1
                user.failed_handovers += 1
            if not old_link_valid and old_sat_id >= 0:
                self._block_user(user, old_sat_id)
            return self._handover_failure_reward()

        migration_result = self.mec_manager.commit_user_task_migration(
            migration_plan,
            handover_delay=self.config.handover_delay_sec,
        )
        if migration_result['failed'] > 0 or migration_result['failure_reason']:
            self.stats['migration_rejections'] += 1
            if not is_reconnection:
                self.stats['failed_handovers'] += 1
                self.stats['handover_aborted'] += 1
                user.handover_count += 1
                user.failed_handovers += 1
            if not old_link_valid and old_sat_id >= 0:
                self._block_user(user, old_sat_id)
            return self._handover_failure_reward()

        old_server = (
            self.mec_manager.get_server(old_sat_id)
            if old_sat_id >= 0 else None
        )
        new_server = self.mec_manager.get_server(target_sat.sat_id)
        if old_server is not None:
            old_server.remove_user(user.user_id)
        if new_server is not None:
            new_server.add_user(user.user_id)

        if is_reconnection:
            user.connect_to_satellite(target_sat.sat_id, self.current_time)
            self.stats['reconnections'] += 1
        else:
            user.start_handover(target_sat.sat_id, self.current_time)
            user.complete_handover(
                target_sat.sat_id,
                self.current_time,
                success=True,
            )
            self.stats['successful_handovers'] += 1
            self.stats['handover_committed'] += 1

        self._step_handover_interruption_seconds[user.user_id] += float(
            self.config.handover_delay_sec
        )
        return 0.0
    
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
            本步可立即发放的任务结算奖励；MEC 入队任务完成后延迟发放
        """
        reward = 0.0
        offload_ratio = float(np.clip(offload_ratio, 0.0, 1.0))
        if offload_ratio < self.config.min_effective_offload_ratio:
            offload_ratio = 0.0
        wait_delay = max(float(self.current_time) - float(task.creation_time), 0.0)
        
        # 获取卫星信息
        sat_id = user.serving_satellite
        if sat_id < 0:
            raise RuntimeError("已连接用户缺少服务卫星")
        
        # 获取链路信息
        vis_info = self._get_satellite_visibility(user, sat_id)
        if vis_info is None or not vis_info.is_visible:
            raise RuntimeError("任务卸载前服务链路已失效")
        
        server = self.mec_manager.get_server(sat_id)
        if server is None:
            raise RuntimeError(f"服务卫星 {sat_id} 缺少 MEC 服务器")
        
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
                
                if total_delay <= task.max_delay:
                    self.stats['completed_tasks'] += 1
                    self.stats['successful_task_delay_samples'].append(total_delay)
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
                self.stats['successful_task_delay_samples'].append(total_delay)
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
        slot_start = float(self.current_time)
        slot_duration = float(self.config.time_step_sec)
        
        # 传播星座
        self.constellation.propagate(slot_duration)
        self.geometry_version += 1
        self._invalidate_visibility_cache()
        
        # ========== 处理所有卫星的任务队列 ==========
        completed_tasks = self.mec_manager.process_all_queues(
            slot_start, slot_duration
        )

        # 当前状态在本 slot 结束时刻。
        self.current_time = slot_start + slot_duration
        self._expire_pending_user_tasks()
        
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
                self.stats['successful_task_delay_samples'].append(total_delay)
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
                    self._block_user(user, stale_sat_id)
    
    def _get_visible_satellites(self, user: User) -> List[VisibilityInfo]:
        """获取用户可见的卫星列表（带缓存）"""
        if self._visibility_cache_version != self.geometry_version:
            self._invalidate_visibility_cache()
        uid = user.user_id
        if uid in self._visibility_cache:
            return self._visibility_cache[uid]
        result = self._compute_visibility_batch(uid)
        self._visibility_cache[uid] = result
        return result

    def _get_handover_candidates(self, user: User) -> List[VisibilityInfo]:
        """返回可切换目标，明确排除当前服务卫星。"""
        current_satellite = int(user.serving_satellite)
        return [
            item
            for item in self._get_visible_satellites(user)
            if item.sat_id != current_satellite
        ]
    
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
        """估算当前几何状态下指定卫星的剩余可见时间。"""
        return float(
            self._estimate_rvt_batch(
                user.user_id,
                np.asarray([sat_id], dtype=np.int64),
                np.asarray([current_elevation], dtype=np.float64),
            )[0]
        )

    def get_pre_handover_mask(self) -> np.ndarray:
        """
        获取 pre-handover 掩码

        返回 bool 数组，True 表示该用户需要切换（RVT 低或无连接），
        False 表示安全用户（应强制 handover_action=0）。
        """
        mask = np.zeros(self.num_users, dtype=bool)
        threshold = float(getattr(self.config, "pre_handover_rvt_sec", 30.0))
        for uid, user in enumerate(self.user_manager.users):
            if user.serving_satellite < 0:
                mask[uid] = True
                continue
            vis = self._get_satellite_visibility(user, user.serving_satellite)
            if vis is None or not vis.is_visible:
                mask[uid] = True
                continue
            if float(vis.rvt_seconds) < threshold:
                mask[uid] = True
                continue
            server = self.mec_manager.get_server(user.serving_satellite)
            if server is not None and float(server.utilization) >= 0.95:
                mask[uid] = True
        return mask

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
            'mec_load_fairness': self._last_load_balance_score,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'user_rewards': self.last_user_rewards.copy(),
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
    
# ==================== 便捷函数 ====================

def make_env(config: Optional[EnvConfig] = None, **kwargs) -> LEOSatelliteEnv:
    """创建环境的便捷函数"""
    if config is None:
        config = EnvConfig(**kwargs)
    return LEOSatelliteEnv(config)
