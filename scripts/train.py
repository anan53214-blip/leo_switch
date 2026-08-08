"""
LEO卫星网络切换与任务卸载联合优化 - 完整训练脚本
===============================================

本脚本整合 HAN (异质图注意力网络) + MAPPO (多智能体PPO) 进行训练。

【训练流程】
1. 环境初始化：创建LEO卫星仿真环境
2. 图构建：构建异质图，提取节点/边特征  
3. HAN编码：使用HAN获取节点嵌入
4. MAPPO决策：基于嵌入进行多智能体决策
5. 环境交互：执行动作，收集经验
6. 策略更新：使用PPO更新Actor/Critic

【使用方法】
```bash
# 基本训练
python scripts/train.py

# 指定参数
python scripts/train.py --num_users 10 --total_timesteps 150000 --seed 42

# 从检查点恢复
python scripts/train.py --load_path results/models/checkpoint_100000.pt
```

【关键组件】
- LEOSatelliteEnv: Gymnasium环境
- HeteroGraphBuilder: 异质图构建
- HANEncoder: 异质图注意力网络
- MAPPO: 多智能体PPO算法
"""

import sys
import time
import argparse
import logging
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from scripts.load_balance_metrics import normalize_load_balance_metrics
from scripts.paper_metrics import derive_paper_metrics

# 导入项目模块
from src.environment.gym_env import (
    LEOSatelliteEnv,
    EnvConfig,
    REWARD_BREAKDOWN_KEYS,
    build_env_config,
    summarize_env_stats,
)
from src.features.satellite_load import (
    SATELLITE_CONTEXT_FEATURE_DIM,
    SATELLITE_LOAD_FEATURE_DIM,
    SHARED_CONSTRAINT_DIM,
    build_satellite_context_features,
    build_satellite_load_features,
    build_shared_constraint_vector,
)
from src.graph.builder import HeteroGraphBuilder
from src.graph.features import FeatureExtractor
from src.model.hetero_gnn import HANEncoder, HANConfig
from src.algorithm.mappo import MAPPO, MAPPOConfig
from src.algorithm.attention_mappo import AttentionMAPPO
from src.algorithm.buffer import MultiAgentRolloutBuffer
from src.algorithm.replay_buffer import MultiAgentReplayBuffer
from src.algorithm.maddpg import MADDPGAlgorithm, MADDPGConfig
from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig


MODEL_SCHEMA_VERSION = 5
GEOMETRY_SCHEMA_VERSION = 3
ENVIRONMENT_SCHEMA_VERSION = 13
EVALUATION_SEED_OFFSET = 1_000_000


def reward_breakdown(stats: Dict[str, Any]) -> Dict[str, float]:
    """提取当前 reward 结构的全部分项。"""
    return {
        key: float(stats.get(key, 0.0))
        for key in REWARD_BREAKDOWN_KEYS
    }


BEST_MODEL_METRIC_CHOICES = (
    "reward",
    "avg_delay",
    "avg_success_delay",
    "p95_success_delay",
    "total_energy",
    "successful_task_throughput",
    "service_availability_rate",
    "handover_failure_rate",
    "load_balance_coefficient",
    "load_balance_variance",
    "mec_load_fairness",
    "jain_mec_load_fairness",
    "avg_load_balance_score",
    "energy_per_successful_task",
    "task_completion_rate",
    "task_success_rate",
    "task_failure_rate",
    "task_settlement_rate",
)

BEST_MODEL_METRIC_LABELS = {
    "reward": "reward",
    "avg_delay": "average delay",
    "avg_success_delay": "successful-task average delay",
    "p95_success_delay": "successful-task P95 delay",
    "total_energy": "energy per resolved task",
    "successful_task_throughput": "successful task throughput",
    "service_availability_rate": "service availability",
    "handover_failure_rate": "handover failure rate",
    "load_balance_coefficient": "load balance coefficient",
    "load_balance_variance": "load balance variance",
    "mec_load_fairness": "MEC load fairness",
    "jain_mec_load_fairness": "MEC load Jain fairness",
    "avg_load_balance_score": "load balance",
    "energy_per_successful_task": "energy per successful task",
    "task_completion_rate": "task completion",
    "task_success_rate": "task success",
    "task_failure_rate": "task failure rate",
    "task_settlement_rate": "task settlement",
}


def best_model_metric_label(metric_name: str) -> str:
    """Return a readable label for checkpoint-selection metrics."""
    return BEST_MODEL_METRIC_LABELS.get(metric_name, metric_name)


def energy_per_resolved_task(record: Dict[str, Any]) -> float:
    total_energy = float(record.get("total_energy", 0.0))
    resolved_tasks = max(float(record.get("resolved_tasks", 0.0)), 1.0)
    return total_energy / resolved_tasks


def energy_per_successful_task(record: Dict[str, Any]) -> float:
    total_energy = float(record.get("total_energy", 0.0))
    completed_tasks = max(float(record.get("completed_tasks", 0.0)), 1.0)
    return total_energy / completed_tasks


def summarize_env_stats_with_load_balance(env_stats: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize_env_stats(env_stats)
    if "load_variance_samples" in env_stats:
        summary["load_variance_samples"] = env_stats.get("load_variance_samples", [])
    return derive_paper_metrics(normalize_load_balance_metrics(summary))


def _bounded_unit_score(value: Any) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def compute_model_selection_score(record: Dict[str, Any], metric_name: str) -> float:
    """Convert an evaluation record into a higher-is-better selection score."""
    metric_name = metric_name or "reward"

    if metric_name == "reward":
        return float(record.get("eval_mean_reward", record.get("mean_reward", 0.0)))
    if metric_name == "avg_delay":
        return -float(record.get("avg_delay", float("inf")))
    if metric_name == "avg_success_delay":
        if float(record.get("completed_tasks", 0.0)) <= 0.0:
            return float("-inf")
        return -float(record.get("avg_success_delay", float("inf")))
    if metric_name == "p95_success_delay":
        if float(record.get("completed_tasks", 0.0)) <= 0.0:
            return float("-inf")
        return -float(record.get("p95_success_delay", float("inf")))
    if metric_name == "total_energy":
        return -energy_per_resolved_task(record)
    if metric_name == "energy_per_successful_task":
        if float(record.get("completed_tasks", 0.0)) <= 0.0:
            return float("-inf")
        return -energy_per_successful_task(record)
    if metric_name == "successful_task_throughput":
        return float(derive_paper_metrics(record).get("successful_task_throughput", 0.0))
    if metric_name == "service_availability_rate":
        return _bounded_unit_score(record.get("service_availability_rate", 0.0))
    if metric_name == "handover_failure_rate":
        return -_bounded_unit_score(record.get("handover_failure_rate", 0.0))
    if metric_name == "load_balance_coefficient":
        return _bounded_unit_score(
            normalize_load_balance_metrics(record).get("load_balance_coefficient", 0.0)
        )
    if metric_name == "load_balance_variance":
        return -float(record.get("load_balance_variance", 0.0))
    if metric_name == "avg_load_balance_score":
        return _bounded_unit_score(
            normalize_load_balance_metrics(record).get("avg_load_balance_score", 0.0)
        )
    if metric_name == "mec_load_fairness":
        return _bounded_unit_score(
            normalize_load_balance_metrics(record).get("mec_load_fairness", 0.0)
        )
    if metric_name == "jain_mec_load_fairness":
        return _bounded_unit_score(
            derive_paper_metrics(record).get("jain_mec_load_fairness", 0.0)
        )
    if metric_name == "task_completion_rate":
        return _bounded_unit_score(record.get("task_completion_rate", 0.0))
    if metric_name == "task_success_rate":
        return _bounded_unit_score(record.get("task_success_rate", record.get("task_completion_rate", 0.0)))
    if metric_name == "task_failure_rate":
        return -_bounded_unit_score(record.get("task_failure_rate", record.get("deadline_violation_rate", 0.0)))
    if metric_name == "task_settlement_rate":
        return _bounded_unit_score(record.get("task_settlement_rate", record.get("task_resolution_rate", 0.0)))
    raise ValueError(f"Unsupported best-model metric: {metric_name}")


# ============================================================
# 配置类
# ============================================================

@dataclass
class TrainConfig:
    """
    完整训练配置
    
    整合环境、HAN、MAPPO、训练等所有参数
    """
    # ---------- 实验信息 ----------
    exp_name: str = "han_mappo_delay_focus"      # 实验名称
    seed: int = 42                        # 随机种子
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---------- 环境参数 ----------
    num_planes: int = EnvConfig.num_planes
    sats_per_plane: int = EnvConfig.sats_per_plane
    altitude_km: float = EnvConfig.altitude_km
    inclination_deg: float = EnvConfig.inclination_deg
    num_users: int = EnvConfig.num_users
    user_center_lat: float = EnvConfig.user_center_lat
    user_center_lon: float = EnvConfig.user_center_lon
    user_radius_deg: float = EnvConfig.user_radius_deg
    resample_users_on_reset: bool = EnvConfig.resample_users_on_reset
    max_steps: int = EnvConfig.max_steps
    time_step_sec: float = EnvConfig.time_step_sec
    randomize_episode_start: bool = EnvConfig.randomize_episode_start
    episode_start_time_jitter_sec: float = EnvConfig.episode_start_time_jitter_sec
    min_elevation_deg: float = EnvConfig.min_elevation_deg
    handover_delay_sec: float = EnvConfig.handover_delay_sec
    rvt_threshold_sec: float = EnvConfig.rvt_threshold_sec
    task_arrival_prob: float = EnvConfig.task_arrival_prob
    task_arrival_seed_offset: int = EnvConfig.task_arrival_seed_offset
    carrier_frequency_ghz: float = EnvConfig.carrier_frequency_ghz
    bandwidth_mhz: float = EnvConfig.bandwidth_mhz
    satellite_tx_power_dbm: float = EnvConfig.satellite_tx_power_dbm
    satellite_antenna_gain_db: float = EnvConfig.satellite_antenna_gain_db
    user_tx_power_dbm: float = EnvConfig.user_tx_power_dbm
    user_antenna_gain_db: float = EnvConfig.user_antenna_gain_db
    user_pa_efficiency: float = EnvConfig.user_pa_efficiency
    user_circuit_power_w: float = EnvConfig.user_circuit_power_w
    ofdma_uplink_sharing: bool = EnvConfig.ofdma_uplink_sharing
    noise_temperature_k: float = EnvConfig.noise_temperature_k
    noise_figure_db: float = EnvConfig.noise_figure_db
    rain_attenuation_db: float = EnvConfig.rain_attenuation_db
    atmospheric_loss_db: float = EnvConfig.atmospheric_loss_db
    pointing_loss_db: float = EnvConfig.pointing_loss_db
    polarization_loss_db: float = EnvConfig.polarization_loss_db
    implementation_loss_db: float = EnvConfig.implementation_loss_db
    satellite_cpu_freq_ghz: float = EnvConfig.satellite_cpu_freq_ghz
    satellite_max_cpu_freq_ghz: float = EnvConfig.satellite_max_cpu_freq_ghz
    satellite_num_cores: int = EnvConfig.satellite_num_cores
    mec_max_concurrent_tasks: int = EnvConfig.mec_max_concurrent_tasks
    max_queue_size: int = EnvConfig.max_queue_size
    user_cpu_freq_ghz: float = EnvConfig.user_cpu_freq_ghz
    user_max_cpu_freq_ghz: float = EnvConfig.user_max_cpu_freq_ghz
    kappa: float = EnvConfig.kappa
    user_idle_power_w: float = EnvConfig.user_idle_power_w
    result_ratio: float = EnvConfig.result_ratio
    light_data_range: tuple = EnvConfig.light_data_range
    light_compute_range: tuple = EnvConfig.light_compute_range
    light_delay_range: tuple = EnvConfig.light_delay_range
    medium_data_range: tuple = EnvConfig.medium_data_range
    medium_compute_range: tuple = EnvConfig.medium_compute_range
    medium_delay_range: tuple = EnvConfig.medium_delay_range
    heavy_data_range: tuple = EnvConfig.heavy_data_range
    heavy_compute_range: tuple = EnvConfig.heavy_compute_range
    heavy_delay_range: tuple = EnvConfig.heavy_delay_range
    min_effective_offload_ratio: float = EnvConfig.min_effective_offload_ratio
    reward_delay_weight: float = EnvConfig.reward_delay_weight
    reward_energy_weight: float = EnvConfig.reward_energy_weight
    reward_energy_reference_j: float = EnvConfig.reward_energy_reference_j
    reward_interruption_weight: float = EnvConfig.reward_interruption_weight
    reward_failed_handover_penalty: float = EnvConfig.reward_failed_handover_penalty
    reward_load_balance_weight: float = EnvConfig.reward_load_balance_weight
    pre_handover_rvt_sec: float = EnvConfig.pre_handover_rvt_sec
    handover_min_snr_db: float = EnvConfig.handover_min_snr_db

    # ---------- 图参数 ----------
    max_visible_sats: int = 10            # 最大可见卫星数（候选集）
    
    # ---------- HAN参数 ----------
    han_hidden_dim: int = 64              # HAN隐藏维度
    han_out_dim: int = 64                 # HAN输出维度
    han_num_heads: int = 4                # 注意力头数
    han_num_layers: int = 2               # HAN层数
    # PPO rollout/update 的 log-prob 必须可复现，HAN 训练路径禁用 Dropout。
    han_dropout: float = 0.0
    
    # ---------- Actor参数 ----------
    actor_hidden_dims: tuple = (256, 128)  # Actor隐藏层
    
    # ---------- Critic参数 ----------
    critic_hidden_dims: tuple = (256, 256, 128)  # Critic隐藏层
    
    # ---------- MAPPO参数 ----------
    algorithm: str = "mappo"
    learning_rate: float = 1e-4           # 保守更新，降低中后期策略震荡
    gamma: float = 0.99                   # 折扣因子
    gae_lambda: float = 0.95              # GAE参数
    clip_range: float = 0.2               # PPO clip
    clip_range_vf: Optional[float] = 0.2  # 价值函数clip（抑制Critic剧烈波动）
    value_loss_coef: float = 0.5          # 值函数损失系数
    value_loss_type: str = "huber"       # 值函数损失类型: mse/huber
    normalize_returns: bool = True        # 是否标准化returns后再计算value loss
    value_huber_beta: float = 10.0        # Huber损失beta（仅value_loss_type=huber时有效）
    entropy_coef: float = 0.005            # Keep exploration from collapsing too early
    max_grad_norm: float = 0.5            # 梯度裁剪
    entropy_schedule: str = "constant"    # Entropy schedule: constant / linear
    n_epochs: int = 4                     # 降低同一 rollout 的重复更新强度
    batch_size: int = 512                 # PPO mini-batch size
    maddpg_actor_lr: float = 5e-4
    maddpg_critic_lr: float = 1e-3
    pdqn_lr: float = 1e-3
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    noise_start: float = 0.35
    noise_final: float = 0.05
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay_fraction: float = 0.25
    bc_loss_coef: float = 0.0
    safe_exploration_probability: float = 0.0
    target_update_interval: int = 500
    
    # ---------- 训练参数 ----------
    total_timesteps: int = 150_000        # 总训练步数
    n_steps: int = 1024                   # 每次更新收集步数
    eval_interval: int = 25_000           # 评估间隔
    eval_episodes: int = 5                # 评估episode数
    graph_update_interval: int = 1        # 每步更新动态图
    save_interval: int = 50_000           # 检查点保存间隔
    log_interval: int = 1                 # 日志间隔
    
    # ---------- 路径参数 ----------
    save_path: str = "results/full_train_delay_focus"  # 模型保存路径
    log_path: str = "results/logs"        # 日志路径
    
    # ---------- 加载参数 ----------
    load_path: Optional[str] = None       # 加载检查点路径
    pretrained_han_path: Optional[str] = None  # off-policy 方法使用的已训练 HAN
    
    # ---------- Early Stopping ----------
    early_stop_patience: int = 0          # 连续N次更新无改善则停止（0=禁用）
    best_model_metric: str = "reward"  # best_model.pt 统一按验证集 mean reward 选优


# ============================================================
# HAN-MAPPO 训练器
# ============================================================

class HANMAPPOTrainer:
    """
    HAN + MAPPO 联合训练器
    
    【训练架构】
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      训练循环                                │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   环境状态 ──────► 图构建器 ──────► HAN编码器               │
    │       │              │                 │                    │
    │       │              ▼                 ▼                    │
    │       │         异质图数据        节点嵌入                   │
    │       │                               │                    │
    │       │           ┌───────────────────┘                    │
    │       │           │                                        │
    │       │           ▼                                        │
    │       │       MAPPO策略                                    │
    │       │           │                                        │
    │       │           ▼                                        │
    │       │       动作 (切换+卸载)                              │
    │       │           │                                        │
    │       └──────────►│◄── 执行动作                            │
    │                   │                                        │
    │                   ▼                                        │
    │               奖励 + 新状态                                 │
    │                   │                                        │
    │                   ▼                                        │
    │               缓冲区存储                                    │
    │                   │                                        │
    │                   ▼                                        │
    │               策略更新 (PPO)                                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    ```
    """
    
    def __init__(self, config: TrainConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 设置日志
        self._setup_logging()

        # 所有 on-policy 训练器共享 collect_rollouts，但并非所有变体都会
        # 构建 HAN 图。先建立统一的缓存契约，避免无图子类在首个 rollout
        # 访问尚不存在的属性。
        self._cached_han_user_embed = None
        self._cached_sat_embed = None
        self._cached_graph_snapshot = None
        
        # 创建目录
        self._create_directories()
        
        # 初始化各组件
        self._init_environment()
        self._init_graph_builder()
        self._init_han_encoder()
        self._init_mappo()
        self._init_buffer()
        
        # 训练统计
        self.total_steps = 0
        self.episodes = 0
        self.best_reward = float('-inf')
        self.best_model_score = float('-inf')
        self.training_start_time = None
        
        # Episode统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = []  # 最近100个episode
        
        # 训练历史记录（用于可视化）
        self.training_history: List[Dict] = []
        self.eval_history: List[Dict] = []
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        self.logger.info(f"环境: {config.num_users} 用户, {config.num_planes * config.sats_per_plane} 卫星")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.config.exp_name}_{timestamp}.log"

        logger_name = f"{__name__}.{self.config.exp_name}.{timestamp}.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.log_file = log_file
    
    def _create_directories(self):
        """创建必要目录"""
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_path).mkdir(parents=True, exist_ok=True)
    
    def _init_environment(self):
        """初始化环境"""
        self.logger.info("初始化环境...")
        
        self.env = LEOSatelliteEnv(build_env_config(self.config))
        
        # 提取环境信息
        self.num_agents = self.config.num_users
        self.max_candidates = self.config.max_visible_sats
        
        # 从环境获取实际观测维度
        # 环境的观测空间是 (num_users, user_obs_dim)
        self.raw_obs_dim = self.env.user_obs_dim
        self.han_out_dim = self.config.han_out_dim
        # Raw observation already contains serving-link RVT and task features.
        # Do not append a second, differently normalized copy around HAN.
        self.obs_dim = self.raw_obs_dim + self.han_out_dim
        
        # 全局状态维度 (所有用户观测拼接)
        self.global_state_dim = self.num_agents * self.obs_dim
        
        self.logger.info(f"  - 原始观测维度: {self.raw_obs_dim}")
        self.logger.info(f"  - HAN嵌入维度: {self.han_out_dim}")
        self.logger.info(
            f"  - 拼接后观测维度: {self.obs_dim} "
            f"(raw {self.raw_obs_dim} + HAN {self.han_out_dim})"
        )
        self.logger.info(f"  - 全局状态维度: {self.global_state_dim}")
    
    def _create_eval_env(self) -> LEOSatelliteEnv:
        """Create an isolated environment for evaluation episodes."""
        return LEOSatelliteEnv(copy.deepcopy(self.env.config))

    def _agent_reward_vector(self, rewards) -> np.ndarray:
        """Return per-user local rewards plus the shared cooperative term."""
        local_rewards = getattr(self.env, "last_user_rewards", None)
        if local_rewards is not None:
            reward_array = np.asarray(local_rewards, dtype=np.float32)
            if reward_array.shape == (self.num_agents,):
                return reward_array.copy()
        if isinstance(rewards, dict):
            shared = float(np.mean(list(rewards.values()))) if rewards else 0.0
        elif isinstance(rewards, (int, float)):
            shared = float(rewards)
        else:
            shared = float(np.mean(np.asarray(rewards, dtype=np.float32)))
        return np.full(self.num_agents, shared, dtype=np.float32)

    def _init_graph_builder(self):
        """初始化图构建器"""
        self.logger.info("初始化图构建器...")
        
        self.feature_extractor = FeatureExtractor()
        self.graph_builder = HeteroGraphBuilder(
            feature_extractor=self.feature_extractor,
            add_reverse_edges=True
        )
    
    def _init_han_encoder(self):
        """初始化HAN编码器"""
        self.logger.info("初始化HAN编码器...")
        
        han_config = HANConfig(
            satellite_in_dim=10,
            user_in_dim=16,
            hidden_dim=self.config.han_hidden_dim,
            out_dim=self.config.han_out_dim,
            num_heads=self.config.han_num_heads,
            num_layers=self.config.han_num_layers,
            dropout=self.config.han_dropout,
            use_edge_features=True,
            user_sat_edge_dim=5,
            isl_edge_dim=3
        )
        
        self.han_encoder = HANEncoder(han_config).to(self.device)
        
        # 统计参数量
        han_params = sum(p.numel() for p in self.han_encoder.parameters())
        self.logger.info(f"  - HAN参数量: {han_params:,}")
    
    def _init_mappo(self):
        """初始化MAPPO算法"""
        self.logger.info("初始化MAPPO...")
        
        mappo_config = MAPPOConfig(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            max_candidates=self.max_candidates,
            sat_embed_dim=self.han_out_dim,  # 卫星嵌入维度 = HAN输出维度
            min_effective_offload_ratio=self.config.min_effective_offload_ratio,
            actor_hidden_dims=list(self.config.actor_hidden_dims),
            critic_hidden_dims=list(self.config.critic_hidden_dims),
            actor_dropout=0.0,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            value_loss_coef=self.config.value_loss_coef,
            value_loss_type=self.config.value_loss_type,
            normalize_returns=self.config.normalize_returns,
            value_huber_beta=self.config.value_huber_beta,
            entropy_coef=self.config.entropy_coef,
            entropy_schedule=self.config.entropy_schedule,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.config.device
        )
        
        self.mappo = MAPPO(mappo_config)
        self.mappo.set_representation_module(
            self.han_encoder,
            self._reencode_mappo_batch,
            learning_rate=self.config.learning_rate,
        )
        
        # 统计参数量
        actor_params = sum(p.numel() for p in self.mappo.actor.parameters())
        critic_params = sum(p.numel() for p in self.mappo.critic.parameters())
        self.logger.info(f"  - Actor参数量: {actor_params:,}")
        self.logger.info(f"  - Critic参数量: {critic_params:,}")
    
    def _init_buffer(self):
        """初始化经验缓冲区"""
        self.logger.info("初始化经验缓冲区...")
        
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=self.config.n_steps,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            max_candidates=self.max_candidates,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.config.device
        )
        
        # 关联buffer到MAPPO
        self.mappo.set_buffer(self.buffer)
    
    def _invalidate_han_cache(self) -> None:
        self._cached_han_user_embed = None
        self._cached_sat_embed = None
        self._cached_graph_snapshot = None

    def _encode_graph_snapshot(self, graph):
        """在当前设备上编码一个不可变图快照。"""
        node_features = {
            node_type: torch.as_tensor(
                features,
                dtype=torch.float32,
                device=self.device,
            )
            for node_type, features in graph.node_features.items()
        }
        edge_index = {
            edge_type: (
                torch.as_tensor(
                    indices[0],
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.as_tensor(
                    indices[1],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            for edge_type, indices in graph.edge_index.items()
        }
        edge_features = {
            edge_type: torch.as_tensor(
                features,
                dtype=torch.float32,
                device=self.device,
            )
            for edge_type, features in graph.edge_features.items()
        }
        return self.han_encoder(
            node_features,
            edge_index,
            edge_features,
        )

    def _encode_graph_snapshots(
        self,
        graphs: Sequence[Any],
    ) -> Dict[str, tuple[torch.Tensor, ...]]:
        """将多个互不连通的图快照合并为一次 HAN 前向。

        边索引按各图的节点偏移量平移，因此批图中不存在跨时间步的边。
        ``node_batch`` 让语义注意力仍按原图分别归一化，输出与逐图前向
        保持相同语义。
        """
        if not graphs:
            raise ValueError("graphs must contain at least one snapshot")

        first_graph = graphs[0]
        node_types = tuple(first_graph.node_features)
        edge_types = tuple(first_graph.edge_index)
        node_feature_parts = {node_type: [] for node_type in node_types}
        node_batch_parts = {node_type: [] for node_type in node_types}
        node_sizes = {node_type: [] for node_type in node_types}
        edge_src_parts = {edge_type: [] for edge_type in edge_types}
        edge_dst_parts = {edge_type: [] for edge_type in edge_types}
        edge_feature_parts = {edge_type: [] for edge_type in edge_types}
        node_offsets = {node_type: 0 for node_type in node_types}

        for graph_index, graph in enumerate(graphs):
            if tuple(graph.node_features) != node_types:
                raise ValueError("graph snapshots have inconsistent node types")
            if tuple(graph.edge_index) != edge_types:
                raise ValueError("graph snapshots have inconsistent edge types")
            if set(graph.edge_features) != set(edge_types):
                raise ValueError("graph snapshot is missing edge features")

            graph_offsets = dict(node_offsets)
            for node_type in node_types:
                features = torch.as_tensor(
                    graph.node_features[node_type],
                    dtype=torch.float32,
                    device=self.device,
                )
                count = int(features.shape[0])
                node_feature_parts[node_type].append(features)
                node_batch_parts[node_type].append(
                    torch.full(
                        (count,),
                        graph_index,
                        dtype=torch.long,
                        device=self.device,
                    )
                )
                node_sizes[node_type].append(count)
                node_offsets[node_type] += count

            for edge_type in edge_types:
                src_type, _, dst_type = edge_type
                src_indices, dst_indices = graph.edge_index[edge_type]
                edge_src_parts[edge_type].append(
                    torch.as_tensor(
                        src_indices,
                        dtype=torch.long,
                        device=self.device,
                    ) + graph_offsets[src_type]
                )
                edge_dst_parts[edge_type].append(
                    torch.as_tensor(
                        dst_indices,
                        dtype=torch.long,
                        device=self.device,
                    ) + graph_offsets[dst_type]
                )
                edge_feature_parts[edge_type].append(
                    torch.as_tensor(
                        graph.edge_features[edge_type],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        node_features = {
            node_type: torch.cat(parts, dim=0)
            for node_type, parts in node_feature_parts.items()
        }
        node_batch = {
            node_type: torch.cat(parts, dim=0)
            for node_type, parts in node_batch_parts.items()
        }
        edge_index = {
            edge_type: (
                torch.cat(edge_src_parts[edge_type], dim=0),
                torch.cat(edge_dst_parts[edge_type], dim=0),
            )
            for edge_type in edge_types
        }
        edge_features = {
            edge_type: torch.cat(edge_feature_parts[edge_type], dim=0)
            for edge_type in edge_types
        }
        embeddings = self.han_encoder(
            node_features,
            edge_index,
            edge_features,
            node_batch=node_batch,
            num_graphs=len(graphs),
        )
        return {
            node_type: torch.split(
                embeddings[node_type],
                node_sizes[node_type],
                dim=0,
            )
            for node_type in node_types
        }

    def _reencode_mappo_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """用一次批图 HAN 前向重算 PPO minibatch 的图表示。"""
        time_indices = batch["time_indices"].detach().cpu().tolist()
        unique_time_indices = list(dict.fromkeys(time_indices))
        graphs = []
        for time_index in unique_time_indices:
            graph = self.buffer.graph_snapshots[time_index]
            if graph is None:
                raise RuntimeError(
                    f"rollout 的第 {time_index} 步缺少图快照"
                )
            graphs.append(graph)

        node_embeddings = self._encode_graph_snapshots(graphs)
        stored_observations = torch.as_tensor(
            self.buffer.observations[unique_time_indices],
            dtype=torch.float32,
            device=self.device,
        )
        user_embeddings = torch.stack(
            node_embeddings["user"],
            dim=0,
        )
        reencoded_satellite_embeddings = torch.stack(
            node_embeddings["satellite"],
            dim=0,
        )
        stored_satellite_embeddings = torch.as_tensor(
            self.buffer.satellite_embeddings[unique_time_indices],
            dtype=torch.float32,
            device=self.device,
        )
        satellite_suffix = stored_satellite_embeddings[
            :,
            :,
            self.config.han_out_dim:,
        ]
        satellite_embeddings = torch.cat(
            [reencoded_satellite_embeddings, satellite_suffix],
            dim=2,
        )
        observation_suffix = stored_observations[
            :,
            :,
            self.raw_obs_dim + self.config.han_out_dim:,
        ]
        policy_observations = torch.cat(
            [
                stored_observations[:, :, :self.raw_obs_dim],
                user_embeddings,
                observation_suffix,
            ],
            dim=2,
        )

        time_position_by_index = {
            time_index: position
            for position, time_index in enumerate(unique_time_indices)
        }
        time_positions = torch.as_tensor(
            [
                time_position_by_index[time_index]
                for time_index in time_indices
            ],
            dtype=torch.long,
            device=self.device,
        )
        agent_indices = batch["agent_indices"].long()
        batch["observations"] = policy_observations[
            time_positions,
            agent_indices,
        ]
        batch["satellite_embeddings"] = satellite_embeddings[time_positions]
        batch["global_states"] = policy_observations.flatten(1)[time_positions]
        return batch

    def _raw_policy_observations(self) -> np.ndarray:
        raw_observations = np.asarray(self.env._get_observation(), dtype=np.float32)
        if raw_observations.ndim == 1:
            raw_observations = raw_observations.reshape(1, -1)
        expected_shape = (self.num_agents, self.raw_obs_dim)
        if raw_observations.shape != expected_shape:
            padded = np.zeros(expected_shape, dtype=np.float32)
            copy_rows = min(raw_observations.shape[0], self.num_agents)
            copy_cols = min(raw_observations.shape[1], self.raw_obs_dim)
            padded[:copy_rows, :copy_cols] = raw_observations[:copy_rows, :copy_cols]
            raw_observations = padded
        return raw_observations.astype(np.float32, copy=False)

    def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构建异质图并使用 HAN 编码，返回用户嵌入、卫星嵌入、候选动作掩码和候选卫星ID。

        性能优化：HAN编码每 graph_update_interval 步才重新计算一次，
        中间步骤直接刷新原始观测、动作掩码和候选卫星ID。
        """
        graph_update_interval = max(1, int(self.config.graph_update_interval))
        
        need_full_update = (
            not hasattr(self, '_cached_han_user_embed') or
            self._cached_han_user_embed is None or
            self.total_steps % graph_update_interval == 0
        )
        
        if need_full_update:
            graph = self.graph_builder.build(self.env)
            with torch.no_grad():
                node_embeddings = self._encode_graph_snapshot(graph)
            self._cached_han_user_embed = node_embeddings['user'].detach().cpu().numpy().astype(np.float32)
            self._cached_sat_embed = node_embeddings['satellite'].detach().cpu().numpy().astype(np.float32)
            self._cached_graph_snapshot = graph
        
        available_actions = self.env.get_handover_action_mask(
            self.max_candidates,
            apply_pre_handover_gate=False,
        )

        candidate_sat_ids = np.full(
            (self.num_agents, self.max_candidates),
            -1,
            dtype=np.int64,
        )
        for uid, user in enumerate(self.env.user_manager.users):
            # available_actions and candidate_sat_ids
            visible_sats = self.env._get_handover_candidates(user)
            valid_count = min(len(visible_sats), self.max_candidates)
            if valid_count > 0:
                candidate_sat_ids[uid, :valid_count] = [
                    sat_info.sat_id for sat_info in visible_sats[:valid_count]
                ]

        raw_observations = self._raw_policy_observations()
        user_embeddings = np.concatenate(
            [raw_observations, self._cached_han_user_embed],
            axis=1,
        ).astype(np.float32, copy=False)
        return user_embeddings, self._cached_sat_embed, available_actions, candidate_sat_ids

    def _get_observations(self, env_obs: np.ndarray) -> tuple:
        """
        Compatibility wrapper that returns the HAN-encoded policy inputs.

        Keeping a second observation-construction path here caused the test
        harness to drift away from the actual training and evaluation stack.
        We now route every caller through ``_encode_graph_state`` so action
        masks and user embeddings stay consistent across the codebase.
        """
        del env_obs

        observations, _, available_actions, _ = (
            self._encode_graph_state()
        )
        global_state = observations.reshape(-1).astype(np.float32, copy=False)

        if global_state.size != self.global_state_dim:
            padded_state = np.zeros(self.global_state_dim, dtype=np.float32)
            copy_len = min(global_state.size, self.global_state_dim)
            padded_state[:copy_len] = global_state[:copy_len]
            global_state = padded_state

        return observations, global_state, available_actions

    def _apply_pre_handover_action_mask(
        self,
        available_actions: np.ndarray,
        pre_handover_mask: np.ndarray,
    ) -> np.ndarray:
        """
        应用 pre-handover 掩码

        对于安全用户（pre_handover_mask=False），强制只能选择 action 0（不切换）。
        对于需要切换的用户（pre_handover_mask=True），保持原有可用动作。
        """
        gated = np.asarray(available_actions, dtype=np.float32).copy()
        pre_handover_mask = np.asarray(pre_handover_mask, dtype=bool)
        safe_users = ~pre_handover_mask
        gated[safe_users, 1:] = 0.0
        gated[:, 0] = 1.0
        return gated

    def _process_actions(self, actions: Dict) -> np.ndarray:
        """
        处理MAPPO输出的动作为环境格式
        
        Args:
            actions: MAPPO输出 {'handover': (N,), 'offload': (N,)}
            
        Returns:
            env_actions: shape=(num_users, 2) 环境动作格式
        """
        handover = actions['handover']  # (num_agents,) int
        offload = actions['offload']    # (num_agents,) float
        
        # 构建环境动作 (num_users, 2)
        env_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        env_actions[:, 0] = handover   # 切换动作
        env_actions[:, 1] = offload    # 卸载比例
        
        return env_actions

    @staticmethod
    def _empty_env_stats() -> Dict[str, Any]:
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
            'total_energy': 0.0,
            'successful_task_delay_samples': [],
            'reward_task_success': 0.0,
            'reward_load_balance': 0.0,
            'penalty_delay': 0.0,
            'penalty_energy': 0.0,
            'penalty_task_failure': 0.0,
            'penalty_service_interruption': 0.0,
            'penalty_failed_handover': 0.0,
            'load_balance_sum': 0.0,
            'mec_activity_sum': 0.0,
            'load_balance_samples': 0,
            'jain_load_fairness_sum': 0.0,
            'jain_load_fairness_samples': 0,
            'load_variance_samples': [],
        }

    @classmethod
    def _accumulate_env_stats(
        cls,
        target: Dict[str, Any],
        source: Dict[str, Any],
    ) -> Dict[str, Any]:
        for key, default_value in cls._empty_env_stats().items():
            if isinstance(default_value, list):
                target.setdefault(key, [])
                target[key] += list(source.get(key, default_value) or [])
            else:
                target[key] += source.get(key, default_value)
        return target
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集一轮训练数据
        
        Returns:
            统计信息字典
        """
        self.buffer.reset()
        self.mappo.actor.eval()
        self.mappo.critic.eval()
        self.han_encoder.eval()
        
        episode_rewards = []
        episode_lengths = []
        current_reward = 0
        current_length = 0
        
        # 追踪每步奖励（用于没有完成episode时的统计）
        step_rewards = []
        rollout_env_stats = self._empty_env_stats()
        
        obs, info = self.env.reset()
        self._invalidate_han_cache()
        observations, satellite_embeddings, available_actions, candidate_sat_ids = self._encode_graph_state()
        # 应用 pre-handover 掩码
        pre_handover_mask = self.env.get_pre_handover_mask()
        available_actions = self._apply_pre_handover_action_mask(available_actions, pre_handover_mask)
        global_state = observations.flatten()

        for step in range(self.config.n_steps):
            user_tasks = getattr(self.env, "user_tasks", None)
            continuous_action_mask = np.asarray(
                [
                    True if user_tasks is None else user_tasks.get(user_id) is not None
                    for user_id in range(self.num_agents)
                ],
                dtype=np.float32,
            )
            rollout_graph_snapshot = getattr(
                self,
                "_cached_graph_snapshot",
                None,
            )
            # 选择动作
            with torch.no_grad():
                actions, log_probs, value = self.mappo.act(
                    observations,
                    available_actions,
                    satellite_embeddings=satellite_embeddings,
                    candidate_sat_ids=candidate_sat_ids,
                    continuous_action_mask=continuous_action_mask,
                )
            
            # 执行动作
            env_actions = self._process_actions(actions)
            _, rewards, terminated, truncated, info = self.env.step(
                env_actions,
                return_observation=False,
                return_info=True
            )
            info = info or {}
            
            done = terminated or truncated
            timeout_bootstrap_value = 0.0
            if truncated and not terminated:
                # TimeLimit 不是 MDP 终态：在 reset 前对真实终态观测估值，
                # 但 GAE 递推仍在此处截断，不能穿过下一个 episode。
                self._invalidate_han_cache()
                terminal_observations, terminal_satellite_embeddings, _, _ = (
                    self._encode_graph_state()
                )
                with torch.no_grad():
                    timeout_bootstrap_value = self.mappo.get_value(
                        terminal_observations,
                        satellite_embeddings=terminal_satellite_embeddings,
                    )
            
            agent_rewards = self._agent_reward_vector(rewards)

            # 统计口径：使用每步全局平均奖励（避免多智能体重复求和导致量纲放大）
            if isinstance(rewards, (int, float)):
                reward = float(rewards)
            elif isinstance(rewards, dict):
                reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            else:
                reward = float(np.mean(np.asarray(rewards, dtype=np.float32)))
            
            # 追踪每步奖励
            step_rewards.append(reward)
            
            # 存储到缓冲区
            self.buffer.add(
                obs=observations,
                global_state=global_state,
                satellite_embeddings=satellite_embeddings,
                actions_discrete=actions['handover'],
                actions_continuous=actions['offload'],
                rewards=agent_rewards,
                done=done,
                value=value,
                log_probs=log_probs,
                candidate_masks=available_actions,
                continuous_action_masks=continuous_action_mask,
                candidate_sat_ids=candidate_sat_ids,
                timeout_bootstrap_value=timeout_bootstrap_value,
                graph_snapshot=rollout_graph_snapshot,
            )
            
            # 更新统计
            current_reward += reward
            current_length += 1
            self.total_steps += 1
            
            # 处理episode结束
            if done:
                self._accumulate_env_stats(rollout_env_stats, self.env.get_stats_summary())
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                self.recent_rewards.append(current_reward)
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)
                
                current_reward = 0
                current_length = 0
                self.episodes += 1

                self.env.reset()
                self._invalidate_han_cache()

            # 更新观测
            observations, satellite_embeddings, available_actions, candidate_sat_ids = self._encode_graph_state()
            # 应用 pre-handover 掩码
            pre_handover_mask = self.env.get_pre_handover_mask()
            available_actions = self._apply_pre_handover_action_mask(available_actions, pre_handover_mask)
            global_state = observations.flatten()

        # 计算最后一步的价值（用于GAE）
        with torch.no_grad():
            last_value = self.mappo.get_value(observations, satellite_embeddings=satellite_embeddings)
        
        # 计算优势和回报
        self.buffer.compute_returns_and_advantages(last_value, last_done=done)

        if not done:
            self._accumulate_env_stats(rollout_env_stats, self.env.stats.copy())
        
        # 统计信息
        rollout_total_reward = sum(step_rewards)
        rollout_mean_reward = np.mean(step_rewards) if step_rewards else 0
        
        # 获取环境统计（切换成功率、任务完成率、延迟、能耗等）
        env_stats = rollout_env_stats
        summary_env_stats = summarize_env_stats_with_load_balance(env_stats)
        
        stats = {
            'episodes': len(episode_rewards),
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_reward': np.std(episode_rewards) if len(episode_rewards) > 1 else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'total_steps': self.total_steps,
            'rollout_total_reward': rollout_total_reward,
            'rollout_mean_reward': rollout_mean_reward,
            # 环境指标
            'total_handovers': env_stats.get('total_handovers', 0),
            'handover_attempts': env_stats.get('handover_attempts', 0),
            'handover_committed': env_stats.get('handover_committed', 0),
            'handover_aborted': env_stats.get('handover_aborted', 0),
            'handover_radio_failures': env_stats.get('handover_radio_failures', 0),
            'migration_rejections': env_stats.get('migration_rejections', 0),
            'reconnection_attempts': env_stats.get('reconnection_attempts', 0),
            'reconnections': env_stats.get('reconnections', 0),
            'successful_handovers': env_stats.get('successful_handovers', 0),
            'failed_handovers': env_stats.get('failed_handovers', 0),
            'forced_disconnects': env_stats.get('forced_disconnects', 0),
            'total_user_seconds': env_stats.get('total_user_seconds', 0.0),
            'blocked_user_seconds': env_stats.get('blocked_user_seconds', 0.0),
            'handover_interruption_seconds': env_stats.get('handover_interruption_seconds', 0.0),
            'service_interruption_seconds': env_stats.get('service_interruption_seconds', 0.0),
            'handover_success_rate': summary_env_stats.get('handover_success_rate', 0.0),
            'handover_failure_rate': summary_env_stats.get('handover_failure_rate', 0.0),
            'forced_termination_rate': summary_env_stats.get('forced_termination_rate', 0.0),
            'service_availability_rate': summary_env_stats.get('service_availability_rate', 0.0),
            'service_continuity_rate': summary_env_stats.get('service_continuity_rate', 0.0),
            'successful_task_throughput': summary_env_stats.get('successful_task_throughput', 0.0),
            'total_tasks': env_stats.get('total_tasks', 0),
            'completed_tasks': env_stats.get('completed_tasks', 0),
            'deadline_violations': env_stats.get('deadline_violations', 0),
            'failed_tasks': env_stats.get('failed_tasks', 0),
            'resolved_tasks': summary_env_stats.get('resolved_tasks', 0),
            'pending_tasks': summary_env_stats.get('pending_tasks', 0),
            'task_completion_rate': summary_env_stats.get('task_completion_rate', 0.0),
            'task_success_rate': summary_env_stats.get('task_success_rate', 0.0),
            'task_failure_rate': summary_env_stats.get('task_failure_rate', 0.0),
            'task_settlement_rate': summary_env_stats.get('task_settlement_rate', 0.0),
            'task_resolution_rate': summary_env_stats.get('task_resolution_rate', 0.0),
            'pending_task_rate': summary_env_stats.get('pending_task_rate', 0.0),
            'deadline_violation_rate': summary_env_stats.get('deadline_violation_rate', 0.0),
            'avg_delay': summary_env_stats.get('avg_delay', 0.0),
            'avg_success_delay': summary_env_stats.get('avg_success_delay', 0.0),
            'p95_success_delay': summary_env_stats.get('p95_success_delay', 0.0),
            'handover_frequency': summary_env_stats.get('handover_frequency', 0.0),
            'handovers_per_user_minute': summary_env_stats.get('handovers_per_user_minute', 0.0),
            'blocked_time_ratio': summary_env_stats.get('blocked_time_ratio', 0.0),
            'total_energy': env_stats.get('total_energy', 0.0),
            'energy_per_resolved_task': (
                env_stats.get('total_energy', 0.0) /
                max(float(summary_env_stats.get('resolved_tasks', 0.0)), 1.0)
            ),
            'energy_per_successful_task': summary_env_stats.get('energy_per_successful_task', 0.0),
            'load_balance_variance': summary_env_stats.get('load_balance_variance', 0.0),
            'load_balance_coefficient': summary_env_stats.get('load_balance_coefficient', 0.0),
            'load_variance_sample_count': summary_env_stats.get('load_variance_sample_count', 0),
            'mec_load_fairness': summary_env_stats.get('mec_load_fairness', 0.0),
            'jain_mec_load_fairness': summary_env_stats.get('jain_mec_load_fairness', 0.0),
            'avg_load_balance_score': summary_env_stats.get('avg_load_balance_score', 0.0),
            'active_load_balance_score': summary_env_stats.get('active_load_balance_score', 0.0),
            **reward_breakdown(env_stats),
        }
        
        return stats
    
    def train(self):
        """
        执行训练
        """
        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info(f"  总步数: {self.config.total_timesteps:,}")
        self.logger.info(f"  每轮步数: {self.config.n_steps}")
        self.logger.info(f"  图更新间隔: {self.config.graph_update_interval}")
        self.logger.info(f"  评估episodes: {self.config.eval_episodes}")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info("=" * 60)
        
        self.training_start_time = time.time()
        
        num_updates = int(np.ceil(self.config.total_timesteps / self.config.n_steps))
        
        # Early stopping 状态
        es_patience = getattr(self.config, 'early_stop_patience', 0)
        es_best_reward = float('-inf')
        es_counter = 0
        
        for update in range(1, num_updates + 1):
            update_start = time.time()
            
            # 1. 收集数据
            rollout_start = time.time()
            rollout_stats = self.collect_rollouts()
            rollout_elapsed = time.time() - rollout_start
            
            # 2. 更新策略
            ppo_update_start = time.time()
            update_stats = self.mappo.update()
            ppo_update_elapsed = time.time() - ppo_update_start
            update_stats["rollout_elapsed_sec"] = rollout_elapsed
            update_stats["ppo_update_elapsed_sec"] = ppo_update_elapsed
            
            # 3. 记录日志 + 保存训练历史
            elapsed = time.time() - update_start
            record = {
                'update': update,
                'total_steps': self.total_steps,
                'episodes': self.episodes,
                'elapsed_sec': elapsed,
                # rollout 指标
                'mean_reward': rollout_stats.get('mean_reward', 0),
                'std_reward': rollout_stats.get('std_reward', 0),
                'mean_length': rollout_stats.get('mean_length', 0),
                'rollout_total_reward': rollout_stats.get('rollout_total_reward', 0),
                'rollout_mean_reward': rollout_stats.get('rollout_mean_reward', 0),
                'recent_mean_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0,
                # MAPPO update 指标
                'actor_loss': update_stats.get('actor_loss', 0),
                'critic_loss': update_stats.get('critic_loss', 0),
                'entropy': update_stats.get('entropy', 0),
                'kl_divergence': update_stats.get('kl_divergence', 0),
                'clip_fraction': update_stats.get('clip_fraction', 0),
                'han_grad_norm': update_stats.get('han_grad_norm', 0),
                'han_parameter_delta': update_stats.get('han_parameter_delta', 0),
                'han_metapaths_executed': update_stats.get(
                    'han_metapaths_executed', 0
                ),
                # 环境指标
                'total_handovers': rollout_stats.get('total_handovers', 0),
                'handover_attempts': rollout_stats.get('handover_attempts', 0),
                'handover_committed': rollout_stats.get('handover_committed', 0),
                'handover_aborted': rollout_stats.get('handover_aborted', 0),
                'handover_radio_failures': rollout_stats.get('handover_radio_failures', 0),
                'migration_rejections': rollout_stats.get('migration_rejections', 0),
                'reconnection_attempts': rollout_stats.get('reconnection_attempts', 0),
                'reconnections': rollout_stats.get('reconnections', 0),
                'successful_handovers': rollout_stats.get('successful_handovers', 0),
                'failed_handovers': rollout_stats.get('failed_handovers', 0),
                'forced_disconnects': rollout_stats.get('forced_disconnects', 0),
                'total_user_seconds': rollout_stats.get('total_user_seconds', 0.0),
                'blocked_user_seconds': rollout_stats.get('blocked_user_seconds', 0.0),
                'handover_interruption_seconds': rollout_stats.get('handover_interruption_seconds', 0.0),
                'service_interruption_seconds': rollout_stats.get('service_interruption_seconds', 0.0),
                'handover_success_rate': rollout_stats.get('handover_success_rate', 0),
                'handover_failure_rate': rollout_stats.get('handover_failure_rate', 0),
                'forced_termination_rate': rollout_stats.get('forced_termination_rate', 0),
                'service_availability_rate': rollout_stats.get('service_availability_rate', 0),
                'service_continuity_rate': rollout_stats.get('service_continuity_rate', 0),
                'successful_task_throughput': derive_paper_metrics(rollout_stats).get('successful_task_throughput', 0.0),
                'total_tasks': rollout_stats.get('total_tasks', 0),
                'completed_tasks': rollout_stats.get('completed_tasks', 0),
                'deadline_violations': rollout_stats.get('deadline_violations', 0),
                'failed_tasks': rollout_stats.get('failed_tasks', 0),
                'resolved_tasks': rollout_stats.get('resolved_tasks', 0),
                'pending_tasks': rollout_stats.get('pending_tasks', 0),
                'task_completion_rate': rollout_stats.get('task_completion_rate', 0),
                'task_success_rate': rollout_stats.get('task_success_rate', 0),
                'task_failure_rate': rollout_stats.get('task_failure_rate', 0),
                'task_settlement_rate': rollout_stats.get('task_settlement_rate', 0),
                'task_resolution_rate': rollout_stats.get('task_resolution_rate', 0),
                'pending_task_rate': rollout_stats.get('pending_task_rate', 0),
                'deadline_violation_rate': rollout_stats.get('deadline_violation_rate', 0),
                'avg_delay': rollout_stats.get('avg_delay', 0),
                'avg_success_delay': rollout_stats.get('avg_success_delay', 0),
                'p95_success_delay': rollout_stats.get('p95_success_delay', 0),
                'handover_frequency': rollout_stats.get('handover_frequency', 0),
                'handovers_per_user_minute': rollout_stats.get('handovers_per_user_minute', 0),
                'blocked_time_ratio': rollout_stats.get('blocked_time_ratio', 0),
                'total_energy': rollout_stats.get('total_energy', 0),
                'energy_per_resolved_task': energy_per_resolved_task(rollout_stats),
                'energy_per_successful_task': energy_per_successful_task(rollout_stats),
                'load_balance_variance': rollout_stats.get('load_balance_variance', 0),
                'load_balance_coefficient': rollout_stats.get('load_balance_coefficient', rollout_stats.get('mec_load_fairness', 0)),
                'load_variance_sample_count': rollout_stats.get('load_variance_sample_count', 0),
                'mec_load_fairness': rollout_stats.get('mec_load_fairness', rollout_stats.get('avg_load_balance_score', 0)),
                'jain_mec_load_fairness': rollout_stats.get('jain_mec_load_fairness', 0),
                'avg_load_balance_score': rollout_stats.get('avg_load_balance_score', 0),
                'active_load_balance_score': rollout_stats.get('active_load_balance_score', 0),
                **reward_breakdown(rollout_stats),
            }
            self.training_history.append(record)
            
            if update % self.config.log_interval == 0:
                self._log_training(update, rollout_stats, update_stats, update_start)
            
            # 4. 评估
            if self.total_steps % self.config.eval_interval < self.config.n_steps:
                self._evaluate()
            
            # 5. 保存检查点
            if self.total_steps % self.config.save_interval < self.config.n_steps:
                self._save_checkpoint()
            
            # 6. Early stopping 检查
            if es_patience > 0 and self.recent_rewards:
                current_reward = float(np.mean(self.recent_rewards))
                if current_reward > es_best_reward * 1.001:  # 相对改善阈值 0.1%
                    es_best_reward = current_reward
                    es_counter = 0
                else:
                    es_counter += 1
                if es_counter >= es_patience:
                    self.logger.info(f"Early stopping: 连续 {es_patience} 次更新无改善，停止训练")
                    break
        
        # 训练结束
        total_time = time.time() - self.training_start_time
        self.logger.info("=" * 60)
        self.logger.info("训练完成!")
        self.logger.info(f"  总时间: {total_time / 3600:.2f} 小时")
        self.logger.info(f"  总步数: {self.total_steps:,}")
        self.logger.info(f"  总episode: {self.episodes}")
        self.logger.info(f"  最佳奖励: {self.best_reward:.2f}")
        self.logger.info(
            f"  最佳模型指标: {best_model_metric_label(self.config.best_model_metric)} = "
            f"{self.best_model_score:.4f}"
        )
        self.logger.info("=" * 60)
        
        # 保存最终模型
        self._save_checkpoint(final=True)
        
        # 保存训练历史（供可视化使用）
        self._save_training_history()
    
    def _log_training(self, update: int, rollout_stats: Dict, update_stats: Dict, start_time: float):
        """记录训练日志"""
        elapsed = time.time() - start_time
        fps = self.config.n_steps / elapsed
        
        # 优先显示 episode 奖励，否则显示 rollout 每步平均奖励
        if self.recent_rewards:
            display_reward = np.mean(self.recent_rewards)
            reward_type = "Ep"
        else:
            display_reward = rollout_stats.get('rollout_total_reward', 0)
            reward_type = "Roll"
        
        self.logger.info(
            f"Update {update:4d} | "
            f"Steps: {self.total_steps:7,} | "
            f"Episodes: {self.episodes:5d} | "
            f"{reward_type}Reward: {display_reward:8.2f} | "
            f"FPS: {fps:6.0f} | "
            f"Actor Loss: {update_stats.get('actor_loss', 0):.4f} | "
            f"Critic Loss: {update_stats.get('critic_loss', 0):.4f} | "
            f"HAN Grad: {update_stats.get('han_grad_norm', 0):.4f} | "
            f"HAN Δ: {update_stats.get('han_parameter_delta', 0):.6f} | "
            f"HAN Metapaths: {update_stats.get('han_metapaths_executed', 0)}"
        )
        self.logger.info(
            "  Phase | "
            f"Rollout: {update_stats.get('rollout_elapsed_sec', 0.0):.2f}s | "
            f"PPO update: {update_stats.get('ppo_update_elapsed_sec', 0.0):.2f}s"
        )
        self.logger.info(
            "  Env | "
            f"HO: {rollout_stats.get('handover_success_rate', 0):.2%} | "
            f"Throughput: {derive_paper_metrics(rollout_stats).get('successful_task_throughput', 0.0):.2f}/user-min | "
            f"Task: {rollout_stats.get('task_completion_rate', 0):.2%} | "
            f"Resolved: {rollout_stats.get('task_resolution_rate', 0):.2%} | "
            f"Delay: {rollout_stats.get('avg_delay', 0):.3f}s | "
            f"Energy: {rollout_stats.get('total_energy', 0):.2f}J | "
            f"MEC fair: {rollout_stats.get('mec_load_fairness', rollout_stats.get('avg_load_balance_score', 0)):.3f}"
        )
        self.logger.info(
            "  Reward | "
            f"Success: {rollout_stats.get('reward_task_success', 0):.2f} | "
            f"Load: {rollout_stats.get('reward_load_balance', 0):.2f} | "
            f"Delay: {rollout_stats.get('penalty_delay', 0):.2f} | "
            f"Energy: {rollout_stats.get('penalty_energy', 0):.2f} | "
            f"TaskFail: {rollout_stats.get('penalty_task_failure', 0):.2f} | "
            f"Interrupt: {rollout_stats.get('penalty_service_interruption', 0):.2f} | "
            f"HOFail: {rollout_stats.get('penalty_failed_handover', 0):.2f}"
        )
    
    def _run_isolated_on_policy_evaluation(self):
        """在独立环境和固定种子上运行 on-policy 评估。"""
        eval_rewards = []
        eval_lengths = []
        eval_env_stats = self._empty_env_stats()
        training_env = self.env
        eval_env = self._create_eval_env()
        cached_state = (
            getattr(self, '_cached_han_user_embed', None),
            getattr(self, '_cached_sat_embed', None),
            getattr(self, '_cached_graph_snapshot', None),
        )
        model_modes = (
            self.han_encoder.training,
            self.mappo.actor.training,
            self.mappo.critic.training,
        )

        try:
            self.env = eval_env
            self.han_encoder.eval()
            self.mappo.actor.eval()
            self.mappo.critic.eval()

            for ep in range(self.config.eval_episodes):
                self.env.reset(
                    seed=self.config.seed + EVALUATION_SEED_OFFSET + ep
                )
                self._invalidate_han_cache()
                (
                    observations,
                    satellite_embeddings,
                    available_actions,
                    candidate_sat_ids,
                ) = self._encode_graph_state()
                # 应用 pre-handover 掩码
                pre_handover_mask = self.env.get_pre_handover_mask()
                available_actions = self._apply_pre_handover_action_mask(available_actions, pre_handover_mask)

                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done:
                    with torch.no_grad():
                        actions, _, _ = self.mappo.act(
                            observations,
                            available_actions,
                            satellite_embeddings=satellite_embeddings,
                            candidate_sat_ids=candidate_sat_ids,
                            deterministic=True
                        )

                    env_actions = self._process_actions(actions)
                    _, rewards, terminated, truncated, _ = self.env.step(
                        env_actions,
                        return_observation=False,
                        return_info=False
                    )
                    done = bool(terminated or truncated)

                    if isinstance(rewards, (int, float)):
                        episode_reward += rewards
                    elif isinstance(rewards, dict):
                        episode_reward += sum(rewards.values())
                    else:
                        episode_reward += np.sum(rewards)

                    episode_length += 1
                    if not done:
                        (
                            observations,
                            satellite_embeddings,
                            available_actions,
                            candidate_sat_ids,
                        ) = self._encode_graph_state()
                        pre_handover_mask = self.env.get_pre_handover_mask()
                        available_actions = self._apply_pre_handover_action_mask(
                            available_actions,
                            pre_handover_mask,
                        )

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                self._accumulate_env_stats(
                    eval_env_stats,
                    self.env.get_stats_summary(),
                )
        finally:
            self.env = training_env
            (
                self._cached_han_user_embed,
                self._cached_sat_embed,
                self._cached_graph_snapshot,
            ) = cached_state
            self.han_encoder.train(model_modes[0])
            self.mappo.actor.train(model_modes[1])
            self.mappo.critic.train(model_modes[2])
            if eval_env is not training_env and hasattr(eval_env, 'close'):
                eval_env.close()

        return eval_rewards, eval_lengths, eval_env_stats

    def _evaluate(self):
        """评估当前策略"""
        self.logger.info("-" * 40)
        self.logger.info("开始评估...")

        (
            eval_rewards,
            eval_lengths,
            eval_env_stats,
        ) = self._run_isolated_on_policy_evaluation()
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_length = np.mean(eval_lengths)
        summary_env_stats = summarize_env_stats_with_load_balance(eval_env_stats)
        
        # 记录评估结果到 eval_history
        eval_record = {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'eval_mean_reward': float(mean_reward),
            'eval_std_reward': float(std_reward),
            'eval_mean_length': float(mean_length),
            'eval_rewards': [float(r) for r in eval_rewards],
            'total_handovers': eval_env_stats.get('total_handovers', 0),
            'handover_attempts': eval_env_stats.get('handover_attempts', 0),
            'handover_committed': eval_env_stats.get('handover_committed', 0),
            'handover_aborted': eval_env_stats.get('handover_aborted', 0),
            'handover_radio_failures': eval_env_stats.get('handover_radio_failures', 0),
            'migration_rejections': eval_env_stats.get('migration_rejections', 0),
            'reconnection_attempts': eval_env_stats.get('reconnection_attempts', 0),
            'reconnections': eval_env_stats.get('reconnections', 0),
            'successful_handovers': eval_env_stats.get('successful_handovers', 0),
            'failed_handovers': eval_env_stats.get('failed_handovers', 0),
            'forced_disconnects': eval_env_stats.get('forced_disconnects', 0),
            'total_user_seconds': eval_env_stats.get('total_user_seconds', 0.0),
            'blocked_user_seconds': eval_env_stats.get('blocked_user_seconds', 0.0),
            'handover_interruption_seconds': eval_env_stats.get('handover_interruption_seconds', 0.0),
            'service_interruption_seconds': eval_env_stats.get('service_interruption_seconds', 0.0),
            'handover_success_rate': summary_env_stats.get('handover_success_rate', 0.0),
            'handover_failure_rate': summary_env_stats.get('handover_failure_rate', 0.0),
            'forced_termination_rate': summary_env_stats.get('forced_termination_rate', 0.0),
            'service_availability_rate': summary_env_stats.get('service_availability_rate', 0.0),
            'service_continuity_rate': summary_env_stats.get('service_continuity_rate', 0.0),
            'successful_task_throughput': summary_env_stats.get('successful_task_throughput', 0.0),
            'resolved_tasks': summary_env_stats.get('resolved_tasks', 0),
            'pending_tasks': summary_env_stats.get('pending_tasks', 0),
            'total_tasks': eval_env_stats.get('total_tasks', 0),
            'completed_tasks': eval_env_stats.get('completed_tasks', 0),
            'deadline_violations': eval_env_stats.get('deadline_violations', 0),
            'failed_tasks': eval_env_stats.get('failed_tasks', 0),
            'task_completion_rate': summary_env_stats.get('task_completion_rate', 0.0),
            'task_success_rate': summary_env_stats.get('task_success_rate', 0.0),
            'task_failure_rate': summary_env_stats.get('task_failure_rate', 0.0),
            'task_settlement_rate': summary_env_stats.get('task_settlement_rate', 0.0),
            'task_resolution_rate': summary_env_stats.get('task_resolution_rate', 0.0),
            'pending_task_rate': summary_env_stats.get('pending_task_rate', 0.0),
            'deadline_violation_rate': summary_env_stats.get('deadline_violation_rate', 0.0),
            'avg_delay': summary_env_stats.get('avg_delay', 0.0),
            'avg_success_delay': summary_env_stats.get('avg_success_delay', 0.0),
            'p95_success_delay': summary_env_stats.get('p95_success_delay', 0.0),
            'handover_frequency': summary_env_stats.get('handover_frequency', 0.0),
            'handovers_per_user_minute': summary_env_stats.get('handovers_per_user_minute', 0.0),
            'blocked_time_ratio': summary_env_stats.get('blocked_time_ratio', 0.0),
            'total_energy': eval_env_stats.get('total_energy', 0.0),
            'energy_per_resolved_task': (
                eval_env_stats.get('total_energy', 0.0) / max(float(summary_env_stats.get('resolved_tasks', 0.0)), 1.0)
            ),
            'energy_per_successful_task': summary_env_stats.get('energy_per_successful_task', 0.0),
            'load_balance_variance': summary_env_stats.get('load_balance_variance', 0.0),
            'load_balance_coefficient': summary_env_stats.get('load_balance_coefficient', 0.0),
            'load_variance_sample_count': summary_env_stats.get('load_variance_sample_count', 0),
            'mec_load_fairness': summary_env_stats.get('mec_load_fairness', 0.0),
            'jain_mec_load_fairness': summary_env_stats.get('jain_mec_load_fairness', 0.0),
            'avg_load_balance_score': summary_env_stats.get('avg_load_balance_score', 0.0),
            'active_load_balance_score': summary_env_stats.get('active_load_balance_score', 0.0),
            **reward_breakdown(eval_env_stats),
        }
        selection_metric = getattr(
            self.config,
            'best_model_metric',
            TrainConfig.best_model_metric,
        )
        eval_record['best_model_metric'] = selection_metric
        eval_record['best_model_score'] = float(
            compute_model_selection_score(eval_record, selection_metric)
        )
        self.eval_history.append(eval_record)
        
        self.logger.info(
            f"评估结果: 奖励 = {mean_reward:.2f} ± {std_reward:.2f}, "
            f"长度 = {mean_length:.0f}, "
            f"时延 = {eval_record['avg_delay']:.3f}s, "
            f"能耗 = {eval_record['total_energy']:.2f}J, "
            f"MEC公平性 = {eval_record['mec_load_fairness']:.3f}"
        )
        
        reward_improved = mean_reward > self.best_reward
        if reward_improved:
            self.best_reward = mean_reward

        # 使用可配置的多指标口径选择 best_model.pt
        if eval_record['best_model_score'] > self.best_model_score:
            self.best_model_score = eval_record['best_model_score']
            self.logger.info(
                f"新的最佳模型 ({best_model_metric_label(selection_metric)}): "
                f"score = {self.best_model_score:.4f}, "
                f"delay = {eval_record['avg_delay']:.3f}s, "
                f"completion = {eval_record['task_completion_rate']:.2%}, "
                f"throughput = {eval_record['successful_task_throughput']:.2f}/user-min, "
                f"mec_fairness = {eval_record['mec_load_fairness']:.3f}"
            )
            self._save_checkpoint(best=True)

        # 继续记录最高 reward，便于和旧实验保持一致
        if reward_improved:
            self.logger.info(f"新的最佳奖励: {self.best_reward:.2f}")
        
        self.logger.info("-" * 40)
    
    def _save_training_history(self):
        """保存训练历史数据（JSON格式，供 plot_training_artifacts.py 可视化使用）"""
        save_dir = Path(self.config.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'model_schema_version': MODEL_SCHEMA_VERSION,
            'geometry_schema_version': GEOMETRY_SCHEMA_VERSION,
            'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
            'config': asdict(self.config),
            'training': self.training_history,
            'evaluation': self.eval_history,
            'summary': {
                'total_steps': self.total_steps,
                'total_episodes': self.episodes,
                'best_reward': float(self.best_reward),
                'best_model_metric': self.config.best_model_metric,
                'best_model_score': float(self.best_model_score),
                'training_time_sec': time.time() - self.training_start_time if self.training_start_time else 0,
            }
        }
        
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"训练历史已保存: {history_path}")
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """保存检查点"""
        save_dir = Path(self.config.save_path)
        
        if best:
            filename = "best_model.pt"
        elif final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_{self.total_steps}.pt"
        
        save_path = save_dir / filename
        
        checkpoint = {
            'model_schema_version': MODEL_SCHEMA_VERSION,
            'geometry_schema_version': GEOMETRY_SCHEMA_VERSION,
            'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'best_reward': self.best_reward,
            'best_model_metric': self.config.best_model_metric,
            'best_model_score': self.best_model_score,
            'config': asdict(self.config),
            'actor_state_dict': self.mappo.actor.state_dict(),
            'critic_state_dict': self.mappo.critic.state_dict(),
            'actor_optimizer_state_dict': self.mappo.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.mappo.critic_optimizer.state_dict(),
            'han_state_dict': self.han_encoder.state_dict(),
            'han_optimizer_state_dict': (
                self.mappo.representation_optimizer.state_dict()
                if self.mappo.representation_optimizer is not None
                else None
            ),
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"模型已保存: {save_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.logger.info(f"加载检查点: {path}")
        
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        schema_version = int(checkpoint.get('model_schema_version', 1))
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                "该检查点来自 P0 修复前的 HAN/环境结构，不能用于恢复训练或正式比较；"
                "请使用修复后的代码重新训练。"
            )
        environment_schema_version = int(
            checkpoint.get('environment_schema_version', 1)
        )
        geometry_schema_version = int(
            checkpoint.get('geometry_schema_version', 1)
        )
        if geometry_schema_version != GEOMETRY_SCHEMA_VERSION:
            raise ValueError(
                "检查点的轨道/图几何版本与当前代码不一致，不能恢复训练。"
            )
        if environment_schema_version != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(
                "该检查点来自 P1 修复前的 MEC 时序、切换候选或奖励结算语义，"
                "不能用于恢复训练或正式比较；请使用当前代码重新训练。"
            )
        
        self.total_steps = checkpoint['total_steps']
        self.episodes = checkpoint['episodes']
        self.best_reward = checkpoint['best_reward']
        self.best_model_score = checkpoint.get('best_model_score', self.best_model_score)
        if 'best_model_metric' in checkpoint:
            self.config.best_model_metric = checkpoint['best_model_metric']
        
        self.mappo.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.mappo.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.mappo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.mappo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'han_state_dict' in checkpoint:
            self.han_encoder.load_state_dict(checkpoint['han_state_dict'])
        if (
            checkpoint.get('han_optimizer_state_dict') is not None
            and self.mappo.representation_optimizer is not None
        ):
            self.mappo.representation_optimizer.load_state_dict(
                checkpoint['han_optimizer_state_dict']
            )
        
        self.logger.info(f"从步数 {self.total_steps:,} 恢复训练")


class AttentionMAPPOTrainer(HANMAPPOTrainer):
    """Raw observation MAPPO with satellite-load attention over candidate links."""

    algorithm_name = "attn_mappo"
    sat_feature_dim = 8

    def _init_environment(self):
        self.logger.info("Initializing environment for AttentionMAPPO...")

        self.env = LEOSatelliteEnv(build_env_config(self.config))
        self.num_agents = self.config.num_users
        self.max_candidates = self.config.max_visible_sats
        self.raw_obs_dim = self.env.user_obs_dim
        self.han_out_dim = self.sat_feature_dim
        self.obs_dim = self.raw_obs_dim
        self.global_state_dim = self.num_agents * self.obs_dim

        self.logger.info(f"  - Raw observation dim: {self.raw_obs_dim}")
        self.logger.info(f"  - Satellite load feature dim: {self.sat_feature_dim}")
        self.logger.info(f"  - Policy observation dim: {self.obs_dim}")
        self.logger.info(f"  - Global state dim: {self.global_state_dim}")

    def _init_graph_builder(self):
        self.feature_extractor = None
        self.graph_builder = None
        self.logger.info("AttentionMAPPO skips heterogeneous graph construction.")

    def _init_han_encoder(self):
        self.han_encoder = torch.nn.Identity().to(self.device)
        self.logger.info("AttentionMAPPO uses raw_obs + satellite load attention; no HAN encoder.")

    def _init_mappo(self):
        self.logger.info("Initializing AttentionMAPPO...")

        mappo_config = MAPPOConfig(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            max_candidates=self.max_candidates,
            sat_embed_dim=self.sat_feature_dim,
            min_effective_offload_ratio=self.config.min_effective_offload_ratio,
            actor_hidden_dims=list(self.config.actor_hidden_dims),
            critic_hidden_dims=list(self.config.critic_hidden_dims),
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            value_loss_coef=self.config.value_loss_coef,
            value_loss_type=self.config.value_loss_type,
            normalize_returns=self.config.normalize_returns,
            value_huber_beta=self.config.value_huber_beta,
            entropy_coef=self.config.entropy_coef,
            entropy_schedule=self.config.entropy_schedule,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.config.device,
        )

        self.mappo = AttentionMAPPO(mappo_config)
        actor_params = sum(p.numel() for p in self.mappo.actor.parameters())
        critic_params = sum(p.numel() for p in self.mappo.critic.parameters())
        self.logger.info(f"  - Attention Actor params: {actor_params:,}")
        self.logger.info(f"  - Critic params: {critic_params:,}")

    def _invalidate_han_cache(self) -> None:
        return None

    def _satellite_load_features(self) -> np.ndarray:
        return build_satellite_load_features(self.env, self.num_agents)

    def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_observations = self._raw_policy_observations()
        satellite_features = self._satellite_load_features()
        available_actions = self.env.get_handover_action_mask(
            self.max_candidates,
            apply_pre_handover_gate=False,
        )
        candidate_sat_ids = np.full(
            (self.num_agents, self.max_candidates),
            -1,
            dtype=np.int64,
        )

        for uid, user in enumerate(self.env.user_manager.users):
            visible_sats = self.env._get_handover_candidates(user)
            valid_count = min(len(visible_sats), self.max_candidates)
            if valid_count > 0:
                candidate_sat_ids[uid, :valid_count] = [
                    sat_info.sat_id for sat_info in visible_sats[:valid_count]
                ]

        return raw_observations, satellite_features, available_actions, candidate_sat_ids


class HANCandidateAttentionMAPPOTrainer(HANMAPPOTrainer):
    """Final HAN+Attention MAPPO with predictive queue-risk context."""

    algorithm_name = "han_attn"

    def _init_environment(self):
        super()._init_environment()
        self.obs_dim = self.raw_obs_dim + self.han_out_dim + SHARED_CONSTRAINT_DIM
        self.global_state_dim = self.num_agents * self.obs_dim
        self.logger.info(
            f"  - HAN+Attn policy observation dim: {self.obs_dim} "
            f"(base HAN+MAPPO obs + shared constraints {SHARED_CONSTRAINT_DIM})"
        )
        self.logger.info(f"  - HAN+Attn global state dim: {self.global_state_dim}")

    def _init_mappo(self):
        self.logger.info("Initializing HAN+Attn MAPPO with context-aware candidate features...")

        fused_sat_dim = self.config.han_out_dim + SATELLITE_CONTEXT_FEATURE_DIM
        mappo_config = MAPPOConfig(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            max_candidates=self.max_candidates,
            sat_embed_dim=fused_sat_dim,
            risk_feature_start=self.config.han_out_dim + SATELLITE_LOAD_FEATURE_DIM,
            risk_feature_dim=SATELLITE_CONTEXT_FEATURE_DIM - SATELLITE_LOAD_FEATURE_DIM,
            min_effective_offload_ratio=self.config.min_effective_offload_ratio,
            actor_hidden_dims=list(self.config.actor_hidden_dims),
            critic_hidden_dims=list(self.config.critic_hidden_dims),
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            value_loss_coef=self.config.value_loss_coef,
            value_loss_type=self.config.value_loss_type,
            normalize_returns=self.config.normalize_returns,
            value_huber_beta=self.config.value_huber_beta,
            entropy_coef=self.config.entropy_coef,
            entropy_schedule=self.config.entropy_schedule,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.config.device,
        )

        self.mappo = AttentionMAPPO(mappo_config)
        self.mappo.set_representation_module(
            self.han_encoder,
            self._reencode_mappo_batch,
            learning_rate=self.config.learning_rate,
        )
        actor_params = sum(p.numel() for p in self.mappo.actor.parameters())
        critic_params = sum(p.numel() for p in self.mappo.critic.parameters())
        self.logger.info(f"  - HAN+Attn Actor params: {actor_params:,}")
        self.logger.info(f"  - Critic params: {critic_params:,}")

    def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observations, han_satellites, available_actions, candidate_sat_ids = (
            HANMAPPOTrainer._encode_graph_state(self)
        )
        context_features = build_satellite_context_features(self.env, self.num_agents)
        shared_constraints = build_shared_constraint_vector(self.env, self.num_agents)
        shared_repeated = np.repeat(
            shared_constraints[None, :],
            self.num_agents,
            axis=0,
        )
        observations = np.concatenate(
            [observations, shared_repeated],
            axis=1,
        ).astype(np.float32, copy=False)
        satellite_tokens = np.concatenate(
            [han_satellites, context_features],
            axis=1,
        ).astype(np.float32, copy=False)
        return observations, satellite_tokens, available_actions, candidate_sat_ids


class HANMADDPGTrainer(HANMAPPOTrainer):
    """HAN feature encoder with off-policy MADDPG on cached no-grad user embeddings."""

    algorithm_name = "maddpg"

    def _load_frozen_pretrained_han(self) -> None:
        """Load a trained HAN for replay-based algorithms and freeze it.

        Replay entries currently contain encoded observations rather than raw
        graph snapshots. End-to-end encoder updates would therefore make old
        entries stale. Requiring a trained frozen encoder is preferable to the
        previous silent use of a randomly initialized HAN.
        """
        checkpoint_path = getattr(self.config, "pretrained_han_path", None)
        if not checkpoint_path:
            raise ValueError(
                "HAN off-policy training requires pretrained_han_path; "
                "refusing to train with a frozen random HAN encoder"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        schema_contract = {
            'model_schema_version': MODEL_SCHEMA_VERSION,
            'geometry_schema_version': GEOMETRY_SCHEMA_VERSION,
            'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
        }
        for key, expected in schema_contract.items():
            actual = int(checkpoint.get(key, 0))
            if actual != expected:
                raise ValueError(
                    f"Frozen HAN checkpoint {checkpoint_path} has {key}={actual}; "
                    f"expected {expected}"
                )
        state_dict = checkpoint.get("han_state_dict")
        if state_dict is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not contain han_state_dict"
            )
        self.han_encoder.load_state_dict(state_dict)
        self.han_encoder.eval()
        for parameter in self.han_encoder.parameters():
            parameter.requires_grad_(False)
        self.logger.info(f"  - Loaded frozen pretrained HAN: {checkpoint_path}")

    def _init_mappo(self):
        self.logger.info("初始化 HAN+MADDPG...")
        self._load_frozen_pretrained_han()
        maddpg_config = MADDPGConfig(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            max_candidates=self.max_candidates,
            sat_embed_dim=self.han_out_dim,
            actor_hidden_dims=tuple(self.config.actor_hidden_dims),
            critic_hidden_dims=(512, 256, 128),
            actor_lr=self.config.maddpg_actor_lr,
            critic_lr=self.config.maddpg_critic_lr,
            gamma=self.config.gamma,
            tau=0.01,
            noise_start=self.config.noise_start,
            noise_final=self.config.noise_final,
            noise_decay_steps=max(int(self.config.total_timesteps), 1),
            batch_size=self.config.batch_size,
            replay_size=self.config.replay_size,
            warmup_steps=self.config.warmup_steps,
            grad_clip_norm=self.config.max_grad_norm,
            seed=self.config.seed,
            device=self.config.device,
        )
        self.algorithm = MADDPGAlgorithm(maddpg_config)
        self.maddpg = self.algorithm
        actor_params = sum(p.numel() for p in self.algorithm.actor.parameters())
        critic_params = sum(p.numel() for p in self.algorithm.critic.parameters())
        self.logger.info(f"  - MADDPG Actor参数量: {actor_params:,}")
        self.logger.info(f"  - MADDPG Critic参数量: {critic_params:,}")

    def _init_buffer(self):
        self.logger.info("初始化 off-policy replay buffer...")
        self.buffer = MultiAgentReplayBuffer(
            capacity=self.config.replay_size,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_feature_dim=self.max_candidates + 2,
            mask_dim=self.max_candidates + 1,
            device=self.config.device,
        )

    @staticmethod
    def _scalar_reward(rewards) -> float:
        if isinstance(rewards, (int, float)):
            return float(rewards)
        if isinstance(rewards, dict):
            return float(np.mean(list(rewards.values()))) if rewards else 0.0
        return float(np.mean(np.asarray(rewards, dtype=float)))

    def _replay_reward(self, reward_value: float):
        return self._agent_reward_vector(reward_value)

    def _reset_encoded_env(self, seed: Optional[int] = None):
        self._cached_han_user_embed = None
        self._cached_sat_embed = None
        self.env.reset(seed=seed)
        user_emb, sat_emb, masks, _ = self._encode_graph_state()
        masks = self._apply_pre_handover_action_mask(
            masks,
            self.env.get_pre_handover_mask(),
        )
        return user_emb, sat_emb, masks

    def _select_train_action(self, observations: np.ndarray, masks: np.ndarray, step_idx: int):
        if step_idx < self.config.warmup_steps:
            return self.algorithm.random_actions(masks.astype(bool))
        return self.algorithm.act(observations, masks.astype(bool), deterministic=False)

    def _select_eval_action(self, observations: np.ndarray, masks: np.ndarray):
        return self.algorithm.act(observations, masks.astype(bool), deterministic=True)

    def _action_features_from_env_actions(
        self,
        env_actions: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        handover_dim = int(self.max_candidates) + 1
        actions = np.asarray(env_actions, dtype=np.float32).reshape(self.num_agents, 2).copy()
        masks_np = np.asarray(masks, dtype=bool)
        handover = np.clip(np.rint(actions[:, 0]).astype(np.int64), 0, handover_dim - 1)
        for agent_id, action_id in enumerate(handover):
            if masks_np.shape == (self.num_agents, handover_dim) and not masks_np[agent_id, action_id]:
                valid = np.flatnonzero(masks_np[agent_id])
                handover[agent_id] = int(valid[0]) if len(valid) else 0
        offload = np.clip(actions[:, 1], 0.0, 1.0).astype(np.float32)
        features = np.zeros((self.num_agents, handover_dim + 1), dtype=np.float32)
        features[np.arange(self.num_agents), handover] = 1.0
        features[:, -1] = offload
        env_actions = np.column_stack([handover, offload]).astype(np.float32)
        return env_actions, features, handover

    def _safe_heuristic_actions(self, masks: np.ndarray):
        masks_np = np.asarray(masks, dtype=bool)
        actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        rvt_threshold = float(getattr(self.env.config, 'rvt_threshold_sec', 60.0))

        for uid, user in enumerate(self.env.user_manager.users):
            visible_sats = list(self.env._get_handover_candidates(user))[: self.max_candidates]
            keep_current = False
            if user.serving_satellite >= 0:
                vis = self.env._get_satellite_visibility(user, user.serving_satellite)
                keep_current = bool(vis is not None and vis.is_visible and vis.rvt_seconds >= rvt_threshold)

            handover = 0
            if not keep_current and visible_sats:
                target_idx = int(np.argmax([sat.elevation_deg for sat in visible_sats]))
                candidate = target_idx + 1
                if candidate < masks_np.shape[1] and masks_np[uid, candidate]:
                    handover = candidate

            task = self.env.user_tasks.get(uid)
            has_service_target = keep_current or handover > 0
            actions[uid, 0] = float(handover)
            actions[uid, 1] = 0.5 if task is not None and has_service_target else 0.0

        return self._action_features_from_env_actions(actions, masks_np)

    def _mixed_safe_random_actions(self, masks: np.ndarray, safe_probability: float = 0.0):
        masks_np = np.asarray(masks, dtype=bool)
        safe_actions, safe_features, safe_handover = self._safe_heuristic_actions(masks_np)
        random_actions, random_features, random_handover = self.algorithm.random_actions(masks_np)
        rng = getattr(self.algorithm, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.config.seed + self.total_steps)
        use_safe = rng.random(self.num_agents) < float(safe_probability)
        env_actions = np.where(use_safe[:, None], safe_actions, random_actions).astype(np.float32)
        action_features = np.where(use_safe[:, None], safe_features, random_features).astype(np.float32)
        handover = np.where(use_safe, safe_handover, random_handover).astype(np.int64)
        return env_actions, action_features, handover

    def _record_from_stats(
        self,
        update: int,
        episode_reward: float,
        episode_length: int,
        env_stats: Dict[str, float],
        update_stats: Dict[str, float],
        elapsed: float,
        partial_episode: bool = False,
    ) -> Dict[str, float]:
        summary = summarize_env_stats_with_load_balance(env_stats)
        record = {
            'update': update,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'elapsed_sec': elapsed,
            'mean_reward': float(episode_reward),
            'std_reward': 0.0,
            'mean_length': float(episode_length),
            'rollout_total_reward': float(episode_reward),
            'rollout_mean_reward': float(episode_reward / max(episode_length, 1)),
            'recent_mean_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
            'actor_loss': float(update_stats.get('actor_loss', 0.0)),
            'critic_loss': float(update_stats.get('critic_loss', update_stats.get('q_loss', 0.0))),
            'q_loss': float(update_stats.get('q_loss', 0.0)),
            'param_loss': float(update_stats.get('param_loss', 0.0)),
            'epsilon': float(update_stats.get('epsilon', 0.0)),
            'exploration_noise': float(update_stats.get('exploration_noise', 0.0)),
            'total_handovers': env_stats.get('total_handovers', 0),
            'handover_attempts': env_stats.get('handover_attempts', 0),
            'handover_committed': env_stats.get('handover_committed', 0),
            'handover_aborted': env_stats.get('handover_aborted', 0),
            'handover_radio_failures': env_stats.get('handover_radio_failures', 0),
            'migration_rejections': env_stats.get('migration_rejections', 0),
            'reconnection_attempts': env_stats.get('reconnection_attempts', 0),
            'reconnections': env_stats.get('reconnections', 0),
            'successful_handovers': env_stats.get('successful_handovers', 0),
            'failed_handovers': env_stats.get('failed_handovers', 0),
            'forced_disconnects': env_stats.get('forced_disconnects', 0),
            'total_user_seconds': env_stats.get('total_user_seconds', 0.0),
            'blocked_user_seconds': env_stats.get('blocked_user_seconds', 0.0),
            'handover_interruption_seconds': env_stats.get('handover_interruption_seconds', 0.0),
            'service_interruption_seconds': env_stats.get('service_interruption_seconds', 0.0),
            'handover_success_rate': summary.get('handover_success_rate', 0.0),
            'handover_failure_rate': summary.get('handover_failure_rate', 0.0),
            'forced_termination_rate': summary.get('forced_termination_rate', 0.0),
            'service_availability_rate': summary.get('service_availability_rate', 0.0),
            'service_continuity_rate': summary.get('service_continuity_rate', 0.0),
            'successful_task_throughput': summary.get('successful_task_throughput', 0.0),
            'total_tasks': env_stats.get('total_tasks', 0),
            'completed_tasks': env_stats.get('completed_tasks', 0),
            'deadline_violations': env_stats.get('deadline_violations', 0),
            'failed_tasks': env_stats.get('failed_tasks', 0),
            'resolved_tasks': summary.get('resolved_tasks', 0),
            'pending_tasks': summary.get('pending_tasks', 0),
            'task_completion_rate': summary.get('task_completion_rate', 0.0),
            'task_success_rate': summary.get('task_success_rate', 0.0),
            'task_failure_rate': summary.get('task_failure_rate', 0.0),
            'task_settlement_rate': summary.get('task_settlement_rate', 0.0),
            'task_resolution_rate': summary.get('task_resolution_rate', 0.0),
            'pending_task_rate': summary.get('pending_task_rate', 0.0),
            'deadline_violation_rate': summary.get('deadline_violation_rate', 0.0),
            'avg_delay': summary.get('avg_delay', 0.0),
            'avg_success_delay': summary.get('avg_success_delay', 0.0),
            'p95_success_delay': summary.get('p95_success_delay', 0.0),
            'handover_frequency': summary.get('handover_frequency', 0.0),
            'handovers_per_user_minute': summary.get('handovers_per_user_minute', 0.0),
            'blocked_time_ratio': summary.get('blocked_time_ratio', 0.0),
            'total_energy': env_stats.get('total_energy', 0.0),
            'energy_per_resolved_task': (
                env_stats.get('total_energy', 0.0) /
                max(float(summary.get('resolved_tasks', 0.0)), 1.0)
            ),
            'energy_per_successful_task': summary.get('energy_per_successful_task', 0.0),
            'load_balance_variance': summary.get('load_balance_variance', 0.0),
            'load_balance_coefficient': summary.get('load_balance_coefficient', 0.0),
            'load_variance_sample_count': summary.get('load_variance_sample_count', 0),
            'mec_load_fairness': summary.get('mec_load_fairness', 0.0),
            'jain_mec_load_fairness': summary.get('jain_mec_load_fairness', 0.0),
            'active_mec_load_fairness': summary.get(
                'active_mec_load_fairness',
                summary.get('mec_load_fairness', 0.0),
            ),
            'reachable_jain_mec_load_fairness': summary.get(
                'reachable_jain_mec_load_fairness',
                summary.get('jain_mec_load_fairness', 0.0),
            ),
            'avg_load_balance_score': summary.get('avg_load_balance_score', 0.0),
            'active_load_balance_score': summary.get('active_load_balance_score', 0.0),
            **reward_breakdown(env_stats),
        }
        if partial_episode:
            record['partial_episode'] = True
        return record

    def train(self):
        self.logger.info("=" * 60)
        self.logger.info(f"开始训练 HAN+{self.algorithm_name.upper()}")
        self.logger.info(f"  总步数: {self.config.total_timesteps:,}")
        self.logger.info(f"  warmup: {self.config.warmup_steps:,}")
        self.logger.info(f"  replay_size: {self.config.replay_size:,}")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info("=" * 60)

        self.training_start_time = time.time()
        observations, _, masks = self._reset_encoded_env(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        update_stats: Dict[str, float] = {}
        update = 0

        for step_idx in range(int(self.config.total_timesteps)):
            update_start = time.time()
            env_actions, action_features, _ = self._select_train_action(observations, masks, step_idx)
            _, rewards, terminated, truncated, _ = self.env.step(
                env_actions,
                return_observation=False,
                return_info=False,
            )
            done = bool(terminated or truncated)
            reward_value = self._scalar_reward(rewards)
            next_observations, _, next_masks, _ = self._encode_graph_state()
            next_masks = self._apply_pre_handover_action_mask(
                next_masks,
                self.env.get_pre_handover_mask(),
            )
            replay_terminal = bool(terminated)
            replay_next_masks = (
                np.zeros_like(masks, dtype=bool)
                if replay_terminal
                else next_masks.astype(bool)
            )
            self.buffer.add(
                observations,
                action_features,
                self._replay_reward(reward_value),
                next_observations,
                replay_terminal,
                masks.astype(bool),
                replay_next_masks,
            )

            if len(self.buffer) >= max(int(self.config.batch_size), int(self.config.warmup_steps), 1):
                update_stats = self.algorithm.update(self.buffer)
                if hasattr(self.algorithm, "_noise_std"):
                    update_stats["exploration_noise"] = self.algorithm._noise_std()

            observations, masks = next_observations, next_masks
            episode_reward += reward_value
            episode_length += 1
            self.total_steps += 1

            if done:
                update += 1
                self.episodes += 1
                self.recent_rewards.append(episode_reward)
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                record = self._record_from_stats(
                    update,
                    episode_reward,
                    episode_length,
                    self.env.get_stats_summary(),
                    update_stats,
                    time.time() - update_start,
                )
                self.training_history.append(record)
                if update % self.config.log_interval == 0:
                    self._log_training(update, record, update_stats, update_start)
                episode_reward = 0.0
                episode_length = 0
                observations, _, masks = self._reset_encoded_env(seed=self.config.seed + self.total_steps)

            if self.total_steps > 0 and self.total_steps % self.config.eval_interval == 0:
                self._evaluate()
            if self.total_steps > 0 and self.total_steps % self.config.save_interval == 0:
                self._save_checkpoint()

        if episode_length > 0:
            update += 1
            self.training_history.append(
                self._record_from_stats(
                    update,
                    episode_reward,
                    episode_length,
                    self.env.stats.copy(),
                    update_stats,
                    0.0,
                    partial_episode=True,
                )
            )

        if self.config.eval_episodes > 0:
            self._evaluate()
        self._save_checkpoint(final=True)
        self._save_training_history()

    def _evaluate(self):
        self.logger.info("-" * 40)
        self.logger.info(f"开始评估 HAN+{self.algorithm_name.upper()}...")
        eval_rewards = []
        eval_lengths = []
        eval_env_stats = self._empty_env_stats()

        training_env = self.env
        eval_env = self._create_eval_env()
        try:
            self.env = eval_env
            for ep in range(self.config.eval_episodes):
                observations, _, masks = self._reset_encoded_env(
                    seed=self.config.seed + EVALUATION_SEED_OFFSET + ep
                )
                episode_reward = 0.0
                episode_length = 0
                done = False
                while not done:
                    env_actions, _, _ = self._select_eval_action(observations, masks)
                    _, rewards, terminated, truncated, _ = self.env.step(
                        env_actions,
                        return_observation=False,
                        return_info=False,
                    )
                    done = bool(terminated or truncated)
                    episode_reward += self._scalar_reward(rewards)
                    episode_length += 1
                    if not done:
                        observations, _, masks, _ = self._encode_graph_state()
                        masks = self._apply_pre_handover_action_mask(
                            masks,
                            self.env.get_pre_handover_mask(),
                        )
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                self._accumulate_env_stats(eval_env_stats, self.env.get_stats_summary())
        finally:
            self.env = training_env
            self._cached_han_user_embed = None
            self._cached_sat_embed = None
            if eval_env is not training_env and hasattr(eval_env, "close"):
                eval_env.close()

        mean_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
        std_reward = float(np.std(eval_rewards)) if eval_rewards else 0.0
        mean_length = float(np.mean(eval_lengths)) if eval_lengths else 0.0
        summary = summarize_env_stats_with_load_balance(eval_env_stats)
        eval_record = {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'eval_mean_reward': mean_reward,
            'eval_std_reward': std_reward,
            'eval_mean_length': mean_length,
            'eval_rewards': [float(r) for r in eval_rewards],
            'total_handovers': eval_env_stats.get('total_handovers', 0),
            'handover_attempts': eval_env_stats.get('handover_attempts', 0),
            'handover_committed': eval_env_stats.get('handover_committed', 0),
            'handover_aborted': eval_env_stats.get('handover_aborted', 0),
            'handover_radio_failures': eval_env_stats.get('handover_radio_failures', 0),
            'migration_rejections': eval_env_stats.get('migration_rejections', 0),
            'reconnection_attempts': eval_env_stats.get('reconnection_attempts', 0),
            'reconnections': eval_env_stats.get('reconnections', 0),
            'successful_handovers': eval_env_stats.get('successful_handovers', 0),
            'failed_handovers': eval_env_stats.get('failed_handovers', 0),
            'forced_disconnects': eval_env_stats.get('forced_disconnects', 0),
            'total_user_seconds': eval_env_stats.get('total_user_seconds', 0.0),
            'blocked_user_seconds': eval_env_stats.get('blocked_user_seconds', 0.0),
            'handover_interruption_seconds': eval_env_stats.get('handover_interruption_seconds', 0.0),
            'service_interruption_seconds': eval_env_stats.get('service_interruption_seconds', 0.0),
            'handover_success_rate': summary.get('handover_success_rate', 0.0),
            'handover_failure_rate': summary.get('handover_failure_rate', 0.0),
            'forced_termination_rate': summary.get('forced_termination_rate', 0.0),
            'service_availability_rate': summary.get('service_availability_rate', 0.0),
            'service_continuity_rate': summary.get('service_continuity_rate', 0.0),
            'successful_task_throughput': summary.get('successful_task_throughput', 0.0),
            'resolved_tasks': summary.get('resolved_tasks', 0),
            'pending_tasks': summary.get('pending_tasks', 0),
            'total_tasks': eval_env_stats.get('total_tasks', 0),
            'completed_tasks': eval_env_stats.get('completed_tasks', 0),
            'deadline_violations': eval_env_stats.get('deadline_violations', 0),
            'failed_tasks': eval_env_stats.get('failed_tasks', 0),
            'task_completion_rate': summary.get('task_completion_rate', 0.0),
            'task_success_rate': summary.get('task_success_rate', 0.0),
            'task_failure_rate': summary.get('task_failure_rate', 0.0),
            'task_settlement_rate': summary.get('task_settlement_rate', 0.0),
            'task_resolution_rate': summary.get('task_resolution_rate', 0.0),
            'pending_task_rate': summary.get('pending_task_rate', 0.0),
            'deadline_violation_rate': summary.get('deadline_violation_rate', 0.0),
            'avg_delay': summary.get('avg_delay', 0.0),
            'avg_success_delay': summary.get('avg_success_delay', 0.0),
            'p95_success_delay': summary.get('p95_success_delay', 0.0),
            'handover_frequency': summary.get('handover_frequency', 0.0),
            'handovers_per_user_minute': summary.get('handovers_per_user_minute', 0.0),
            'blocked_time_ratio': summary.get('blocked_time_ratio', 0.0),
            'total_energy': eval_env_stats.get('total_energy', 0.0),
            'energy_per_resolved_task': (
                eval_env_stats.get('total_energy', 0.0) / max(float(summary.get('resolved_tasks', 0.0)), 1.0)
            ),
            'energy_per_successful_task': summary.get('energy_per_successful_task', 0.0),
            'load_balance_variance': summary.get('load_balance_variance', 0.0),
            'load_balance_coefficient': summary.get('load_balance_coefficient', 0.0),
            'load_variance_sample_count': summary.get('load_variance_sample_count', 0),
            'mec_load_fairness': summary.get('mec_load_fairness', 0.0),
            'jain_mec_load_fairness': summary.get('jain_mec_load_fairness', 0.0),
            'avg_load_balance_score': summary.get('avg_load_balance_score', 0.0),
            'active_load_balance_score': summary.get('active_load_balance_score', 0.0),
        }
        selection_metric = getattr(
            self.config,
            'best_model_metric',
            TrainConfig.best_model_metric,
        )
        eval_record['best_model_metric'] = selection_metric
        eval_record['best_model_score'] = float(compute_model_selection_score(eval_record, selection_metric))
        self.eval_history.append(eval_record)

        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
        if eval_record['best_model_score'] > self.best_model_score:
            self.best_model_score = eval_record['best_model_score']
            self._save_checkpoint(best=True)
        self.logger.info(
            f"评估结果: 奖励 = {mean_reward:.2f} ± {std_reward:.2f}, "
            f"长度 = {mean_length:.0f}, 延迟 = {eval_record['avg_delay']:.3f}s"
        )
        self.logger.info("-" * 40)

    def _algorithm_checkpoint(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm_name,
            'algorithm_train_step': self.algorithm.train_step,
            'actor_state_dict': self.algorithm.actor.state_dict(),
            'target_actor_state_dict': self.algorithm.target_actor.state_dict(),
            'critic_state_dict': self.algorithm.critic.state_dict(),
            'target_critic_state_dict': self.algorithm.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.algorithm.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.algorithm.critic_optimizer.state_dict(),
        }

    def _load_algorithm_checkpoint(self, checkpoint: Dict[str, Any]):
        self.algorithm.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.algorithm.target_actor.load_state_dict(checkpoint.get('target_actor_state_dict', checkpoint['actor_state_dict']))
        self.algorithm.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.algorithm.target_critic.load_state_dict(checkpoint.get('target_critic_state_dict', checkpoint['critic_state_dict']))
        self.algorithm.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.algorithm.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.algorithm.train_step = int(checkpoint.get('algorithm_train_step', 0))

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        save_dir = Path(self.config.save_path)
        if best:
            filename = "best_model.pt"
        elif final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_{self.total_steps}.pt"
        checkpoint = {
            'model_schema_version': MODEL_SCHEMA_VERSION,
            'geometry_schema_version': GEOMETRY_SCHEMA_VERSION,
            'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'best_reward': self.best_reward,
            'best_model_metric': self.config.best_model_metric,
            'best_model_score': self.best_model_score,
            'config': asdict(self.config),
            'han_state_dict': self.han_encoder.state_dict(),
        }
        checkpoint.update(self._algorithm_checkpoint())
        save_path = save_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"模型已保存: {save_path}")

    def load_checkpoint(self, path: str):
        self.logger.info(f"加载检查点: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        model_schema_version = int(checkpoint.get('model_schema_version', 1))
        geometry_schema_version = int(checkpoint.get('geometry_schema_version', 1))
        environment_schema_version = int(
            checkpoint.get('environment_schema_version', 1)
        )
        if (
            model_schema_version != MODEL_SCHEMA_VERSION
            or geometry_schema_version != GEOMETRY_SCHEMA_VERSION
            or environment_schema_version != ENVIRONMENT_SCHEMA_VERSION
        ):
            raise ValueError(
                "检查点的模型、几何或环境 schema 与当前代码不一致，"
                "不能恢复训练或正式比较；请使用当前代码重新训练。"
            )
        self.total_steps = checkpoint['total_steps']
        self.episodes = checkpoint['episodes']
        self.best_reward = checkpoint['best_reward']
        self.best_model_score = checkpoint.get('best_model_score', self.best_model_score)
        if 'best_model_metric' in checkpoint:
            self.config.best_model_metric = checkpoint['best_model_metric']
        self._load_algorithm_checkpoint(checkpoint)
        if 'han_state_dict' in checkpoint:
            self.han_encoder.load_state_dict(checkpoint['han_state_dict'])
        self.logger.info(f"从步数 {self.total_steps:,} 恢复训练")


class HANPDQNTrainer(HANMADDPGTrainer):
    """HAN feature encoder with PDQN over the hybrid handover/offload action."""

    algorithm_name = "pdqn"

    def _epsilon_decay_steps(self) -> int:
        fraction = float(getattr(self.config, "epsilon_decay_fraction", 0.25))
        fraction = min(max(fraction, 0.05), 1.0)
        return max(int(self.config.total_timesteps * fraction), int(self.config.warmup_steps) + 1, 1)

    def _init_mappo(self):
        self.logger.info("初始化 HAN+PDQN...")
        self._load_frozen_pretrained_han()
        pdqn_config = PDQNConfig(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            max_candidates=self.max_candidates,
            q_hidden_dims=(256, 128),
            param_hidden_dims=(128, 64),
            lr=self.config.pdqn_lr,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            replay_size=self.config.replay_size,
            warmup_steps=self.config.warmup_steps,
            target_update_interval=self.config.target_update_interval,
            epsilon_start=self.config.epsilon_start,
            epsilon_final=self.config.epsilon_final,
            epsilon_decay_steps=self._epsilon_decay_steps(),
            grad_clip_norm=self.config.max_grad_norm,
            bc_loss_coef=self.config.bc_loss_coef,
            seed=self.config.seed,
            device=self.config.device,
        )
        self.algorithm = PDQNAlgorithm(pdqn_config)
        self.pdqn = self.algorithm
        q_params = sum(p.numel() for p in self.algorithm.q_net.parameters())
        param_params = sum(p.numel() for p in self.algorithm.param_nets.parameters())
        self.logger.info(f"  - PDQN Q参数量: {q_params:,}")
        self.logger.info(f"  - PDQN 参数网络参数量: {param_params:,}")

    def _select_train_action(self, observations: np.ndarray, masks: np.ndarray, step_idx: int):
        safe_probability = float(
            getattr(self.config, "safe_exploration_probability", 0.0)
        )
        if step_idx < self.config.warmup_steps:
            return self._mixed_safe_random_actions(
                masks.astype(bool),
                safe_probability=safe_probability,
            )
        exploration_actions, _, _ = self._mixed_safe_random_actions(
            masks.astype(bool),
            safe_probability=safe_probability,
        )
        return self.algorithm.act(
            observations,
            masks.astype(bool),
            epsilon=self.algorithm.current_epsilon(),
            exploration_actions=exploration_actions,
        )

    def _select_eval_action(self, observations: np.ndarray, masks: np.ndarray):
        return self.algorithm.act(observations, masks.astype(bool), epsilon=0.0)

    def _algorithm_checkpoint(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm_name,
            'algorithm_train_step': self.algorithm.train_step,
            'q_net_state_dict': self.algorithm.q_net.state_dict(),
            'target_q_net_state_dict': self.algorithm.target_q_net.state_dict(),
            'param_nets_state_dict': self.algorithm.param_nets.state_dict(),
            'target_param_nets_state_dict': self.algorithm.target_param_nets.state_dict(),
            'q_optimizer_state_dict': self.algorithm.q_optimizer.state_dict(),
            'param_optimizer_state_dict': self.algorithm.param_optimizer.state_dict(),
        }

    def _load_algorithm_checkpoint(self, checkpoint: Dict[str, Any]):
        self.algorithm.q_net.load_state_dict(checkpoint['q_net_state_dict'], strict=False)
        self.algorithm.target_q_net.load_state_dict(
            checkpoint.get('target_q_net_state_dict', checkpoint['q_net_state_dict']),
            strict=False,
        )
        self.algorithm.param_nets.load_state_dict(checkpoint['param_nets_state_dict'], strict=False)
        self.algorithm.target_param_nets.load_state_dict(
            checkpoint.get('target_param_nets_state_dict', checkpoint['param_nets_state_dict']),
            strict=False,
        )
        self.algorithm.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.algorithm.param_optimizer.load_state_dict(checkpoint['param_optimizer_state_dict'])
        self.algorithm.train_step = int(checkpoint.get('algorithm_train_step', 0))


# ============================================================
# 命令行接口
# ============================================================

def parse_args():
    """解析命令行参数"""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description='LEO卫星网络切换与任务卸载联合优化训练脚本'
    )
    
    # 实验参数
    parser.add_argument('--exp_name', type=str, default=defaults.exp_name,
                        help='实验名称')
    parser.add_argument('--seed', type=int, default=defaults.seed,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cuda/cpu/auto)')
    parser.add_argument(
        "--algorithm",
        type=str,
        default=defaults.algorithm,
        choices=["mappo", "attn_mappo", "han_attn", "maddpg", "pdqn"],
        help=(
            "Training algorithm: mappo, attn_mappo, han_attn, maddpg, or pdqn"
        ),
    )
    
    # 环境参数
    parser.add_argument('--num_users', type=int, default=defaults.num_users,
                        help='用户数量')
    parser.add_argument('--max_steps', type=int, default=defaults.max_steps,
                        help='每episode最大步数')
    parser.add_argument('--reward_delay_weight', type=float, default=defaults.reward_delay_weight,
                        help='成功任务的归一化时延惩罚权重')
    parser.add_argument('--reward_energy_weight', type=float, default=defaults.reward_energy_weight,
                        help='成功任务的归一化能耗惩罚权重')
    parser.add_argument(
        '--reward_energy_reference_j',
        type=float,
        default=defaults.reward_energy_reference_j,
        help='成功任务能耗归一化参考值（焦耳）',
    )
    parser.add_argument('--reward_interruption_weight', type=float, default=defaults.reward_interruption_weight,
                        help='用户级服务中断比例惩罚权重')
    parser.add_argument('--reward_failed_handover_penalty', type=float, default=defaults.reward_failed_handover_penalty,
                        help='失败切换的固定惩罚')
    parser.add_argument(
        '--reward_load_balance_weight',
        '--reward-load-balance-weight',
        dest='reward_load_balance_weight',
        type=float,
        default=defaults.reward_load_balance_weight,
        help='可达MEC节点Jain公平性的协作奖励权重；设为0可做消融',
    )
    parser.add_argument('--min_effective_offload_ratio', type=float, default=defaults.min_effective_offload_ratio,
                        help='Treat smaller offload ratios as local execution')
    parser.add_argument('--pre_handover_rvt_sec', type=float, default=defaults.pre_handover_rvt_sec,
                        help='Pre-handover RVT threshold (seconds)')
    parser.add_argument('--handover_min_snr_db', type=float, default=defaults.handover_min_snr_db,
                        help='Minimum target-link SNR for handover admission (dB)')
    parser.add_argument(
        '--pretrained_han_path',
        type=str,
        default=defaults.pretrained_han_path,
        help='Trained HAN checkpoint required by replay-based HAN+MADDPG/PDQN',
    )

    # 训练参数
    parser.add_argument('--total_timesteps', type=int, default=defaults.total_timesteps,
                        help='总训练步数')
    parser.add_argument('--n_steps', type=int, default=defaults.n_steps,
                        help='每次更新收集步数')
    parser.add_argument('--learning_rate', type=float, default=defaults.learning_rate,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=defaults.batch_size,
                        help='批大小')
    parser.add_argument('--n_epochs', type=int, default=defaults.n_epochs,
                        help='PPO epochs per update')
    parser.add_argument('--entropy_coef', type=float, default=defaults.entropy_coef,
                        help='Entropy coefficient')
    parser.add_argument('--entropy_schedule', type=str, default=defaults.entropy_schedule, choices=['constant', 'linear'],
                        help='Entropy coefficient schedule')
    parser.add_argument('--bc_loss_coef', type=float, default=defaults.bc_loss_coef,
                        help='PDQN behavior-cloning loss coefficient')
    parser.add_argument(
        '--safe-exploration-probability',
        type=float,
        default=defaults.safe_exploration_probability,
        help='PDQN启发式探索候选概率；默认0以保持算法比较公平',
    )
    
    # HAN参数
    parser.add_argument('--han_hidden_dim', type=int, default=defaults.han_hidden_dim,
                        help='HAN隐藏维度')
    parser.add_argument('--han_num_heads', type=int, default=defaults.han_num_heads,
                        help='注意力头数')
    parser.add_argument('--han_num_layers', type=int, default=defaults.han_num_layers,
                        help='HAN层数')
    
    # 保存加载
    parser.add_argument('--save_path', type=str, default=defaults.save_path,
                        help='模型保存路径')
    parser.add_argument('--log_path', type=str, default=defaults.log_path,
                        help='日志保存路径')
    parser.add_argument('--load_path', type=str, default=None,
                        help='加载检查点路径')
    parser.add_argument('--save_interval', type=int, default=defaults.save_interval,
                        help='检查点保存间隔（步）')
    parser.add_argument('--log_interval', type=int, default=defaults.log_interval,
                        help='训练日志打印间隔（update）')
    
    # 评估
    parser.add_argument('--eval_interval', type=int, default=defaults.eval_interval,
                        help='评估间隔')
    parser.add_argument('--eval_episodes', type=int, default=defaults.eval_episodes,
                        help='每次评估的episode数')
    parser.add_argument('--graph_update_interval', type=int, default=defaults.graph_update_interval,
                        help='图重建间隔（步），增大可提速')
    parser.add_argument('--early_stop_patience', type=int, default=defaults.early_stop_patience,
                        help='连续多少次更新无改善后早停，0表示禁用')
    parser.add_argument('--value_loss_type', type=str, default=defaults.value_loss_type, choices=['mse', 'huber'],
                        help='Critic损失类型')
    parser.add_argument('--disable_return_norm', action='store_true',
                        help='禁用returns标准化（默认开启）')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估，不训练')
    parser.add_argument('--best-model-metric', type=str, default=defaults.best_model_metric,
                        choices=list(BEST_MODEL_METRIC_CHOICES),
                        help='best_model.pt 的选优指标')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # argparse 与 TrainConfig 同名字段直接同步，特殊字段单独处理。
    config = TrainConfig()
    for name, value in vars(args).items():
        if hasattr(config, name):
            setattr(config, name, value)
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.normalize_returns = not args.disable_return_norm
    
    # 创建训练器
    trainer_cls = {
        "mappo": HANMAPPOTrainer,
        "attn_mappo": AttentionMAPPOTrainer,
        "han_attn": HANCandidateAttentionMAPPOTrainer,
        "maddpg": HANMADDPGTrainer,
        "pdqn": HANPDQNTrainer,
    }[config.algorithm]
    trainer = trainer_cls(config)
    
    # 加载检查点
    if config.load_path:
        trainer.load_checkpoint(config.load_path)
    
    # 训练或评估
    if args.eval_only:
        trainer._evaluate()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
