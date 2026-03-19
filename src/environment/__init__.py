"""
环境仿真模块
包含星座、卫星、用户、信道等模型
"""

from .constellation import WalkerConstellation
from .visibility import VisibilityCalculator, VisibilityInfo
from .user import User, UserPosition, UserState, UserGenerator, UserManager
from .task import Task, TaskType, TaskStatus, TaskGenerator, TaskManager, TaskConfig
from .channel import SatelliteChannel, ChannelConfig, MultiUserChannel
from .mec import MECServer, MECConfig, MECManager, OffloadingCalculator, ComputeResult

# Gymnasium环境（需要安装gymnasium）
try:
    from .gym_env import LEOSatelliteEnv, EnvConfig, make_env
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

__all__ = [
    'WalkerConstellation',
    'VisibilityCalculator',
    'VisibilityInfo',
    'User',
    'UserPosition', 
    'UserState',
    'UserGenerator',
    'UserManager',
    'Task',
    'TaskType',
    'TaskStatus',
    'TaskGenerator',
    'TaskManager',
    'TaskConfig',
    'SatelliteChannel',
    'ChannelConfig',
    'MultiUserChannel',
    'MECServer',
    'MECConfig',
    'MECManager',
    'OffloadingCalculator',
    'ComputeResult',
]

# 条件导出Gymnasium环境
if _GYM_AVAILABLE:
    __all__.extend(['LEOSatelliteEnv', 'EnvConfig', 'make_env'])