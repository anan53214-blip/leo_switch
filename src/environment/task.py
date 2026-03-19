"""
计算任务模型
实现论文中的任务卸载模型：T = {D, C, T_max}
- D: 任务数据量 (bits)
- C: 计算量 (CPU cycles)
- T_max: 最大容忍时延 (s)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """任务类型"""
    LIGHT = 0      # 轻量级任务 (如传感器数据)
    MEDIUM = 1     # 中等任务 (如图像识别)
    HEAVY = 2      # 重型任务 (如视频处理)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = 0    # 等待处理
    OFFLOADING = 1 # 正在卸载传输
    COMPUTING = 2  # 正在计算
    COMPLETED = 3  # 已完成
    FAILED = 4     # 失败 (超时)


@dataclass
class Task:
    """
    计算任务模型
    
    对应论文公式: T_i(t) = {D_i(t), C_i(t), T_i^max(t)}
    """
    task_id: int
    user_id: int
    
    # 任务参数
    data_size: float          # D: 数据量 (bits)
    computation: float        # C: 计算量 (CPU cycles)
    max_delay: float          # T_max: 最大容忍时延 (s)
    
    # 任务类型
    task_type: TaskType = TaskType.MEDIUM
    
    # 状态管理
    status: TaskStatus = TaskStatus.PENDING
    creation_time: float = 0.0
    completion_time: float = -1.0
    
    # 卸载决策 (由算法决定)
    offload_ratio: float = 0.0      # λ: 卸载比例 [0, 1]
    target_satellite: int = -1       # 目标卫星ID
    allocated_bandwidth: float = 0.0 # 分配的带宽 (Hz)
    allocated_compute: float = 0.0   # 分配的计算资源 (Hz)
    
    # 实际时延记录
    local_delay: float = 0.0         # 本地计算时延
    transmission_delay: float = 0.0  # 传输时延
    satellite_delay: float = 0.0     # 卫星计算时延
    total_delay: float = 0.0         # 总时延
    
    # 能耗记录
    local_energy: float = 0.0        # 本地计算能耗
    transmission_energy: float = 0.0 # 传输能耗
    total_energy: float = 0.0        # 总能耗
    
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """检查任务是否失败"""
        return self.status == TaskStatus.FAILED
    
    def get_data_size_MB(self) -> float:
        """获取数据量 (MB)"""
        return self.data_size / (8 * 1024 * 1024)
    
    def get_computation_GCycles(self) -> float:
        """获取计算量 (G cycles)"""
        return self.computation / 1e9


@dataclass
class TaskConfig:
    """
    任务配置参数
    定义不同类型任务的参数范围
    """
    # 轻量级任务
    light_data_range: Tuple[float, float] = (0.5e6, 2e6)      # 0.5-2 Mbits
    light_compute_range: Tuple[float, float] = (0.1e9, 0.5e9) # 0.1-0.5 G cycles
    light_delay_range: Tuple[float, float] = (0.5, 2.0)       # 0.5-2 s
    
    # 中等任务
    medium_data_range: Tuple[float, float] = (2e6, 10e6)      # 2-10 Mbits
    medium_compute_range: Tuple[float, float] = (0.5e9, 2e9)  # 0.5-2 G cycles
    medium_delay_range: Tuple[float, float] = (1.0, 5.0)      # 1-5 s
    
    # 重型任务
    heavy_data_range: Tuple[float, float] = (10e6, 50e6)      # 10-50 Mbits
    heavy_compute_range: Tuple[float, float] = (2e9, 10e9)    # 2-10 G cycles
    heavy_delay_range: Tuple[float, float] = (2.0, 10.0)      # 2-10 s


class TaskGenerator:
    """
    任务生成器
    为用户生成随机计算任务
    """
    
    def __init__(
        self,
        config: Optional[TaskConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            config: 任务配置
            seed: 随机种子
        """
        self.config = config or TaskConfig()
        self.rng = np.random.default_rng(seed)
        self.task_counter = 0
    
    def generate_task(
        self,
        user_id: int,
        task_type: Optional[TaskType] = None,
        current_time: float = 0.0
    ) -> Task:
        """
        为指定用户生成一个任务
        
        Args:
            user_id: 用户ID
            task_type: 任务类型（None则随机选择）
            current_time: 当前时间
            
        Returns:
            生成的任务
        """
        # 随机选择任务类型
        if task_type is None:
            # 按概率分布：轻量30%, 中等50%, 重型20%
            task_type = self.rng.choice(
                [TaskType.LIGHT, TaskType.MEDIUM, TaskType.HEAVY],
                p=[0.3, 0.5, 0.2]
            )
        
        # 根据类型获取参数范围
        if task_type == TaskType.LIGHT:
            data_range = self.config.light_data_range
            compute_range = self.config.light_compute_range
            delay_range = self.config.light_delay_range
        elif task_type == TaskType.MEDIUM:
            data_range = self.config.medium_data_range
            compute_range = self.config.medium_compute_range
            delay_range = self.config.medium_delay_range
        else:  # HEAVY
            data_range = self.config.heavy_data_range
            compute_range = self.config.heavy_compute_range
            delay_range = self.config.heavy_delay_range
        
        # 在范围内随机生成参数
        data_size = self.rng.uniform(*data_range)
        computation = self.rng.uniform(*compute_range)
        max_delay = self.rng.uniform(*delay_range)
        
        task = Task(
            task_id=self.task_counter,
            user_id=user_id,
            data_size=data_size,
            computation=computation,
            max_delay=max_delay,
            task_type=task_type,
            creation_time=current_time
        )
        
        self.task_counter += 1
        return task
    
    def generate_tasks_for_users(
        self,
        user_ids: List[int],
        current_time: float = 0.0
    ) -> List[Task]:
        """
        为多个用户生成任务
        
        Args:
            user_ids: 用户ID列表
            current_time: 当前时间
            
        Returns:
            任务列表
        """
        tasks = []
        for user_id in user_ids:
            task = self.generate_task(user_id, current_time=current_time)
            tasks.append(task)
        return tasks


class TaskManager:
    """
    任务管理器
    管理所有任务的生命周期
    """
    
    def __init__(self):
        self.pending_tasks: Dict[int, Task] = {}     # 等待中的任务
        self.active_tasks: Dict[int, Task] = {}      # 正在处理的任务
        self.completed_tasks: List[Task] = []        # 已完成的任务
        self.failed_tasks: List[Task] = []           # 失败的任务
        
        # 统计信息
        self.total_tasks = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
    
    def add_task(self, task: Task):
        """添加新任务"""
        self.pending_tasks[task.task_id] = task
        self.total_tasks += 1
    
    def start_task(self, task_id: int):
        """开始处理任务"""
        if task_id in self.pending_tasks:
            task = self.pending_tasks.pop(task_id)
            task.status = TaskStatus.OFFLOADING
            self.active_tasks[task_id] = task
    
    def complete_task(self, task_id: int, current_time: float):
        """完成任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.completion_time = current_time
            task.total_delay = current_time - task.creation_time
            
            # 检查是否超时
            if task.total_delay > task.max_delay:
                task.status = TaskStatus.FAILED
                self.failed_tasks.append(task)
            else:
                self.completed_tasks.append(task)
                self.total_delay += task.total_delay
                self.total_energy += task.total_energy
    
    def fail_task(self, task_id: int, current_time: float):
        """任务失败"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.completion_time = current_time
            self.failed_tasks.append(task)
        elif task_id in self.pending_tasks:
            task = self.pending_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            self.failed_tasks.append(task)
    
    def get_user_tasks(self, user_id: int) -> List[Task]:
        """获取指定用户的所有任务"""
        all_tasks = (
            list(self.pending_tasks.values()) +
            list(self.active_tasks.values()) +
            self.completed_tasks +
            self.failed_tasks
        )
        return [t for t in all_tasks if t.user_id == user_id]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        
        avg_delay = self.total_delay / max(completed_count, 1)
        avg_energy = self.total_energy / max(completed_count, 1)
        success_rate = completed_count / max(self.total_tasks, 1)
        
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'success_rate': success_rate,
            'average_delay': avg_delay,
            'average_energy': avg_energy,
            'total_delay': self.total_delay,
            'total_energy': self.total_energy
        }
    
    def print_status(self):
        """打印当前状态"""
        stats = self.get_statistics()
        print(f"\n任务统计:")
        print(f"  总任务数: {stats['total_tasks']}")
        print(f"  已完成: {stats['completed_tasks']} (成功率: {stats['success_rate']*100:.1f}%)")
        print(f"  失败: {stats['failed_tasks']}")
        print(f"  等待中: {stats['pending_tasks']}")
        print(f"  处理中: {stats['active_tasks']}")
        print(f"  平均时延: {stats['average_delay']*1000:.1f} ms")