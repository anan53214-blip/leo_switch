"""
MEC（移动边缘计算）模型
实现星载MEC的计算资源管理和任务处理

主要功能：
1. 本地计算时延和能耗
2. 卫星MEC计算时延
3. 任务卸载决策的时延/能耗计算
4. 多用户资源分配

参考论文：
- 宋晓勤等《基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法》
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .channel import SatelliteChannel, ChannelConfig


@dataclass
class MECConfig:
    """
    MEC配置参数
    """
    # 卫星MEC参数
    satellite_cpu_freq_ghz: float = 5.0       # 卫星CPU频率 (GHz) - 降低以增加竞争
    satellite_max_cpu_freq_ghz: float = 8.0   # 最大CPU频率 (GHz)
    satellite_num_cores: int = 2              # CPU核心数，总算力 5 GHz × 2 = 10 Gcycle/s
    mec_max_concurrent_tasks: int = 2          # 最多同时处理的任务数，其余任务FCFS等待
    max_queue_size: int = 6                   # 最大任务队列长度 - 大幅缩小，迫使智能选择
    
    # 用户本地计算参数
    user_cpu_freq_ghz: float = 1.0            # 用户设备CPU频率 (GHz)
    user_max_cpu_freq_ghz: float = 1.5        # 最大CPU频率 (GHz)
    
    # 能耗参数
    kappa: float = 1e-27                      # 能耗系数 (J/cycle³)
    user_idle_power_w: float = 0.05           # 空闲功率 (W)
    
    # 传输参数
    result_ratio: float = 0.1                 # 结果数据与输入数据比例
    

@dataclass
class ComputeResult:
    """
    计算结果
    包含时延和能耗的详细分解
    """
    # 时延分解 (秒)
    local_compute_delay: float = 0.0      # 本地计算时延
    upload_delay: float = 0.0             # 上传时延
    satellite_compute_delay: float = 0.0  # 卫星计算时延
    download_delay: float = 0.0           # 下载时延
    total_delay: float = 0.0              # 总时延
    
    # 能耗分解 (焦耳)
    local_compute_energy: float = 0.0     # 本地计算能耗
    upload_energy: float = 0.0            # 上传能耗
    total_energy: float = 0.0             # 总能耗
    
    # 是否满足时延约束
    deadline_met: bool = True


@dataclass(frozen=True)
class MigrationPlan:
    """用于原子迁移提交的只读任务快照。"""
    user_id: int
    old_sat_id: int
    new_sat_id: int
    task_ids: Tuple[int, ...]
    required_slots: int
    feasible: bool
    failure_reason: Optional[str] = None
    

class MECServer:
    """
    卫星MEC服务器模型
    
    管理单颗卫星的计算资源和任务队列
    """
    
    def __init__(
        self,
        satellite_id: int,
        config: Optional[MECConfig] = None
    ):
        """
        Args:
            satellite_id: 卫星ID
            config: MEC配置
        """
        self.satellite_id = satellite_id
        self.config = config or MECConfig()
        
        # CPU状态
        self.cpu_freq_ghz = self.config.satellite_cpu_freq_ghz
        self.total_capacity_ghz = self.cpu_freq_ghz * max(self.config.satellite_num_cores, 1)
        self.available_freq_ghz = self.total_capacity_ghz  # 可用计算资源
        
        # 任务队列
        self.task_queue: List[Dict] = []
        self.current_load = 0.0  # 当前负载 [0, 1]
        
        # 已连接用户
        self.connected_users: List[int] = []
        
        # 统计信息
        self.total_tasks_processed = 0
        self.total_compute_cycles = 0
    
    @property
    def queue_length(self) -> int:
        """任务队列长度"""
        return len(self.task_queue)
    
    @property
    def is_full(self) -> bool:
        """队列是否已满"""
        return self.queue_length >= self.config.max_queue_size
    
    @property
    def utilization(self) -> float:
        """CPU利用率"""
        return float(np.clip(self.current_load, 0.0, 1.0))
    
    def add_user(self, user_id: int):
        """添加连接用户"""
        if user_id not in self.connected_users:
            self.connected_users.append(user_id)
    
    def remove_user(self, user_id: int):
        """移除用户"""
        if user_id in self.connected_users:
            self.connected_users.remove(user_id)
    
    def allocate_compute_resource(
        self,
        user_id: int,
        requested_freq_ghz: float
    ) -> float:
        """
        为用户分配计算资源
        
        Args:
            user_id: 用户ID
            requested_freq_ghz: 请求的CPU频率 (GHz)
            
        Returns:
            实际分配的CPU频率 (GHz)
        """
        # 限制分配量
        allocated = min(requested_freq_ghz, self.available_freq_ghz)
        allocated = max(allocated, 0.1)  # 最小分配
        
        self.available_freq_ghz -= allocated
        self.available_freq_ghz = max(self.available_freq_ghz, 0)
        
        return allocated
    
    def release_compute_resource(self, freq_ghz: float):
        """释放计算资源"""
        self.available_freq_ghz = min(
            self.available_freq_ghz + freq_ghz,
            self.total_capacity_ghz
        )
    
    def compute_processing_delay(
        self,
        computation_cycles: float,
        allocated_freq_ghz: Optional[float] = None
    ) -> float:
        """
        计算任务处理时延
        
        T_sat = C / f_sat
        
        Args:
            computation_cycles: 计算量 (cycles)
            allocated_freq_ghz: 分配的CPU频率，None时使用平均分配
            
        Returns:
            处理时延 (秒)
        """
        if allocated_freq_ghz is None:
            # 平均分配给所有用户
            num_users = max(len(self.connected_users), 1)
            allocated_freq_ghz = self.total_capacity_ghz / num_users
        
        freq_hz = allocated_freq_ghz * 1e9
        delay = computation_cycles / freq_hz
        
        return delay
    
    def get_state_vector(self) -> np.ndarray:
        """
        获取MEC状态向量（用于神经网络输入）
        
        Returns:
            状态向量 [cpu_util, queue_len, num_users, available_freq]
        """
        return np.array([
            self.utilization,
            self.queue_length / self.config.max_queue_size,
            len(self.connected_users) / 50.0,  # 归一化
            self.available_freq_ghz / max(
                self.config.satellite_max_cpu_freq_ghz * max(self.config.satellite_num_cores, 1),
                1e-6,
            ),
        ])
    
    def reset(self):
        """重置MEC状态"""
        self.available_freq_ghz = self.total_capacity_ghz
        self.task_queue.clear()
        self.connected_users.clear()
        self.current_load = 0.0
        self.total_tasks_processed = 0
        self.total_compute_cycles = 0
    # ==================== 任务队列管理（竞争机制） ====================
    
    def enqueue_task(
        self,
        user_id: int,
        task_id: int,
        offload_cycles: float,
        offload_data_bits: float,
        max_delay: float,
        arrival_time: float,
        upload_delay: float = 0.0,
        download_delay: float = 0.0,
        offload_ratio: float = 1.0,
        upload_energy: float = 0.0,
    ) -> bool:
        """
        将任务加入队列等待处理
        
        Args:
            user_id: 用户ID
            task_id: 任务ID
            offload_cycles: 需要卫星处理的计算量 (cycles)
            offload_data_bits: 卸载的数据量 (bits)
            max_delay: 最大容忍时延 (秒)
            arrival_time: 任务到达时间 (秒)
            upload_delay: 上传时延 (秒)
            download_delay: 下载时延 (秒)
            offload_ratio: 卸载比例
            upload_energy: 上传能耗 (焦耳)
            
        Returns:
            是否成功入队（队列未满时返回 True）
        """
        if self.is_full:
            return False
        
        task_entry = {
            'user_id': user_id,
            'task_id': task_id,
            'offload_cycles': offload_cycles,
            'remaining_cycles': offload_cycles,  # 剩余待处理量
            'offload_data_bits': offload_data_bits,
            'max_delay': max_delay,
            'arrival_time': arrival_time,
            'upload_delay': upload_delay,
            'download_delay': download_delay,
            'offload_ratio': offload_ratio,
            'upload_energy': upload_energy,
            'start_processing_time': None,  # 开始处理时间
            'status': 'queued',  # queued / processing / completed / timeout
        }
        self.task_queue.append(task_entry)
        return True
    
    def process_queue(self, current_time: float, time_step: float) -> List[Dict]:
        """
        按时间步处理队列中的任务（FCFS，有限并行槽共享 CPU）
        
        Args:
            current_time: 当前仿真时间 (秒)
            time_step: 本次时间步长 (秒)
            
        Returns:
            本步完成的任务列表 (包含完成状态、时延等信息)
        """
        completed_this_step: List[Dict] = []
        
        if not self.task_queue:
            # 无任务时释放所有资源
            self.available_freq_ghz = self.total_capacity_ghz
            self.current_load = 0.0
            return completed_this_step
        
        max_concurrent = max(int(self.config.mec_max_concurrent_tasks), 1)
        processing_tasks = [
            task for task in self.task_queue
            if task['status'] == 'processing'
        ]

        # 正常调度不会超过并行槽。迁移或外部恢复若带来额外 processing
        # 任务，则按队列顺序保留前 max_concurrent 个，其余重新进入 FCFS。
        for task in processing_tasks[max_concurrent:]:
            task['status'] = 'queued'
            task['start_processing_time'] = None
        processing_tasks = processing_tasks[:max_concurrent]

        available_slots = max_concurrent - len(processing_tasks)
        if available_slots > 0:
            for task in self.task_queue:
                if available_slots <= 0:
                    break
                if task['status'] != 'queued':
                    continue
                task['status'] = 'processing'
                task['start_processing_time'] = current_time
                processing_tasks.append(task)
                available_slots -= 1

        active_tasks = processing_tasks
        num_active = len(active_tasks)
        
        if num_active == 0:
            self.available_freq_ghz = self.total_capacity_ghz
            self.current_load = 0.0
            return completed_this_step
        
        # 平均分配 CPU 频率给所有活跃任务
        freq_per_task_ghz = self.total_capacity_ghz / num_active
        cycles_per_task = freq_per_task_ghz * 1e9 * time_step  # 本步可处理的 cycles
        
        # 本 slot 实际需求占总容量的比例。旧实现只要队列非空就恒为 1，
        # 导致观测、负载公平性和绘图指标退化成二值量。
        slot_capacity_cycles = self.total_capacity_ghz * 1e9 * time_step
        requested_cycles = sum(
            max(float(task['remaining_cycles']), 0.0)
            for task in active_tasks
        )
        self.current_load = float(np.clip(
            requested_cycles / max(slot_capacity_cycles, 1e-9),
            0.0,
            1.0,
        ))
        self.available_freq_ghz = self.total_capacity_ghz * (1.0 - self.current_load)
        
        tasks_to_remove = []
        
        slot_end = float(current_time) + float(time_step)
        processing_rate_cycles_per_sec = freq_per_task_ghz * 1e9

        for task in active_tasks:
            # 处理计算
            remaining_before = max(float(task['remaining_cycles']), 0.0)
            task['remaining_cycles'] -= cycles_per_task
            
            # 检查是否完成
            if task['remaining_cycles'] <= 0:
                task['remaining_cycles'] = 0
                task['status'] = 'completed'

                # 任务可能在 slot 内提前完成，完成时刻按实际消耗 cycles 插值。
                completion_offset = (
                    remaining_before / processing_rate_cycles_per_sec
                    if processing_rate_cycles_per_sec > 0.0 else float(time_step)
                )
                completion_offset = float(np.clip(completion_offset, 0.0, time_step))
                finish_time = float(current_time) + completion_offset
                
                # 计算总时延 = 上传 + 排队等待 + 处理 + 下载
                queue_wait = task['start_processing_time'] - task['arrival_time']
                processing_time = finish_time - task['start_processing_time']
                total_delay = task['upload_delay'] + queue_wait + processing_time + task['download_delay']
                
                task['total_delay'] = total_delay
                task['queue_wait'] = queue_wait
                task['processing_time'] = processing_time
                task['deadline_met'] = total_delay <= task['max_delay']
                task['finish_time'] = finish_time
                
                completed_this_step.append(task)
                tasks_to_remove.append(task)
                
                self.total_tasks_processed += 1
                self.total_compute_cycles += task['offload_cycles']
            else:
                # 检查是否超时（任务仍在处理但已超过 deadline）
                elapsed = slot_end - task['arrival_time'] + task['upload_delay']
                if elapsed > task['max_delay']:
                    task['status'] = 'timeout'
                    task['total_delay'] = elapsed
                    task['queue_wait'] = max(
                        float(task['start_processing_time'])
                        - float(task['arrival_time']),
                        0.0,
                    )
                    task['processing_time'] = max(
                        slot_end - float(task['start_processing_time']),
                        0.0,
                    )
                    task['deadline_met'] = False
                    task['finish_time'] = slot_end
                    
                    completed_this_step.append(task)
                    tasks_to_remove.append(task)

        # queued 任务不占用 CPU，但仍会随墙钟时间累计等待并可能超时。
        for task in self.task_queue:
            if task['status'] != 'queued' or task in tasks_to_remove:
                continue
            elapsed = slot_end - float(task['arrival_time']) + float(task['upload_delay'])
            if elapsed <= float(task['max_delay']):
                continue
            task['status'] = 'timeout'
            task['total_delay'] = elapsed
            task['queue_wait'] = max(
                slot_end - float(task['arrival_time']),
                0.0,
            )
            task['processing_time'] = 0.0
            task['deadline_met'] = False
            task['finish_time'] = slot_end
            completed_this_step.append(task)
            tasks_to_remove.append(task)
        
        # 移除已完成/超时的任务
        for t in tasks_to_remove:
            self.task_queue.remove(t)
        
        # 在 slot 末尾立即按 FCFS 补充空闲槽，使下一时隙的处理状态与观测一致。
        remaining_processing = [
            task for task in self.task_queue
            if task['status'] == 'processing'
        ]
        available_slots = max_concurrent - len(remaining_processing)
        if available_slots > 0:
            for task in self.task_queue:
                if available_slots <= 0:
                    break
                if task['status'] != 'queued':
                    continue
                task['status'] = 'processing'
                task['start_processing_time'] = slot_end
                remaining_processing.append(task)
                available_slots -= 1

        if not remaining_processing:
            self.available_freq_ghz = self.total_capacity_ghz
            self.current_load = 0.0
        else:
            self.available_freq_ghz = 0.0
            self.current_load = 1.0
        
        return completed_this_step
    
    def get_estimated_wait_time(self) -> float:
        """
        估算新任务的等待时间（基于当前队列）
        
        Returns:
            估算等待时间 (秒)
        """
        if not self.task_queue:
            return 0.0
        
        total_remaining_cycles = sum(t['remaining_cycles'] for t in self.task_queue)
        # 假设新任务需要等待当前所有任务完成（悲观估计）
        wait_time = total_remaining_cycles / (self.total_capacity_ghz * 1e9)
        return wait_time


class OffloadingCalculator:
    """
    任务卸载计算器
    
    计算不同卸载策略下的时延和能耗
    实现论文中的核心公式
    """
    
    def __init__(
        self,
        mec_config: Optional[MECConfig] = None,
        channel_config: Optional[ChannelConfig] = None
    ):
        """
        Args:
            mec_config: MEC配置
            channel_config: 信道配置
        """
        self.mec_config = mec_config or MECConfig()
        self.channel = SatelliteChannel(channel_config)
    
    def compute_local_delay(
        self,
        computation_cycles: float,
        cpu_freq_ghz: Optional[float] = None
    ) -> float:
        """
        计算本地执行时延
        
        T_local = C / f_local
        
        Args:
            computation_cycles: 计算量 (cycles)
            cpu_freq_ghz: 本地CPU频率 (GHz)
            
        Returns:
            本地计算时延 (秒)
        """
        if cpu_freq_ghz is None:
            cpu_freq_ghz = self.mec_config.user_cpu_freq_ghz
        
        freq_hz = cpu_freq_ghz * 1e9
        return computation_cycles / freq_hz
    
    def compute_local_energy(
        self,
        computation_cycles: float,
        cpu_freq_ghz: Optional[float] = None
    ) -> float:
        """
        计算本地执行能耗
        
        E_local = κ * C * f²
        
        其中 κ 是能耗系数（与芯片架构相关）
        
        Args:
            computation_cycles: 计算量 (cycles)
            cpu_freq_ghz: CPU频率 (GHz)
            
        Returns:
            能耗 (焦耳)
        """
        if cpu_freq_ghz is None:
            cpu_freq_ghz = self.mec_config.user_cpu_freq_ghz
        
        freq_hz = cpu_freq_ghz * 1e9
        kappa = self.mec_config.kappa
        
        # E = κ * C * f²
        energy = kappa * computation_cycles * (freq_hz ** 2)
        
        return energy
    
    def compute_transmission_delay(
        self,
        data_size_bits: float,
        distance_km: float,
        elevation_deg: float,
        bandwidth_mhz: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算传输时延（上传+下载）
        
        T_trans = D / R + d / c
        
        Args:
            data_size_bits: 上传数据量 (bits)
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            bandwidth_mhz: 分配带宽 (MHz)
            
        Returns:
            (上传时延, 下载时延) (秒)
        """
        # 上传时延
        upload_delay = self.channel.compute_transmission_delay(
            data_size_bits,
            distance_km,
            elevation_deg,
            'uplink',
            bandwidth_mhz,
        )
        
        # 下载数据量 (结果通常比输入小)
        result_bits = data_size_bits * self.mec_config.result_ratio
        
        # 下载时延
        download_delay = self.channel.compute_transmission_delay(
            result_bits, distance_km, elevation_deg, 'downlink'
        )
        
        return upload_delay, download_delay
    
    def compute_transmission_energy(
        self,
        data_size_bits: float,
        distance_km: float,
        elevation_deg: float,
        tx_power_w: Optional[float] = None,
        bandwidth_mhz: Optional[float] = None,
    ) -> float:
        """
        计算用户终端实际上传取电能耗

        E_trans = (P_RF / eta_PA + P_circuit) * T_airtime

        Args:
            data_size_bits: 上传数据量 (bits)
            distance_km: 距离 (km)
            elevation_deg: 仰角 (度)
            tx_power_w: 天线辐射功率 (W)
            bandwidth_mhz: OFDMA实际分配的上行带宽 (MHz)
            
        Returns:
            传输能耗 (焦耳)
        """
        if tx_power_w is None:
            tx_power_w = self.channel._dbm_to_watt(
                self.channel.config.user_tx_power_dbm
            )

        pa_efficiency = float(self.channel.config.user_pa_efficiency)
        if not 0.0 < pa_efficiency <= 1.0:
            raise ValueError("user_pa_efficiency必须位于(0, 1]区间")
        circuit_power_w = max(
            float(self.channel.config.user_circuit_power_w),
            0.0,
        )

        # 电池只在实际发射比特期间供电；传播时延仍计入任务时延，但不计入
        # 本批数据的终端发射能耗。
        upload_rate_bps = self.channel.compute_channel_capacity(
            distance_km,
            elevation_deg,
            'uplink',
            bandwidth_mhz,
        )
        upload_airtime = (
            float(data_size_bits) / upload_rate_bps
            if upload_rate_bps > 0.0
            else float('inf')
        )
        terminal_draw_power_w = (
            float(tx_power_w) / pa_efficiency + circuit_power_w
        )
        energy = terminal_draw_power_w * upload_airtime
        
        return energy
    
    def compute_offloading_result(
        self,
        data_size_bits: float,
        computation_cycles: float,
        max_delay: float,
        offload_ratio: float,
        distance_km: float,
        elevation_deg: float,
        satellite_freq_ghz: Optional[float] = None,
        local_freq_ghz: Optional[float] = None
    ) -> ComputeResult:
        """
        计算给定卸载比例下的时延和能耗
        
        这是核心函数，实现论文中的任务卸载模型
        
        Args:
            data_size_bits: 任务数据量 (bits)
            computation_cycles: 任务计算量 (cycles)
            max_delay: 最大容忍时延 (秒)
            offload_ratio: 卸载比例 λ ∈ [0, 1]
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            satellite_freq_ghz: 卫星分配的CPU频率
            local_freq_ghz: 本地CPU频率
            
        Returns:
            ComputeResult 包含详细的时延和能耗分解
        """
        result = ComputeResult()
        
        # 使用默认值
        if satellite_freq_ghz is None:
            satellite_freq_ghz = (
                self.mec_config.satellite_cpu_freq_ghz *
                max(self.mec_config.satellite_num_cores, 1)
            )
        if local_freq_ghz is None:
            local_freq_ghz = self.mec_config.user_cpu_freq_ghz
        
        # 确保卸载比例在[0, 1]范围内
        offload_ratio = np.clip(offload_ratio, 0.0, 1.0)
        
        # ========== 本地计算部分 ==========
        local_cycles = (1 - offload_ratio) * computation_cycles
        
        if local_cycles > 0:
            result.local_compute_delay = self.compute_local_delay(
                local_cycles, local_freq_ghz
            )
            result.local_compute_energy = self.compute_local_energy(
                local_cycles, local_freq_ghz
            )
        
        # ========== 卸载部分 ==========
        if offload_ratio > 0:
            # 上传数据量
            upload_bits = offload_ratio * data_size_bits
            
            # 传输时延
            result.upload_delay, result.download_delay = self.compute_transmission_delay(
                upload_bits, distance_km, elevation_deg
            )
            
            # 传输能耗
            result.upload_energy = self.compute_transmission_energy(
                upload_bits, distance_km, elevation_deg
            )
            
            # 卫星计算时延
            offload_cycles = offload_ratio * computation_cycles
            result.satellite_compute_delay = offload_cycles / (satellite_freq_ghz * 1e9)
        
        # ========== 总时延计算 ==========
        # 本地和卸载并行执行，取最大值
        offload_total = (
            result.upload_delay + 
            result.satellite_compute_delay + 
            result.download_delay
        )
        
        result.total_delay = max(result.local_compute_delay, offload_total)
        
        # ========== 总能耗计算 ==========
        result.total_energy = result.local_compute_energy + result.upload_energy
        
        # ========== 检查时延约束 ==========
        result.deadline_met = result.total_delay <= max_delay
        
        return result
    
    def find_optimal_offload_ratio(
        self,
        data_size_bits: float,
        computation_cycles: float,
        max_delay: float,
        distance_km: float,
        elevation_deg: float,
        satellite_freq_ghz: Optional[float] = None,
        objective: str = 'delay',
        num_samples: int = 100
    ) -> Tuple[float, ComputeResult]:
        """
        寻找最优卸载比例
        
        通过网格搜索找到最小化目标（时延或能耗）的卸载比例
        
        Args:
            data_size_bits: 数据量
            computation_cycles: 计算量
            max_delay: 最大时延
            distance_km: 距离
            elevation_deg: 仰角
            satellite_freq_ghz: 卫星CPU频率
            objective: 优化目标 'delay', 'energy', 或 'weighted'
            num_samples: 搜索精度
            
        Returns:
            (最优卸载比例, 对应的计算结果)
        """
        best_ratio = 0.0
        best_result = None
        best_objective = float('inf')
        candidates = []

        for ratio in np.linspace(0, 1, num_samples):
            result = self.compute_offloading_result(
                data_size_bits=data_size_bits,
                computation_cycles=computation_cycles,
                max_delay=max_delay,
                offload_ratio=ratio,
                distance_km=distance_km,
                elevation_deg=elevation_deg,
                satellite_freq_ghz=satellite_freq_ghz
            )
            candidates.append((ratio, result))

        feasible_results = [
            result for _, result in candidates
            if result.deadline_met
        ]
        max_energy = max(
            (float(result.total_energy) for result in feasible_results),
            default=max(
                (float(result.total_energy) for _, result in candidates),
                default=1.0,
            ),
        )
        max_energy = max(max_energy, 1e-12)

        for ratio, result in candidates:
            # 计算目标值
            if objective == 'delay':
                obj_value = result.total_delay
            elif objective == 'energy':
                obj_value = result.total_energy
            else:  # weighted
                # 归一化后加权
                obj_value = 0.5 * (result.total_delay / max_delay) + \
                           0.5 * (result.total_energy / max_energy)
            
            # 只考虑满足时延约束的解
            if result.deadline_met and obj_value < best_objective:
                best_objective = obj_value
                best_ratio = ratio
                best_result = result
        
        # 如果没有满足约束的解，返回最小时延的方案
        if best_result is None:
            best_ratio = 1.0  # 完全卸载通常时延最小
            best_result = self.compute_offloading_result(
                data_size_bits, computation_cycles, max_delay,
                best_ratio, distance_km, elevation_deg, satellite_freq_ghz
            )
        
        return best_ratio, best_result


class MECManager:
    """
    MEC管理器
    
    管理多颗卫星的MEC服务器
    """
    
    def __init__(
        self,
        num_satellites: int,
        config: Optional[MECConfig] = None
    ):
        """
        Args:
            num_satellites: 卫星数量
            config: MEC配置
        """
        self.config = config or MECConfig()
        self.num_satellites = num_satellites
        
        # 创建所有卫星的MEC服务器
        self.servers: Dict[int, MECServer] = {}
        for sat_id in range(num_satellites):
            self.servers[sat_id] = MECServer(sat_id, self.config)
        
        # 卸载计算器
        self.calculator = OffloadingCalculator(config)
    
    def get_server(self, satellite_id: int) -> Optional[MECServer]:
        """获取指定卫星的MEC服务器"""
        return self.servers.get(satellite_id)
    
    def get_all_utilizations(self) -> np.ndarray:
        """获取所有卫星的CPU利用率"""
        utils = np.zeros(self.num_satellites)
        for sat_id, server in self.servers.items():
            utils[sat_id] = server.utilization
        return utils
    
    def get_all_queue_lengths(self) -> np.ndarray:
        """获取所有卫星的队列长度"""
        lengths = np.zeros(self.num_satellites)
        for sat_id, server in self.servers.items():
            lengths[sat_id] = server.queue_length
        return lengths
    
    def find_best_satellite(
        self,
        candidate_satellites: List[int],
        distances_km: Dict[int, float],
        elevations_deg: Dict[int, float]
    ) -> int:
        """
        从候选卫星中选择最佳卸载目标
        
        考虑因素：距离、负载、队列长度
        
        Args:
            candidate_satellites: 候选卫星ID列表
            distances_km: 各卫星距离
            elevations_deg: 各卫星仰角
            
        Returns:
            最佳卫星ID
        """
        if not candidate_satellites:
            return -1
        
        best_sat = candidate_satellites[0]
        best_score = float('inf')
        
        for sat_id in candidate_satellites:
            server = self.servers.get(sat_id)
            if server is None or server.is_full:
                continue
            
            # 综合评分：距离 + 负载
            distance = distances_km.get(sat_id, float('inf'))
            load = server.utilization
            queue = server.queue_length / self.config.max_queue_size
            
            # 加权评分 (越小越好)
            score = 0.4 * (distance / 2000) + 0.3 * load + 0.3 * queue
            
            if score < best_score:
                best_score = score
                best_sat = sat_id
        
        return best_sat
    
    def prepare_user_task_migration(
        self,
        user_id: int,
        old_sat_id: int,
        new_sat_id: int,
    ) -> MigrationPlan:
        """只做容量与任务快照检查，不修改任一队列。"""
        old_server = self.servers.get(old_sat_id)
        new_server = self.servers.get(new_sat_id)

        if new_server is None:
            return MigrationPlan(
                user_id, old_sat_id, new_sat_id, (), 0, False,
                'target_server_unavailable',
            )
        if old_sat_id < 0:
            return MigrationPlan(user_id, old_sat_id, new_sat_id, (), 0, True)
        if old_server is None:
            return MigrationPlan(
                user_id, old_sat_id, new_sat_id, (), 0, False,
                'source_server_unavailable',
            )
        if old_server is new_server:
            return MigrationPlan(
                user_id, old_sat_id, new_sat_id, (), 0, False,
                'same_server',
            )

        user_tasks = [
            task for task in old_server.task_queue
            if task['user_id'] == user_id
        ]
        task_ids = tuple(int(task['task_id']) for task in user_tasks)
        available_slots = max(
            int(new_server.config.max_queue_size) - new_server.queue_length,
            0,
        )
        feasible = len(user_tasks) <= available_slots
        return MigrationPlan(
            user_id=user_id,
            old_sat_id=old_sat_id,
            new_sat_id=new_sat_id,
            task_ids=task_ids,
            required_slots=len(user_tasks),
            feasible=feasible,
            failure_reason=None if feasible else 'target_queue_capacity',
        )

    def commit_user_task_migration(
        self,
        plan: MigrationPlan,
        handover_delay: float = 0.0,
    ) -> Dict[str, Any]:
        """迁移计划全部满足时一次提交，否则两个队列均保持不变。"""
        failed_result = {
            'migrated': 0,
            'failed': int(plan.required_slots),
            'migrated_task_ids': [],
            'failed_task_ids': list(plan.task_ids),
            'failure_reason': plan.failure_reason,
        }
        if not plan.feasible:
            return failed_result

        new_server = self.servers.get(plan.new_sat_id)
        old_server = self.servers.get(plan.old_sat_id)
        if new_server is None:
            return {**failed_result, 'failure_reason': 'target_server_unavailable'}
        if plan.old_sat_id < 0:
            return {
                'migrated': 0,
                'failed': 0,
                'migrated_task_ids': [],
                'failed_task_ids': [],
                'failure_reason': None,
            }
        if old_server is None or old_server is new_server:
            reason = 'source_server_unavailable' if old_server is None else 'same_server'
            return {**failed_result, 'failure_reason': reason}

        current_tasks = [
            task for task in old_server.task_queue
            if task['user_id'] == plan.user_id
        ]
        current_ids = tuple(int(task['task_id']) for task in current_tasks)
        if current_ids != plan.task_ids:
            return {**failed_result, 'failure_reason': 'source_tasks_changed'}
        available_slots = max(
            int(new_server.config.max_queue_size) - new_server.queue_length,
            0,
        )
        if len(current_tasks) > available_slots:
            return {**failed_result, 'failure_reason': 'target_queue_capacity_changed'}

        for task in current_tasks:
            task['upload_delay'] += float(handover_delay)
            # 迁移保留已经完成的 cycles，但必须重新参加目标 MEC 的 FCFS
            # 调度，不能携带 processing 状态绕过目标服务器的并行槽上限。
            task['status'] = 'queued'
            task['start_processing_time'] = None
        old_server.task_queue = [
            task for task in old_server.task_queue
            if task['user_id'] != plan.user_id
        ]
        new_server.task_queue.extend(current_tasks)
        return {
            'migrated': len(current_tasks),
            'failed': 0,
            'migrated_task_ids': list(plan.task_ids),
            'failed_task_ids': [],
            'failure_reason': None,
        }

    def remove_user_tasks(self, sat_id: int, user_id: int) -> List[Dict[str, Any]]:
        """移除并返回用户在指定卫星上的全部未完成任务。"""
        server = self.servers.get(sat_id)
        if server is None:
            return []
        removed = [
            task for task in server.task_queue
            if task['user_id'] == user_id
        ]
        if removed:
            server.task_queue = [
                task for task in server.task_queue
                if task['user_id'] != user_id
            ]
        return removed
    
    def reset_all(self):
        """重置所有MEC服务器"""
        for server in self.servers.values():
            server.reset()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_tasks = sum(s.total_tasks_processed for s in self.servers.values())
        total_cycles = sum(s.total_compute_cycles for s in self.servers.values())
        avg_util = np.mean([s.utilization for s in self.servers.values()])
        
        return {
            'total_tasks_processed': total_tasks,
            'total_compute_cycles': total_cycles,
            'average_utilization': avg_util,
            'num_overloaded': sum(1 for s in self.servers.values() if s.utilization > 0.9),
        }
    
    def process_all_queues(self, current_time: float, time_step: float) -> List[Dict]:
        """
        处理所有卫星的任务队列
        
        Args:
            current_time: 当前仿真时间 (秒)
            time_step: 本次时间步长 (秒)
            
        Returns:
            本步所有卫星完成的任务列表
        """
        all_completed = []
        for sat_id, server in self.servers.items():
            completed = server.process_queue(current_time, time_step)
            for task in completed:
                task['satellite_id'] = sat_id  # 标记处理该任务的卫星
            all_completed.extend(completed)
        return all_completed
    
    def get_total_queue_length(self) -> int:
        """获取所有卫星队列总长度"""
        return sum(s.queue_length for s in self.servers.values())


# ==================== 便捷函数 ====================

def compute_task_delay(
    data_mb: float,
    compute_gcycles: float,
    offload_ratio: float,
    distance_km: float,
    elevation_deg: float
) -> float:
    """
    快速计算任务时延
    
    Args:
        data_mb: 数据量 (MB)
        compute_gcycles: 计算量 (G cycles)
        offload_ratio: 卸载比例
        distance_km: 距离 (km)
        elevation_deg: 仰角 (度)
        
    Returns:
        总时延 (秒)
    """
    calc = OffloadingCalculator()
    result = calc.compute_offloading_result(
        data_size_bits=data_mb * 8 * 1e6,
        computation_cycles=compute_gcycles * 1e9,
        max_delay=10.0,
        offload_ratio=offload_ratio,
        distance_km=distance_km,
        elevation_deg=elevation_deg
    )
    return result.total_delay


def compute_task_energy(
    data_mb: float,
    compute_gcycles: float,
    offload_ratio: float,
    distance_km: float,
    elevation_deg: float
) -> float:
    """
    快速计算任务能耗
    
    Args:
        data_mb: 数据量 (MB)
        compute_gcycles: 计算量 (G cycles)
        offload_ratio: 卸载比例
        distance_km: 距离 (km)
        elevation_deg: 仰角 (度)
        
    Returns:
        总能耗 (焦耳)
    """
    calc = OffloadingCalculator()
    result = calc.compute_offloading_result(
        data_size_bits=data_mb * 8 * 1e6,
        computation_cycles=compute_gcycles * 1e9,
        max_delay=10.0,
        offload_ratio=offload_ratio,
        distance_km=distance_km,
        elevation_deg=elevation_deg
    )
    return result.total_energy
