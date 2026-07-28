"""
Walker星座模型
使用poliastro进行精确轨道计算
实现论文中的星座建模方法
"""

import logging

import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 地球参数
EARTH_RADIUS_KM = 6371.0  # 地球平均半径
EARTH_MU = 398600.4418    # 地球引力常数 km³/s²


@dataclass
class OrbitalElements:
    """
    轨道六根数
    用于描述卫星轨道的基本参数
    """
    semi_major_axis: float    # 半长轴 (km)
    eccentricity: float       # 偏心率 (无量纲)
    inclination: float        # 轨道倾角 (度)
    raan: float               # 升交点赤经 (度)
    arg_periapsis: float      # 近地点幅角 (度)
    true_anomaly: float       # 真近点角 (度)


class WalkerConstellation:
    """
    Walker星座模型
    
    Walker星座表示为 i:T/P/F，其中：
    - i: 轨道倾角
    - T: 总卫星数
    - P: 轨道平面数
    - F: 相位因子 (0 ≤ F < P)
    
    参考论文配置：
    - 6个轨道平面，每平面11颗卫星，共66颗
    - 轨道高度550km，倾角53°
    """
    
    def __init__(
        self,
        num_planes: int = 6,
        sats_per_plane: int = 11,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        phase_factor: int = 1,
        start_time: Optional[datetime] = None
    ):
        """
        初始化Walker星座
        
        Args:
            num_planes: 轨道平面数 P
            sats_per_plane: 每平面卫星数 S
            altitude_km: 轨道高度 (km)
            inclination_deg: 轨道倾角 (度)
            phase_factor: 相位因子 F
            start_time: 仿真起始时间
        """
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.total_sats = num_planes * sats_per_plane
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.phase_factor = phase_factor
        
        # 计算轨道参数
        self.semi_major_axis = EARTH_RADIUS_KM + altitude_km  # 半长轴
        self.orbital_period = self._compute_orbital_period()   # 轨道周期
        
        # 时间管理
        self.start_time = start_time or datetime(2026, 1, 15, 0, 0, 0)
        self.current_time = self.start_time
        
        # 初始化所有卫星的轨道根数
        self.orbital_elements: List[OrbitalElements] = []
        self._initialize_constellation()
        
        logger.info(
            "Walker constellation initialized: %s:%s/%s/%s, altitude=%s km, "
            "period=%.2f min, total_sats=%s",
            inclination_deg,
            self.total_sats,
            num_planes,
            phase_factor,
            altitude_km,
            self.orbital_period / 60.0,
            self.total_sats,
        )
    
    def _compute_orbital_period(self) -> float:
        """
        计算轨道周期 (秒)
        
        使用开普勒第三定律：
        T = 2π * sqrt(a³/μ)
        """
        a_km = self.semi_major_axis
        period_seconds = 2 * np.pi * np.sqrt(a_km**3 / EARTH_MU)
        return period_seconds
    
    def _initialize_constellation(self):
        """
        初始化星座中所有卫星的轨道根数
        
        Walker星座的相位配置：
        - 轨道平面均匀分布在赤道面上
        - 同一平面内卫星均匀分布
        - 不同平面间存在相位偏移
        """
        for plane_idx in range(self.num_planes):
            # 升交点赤经 (RAAN) - 轨道平面均匀分布
            raan = (360.0 / self.num_planes) * plane_idx
            
            for sat_idx in range(self.sats_per_plane):
                # 真近点角 - 同平面卫星均匀分布 + 跨平面相位偏移
                # 相位偏移 = F * 360 / T，其中T为总卫星数
                phase_offset = (self.phase_factor * 360.0 / self.total_sats) * plane_idx
                true_anomaly = (360.0 / self.sats_per_plane) * sat_idx + phase_offset
                true_anomaly = true_anomaly % 360.0  # 归一化到 [0, 360)
                
                # 创建轨道根数
                orbital_elem = OrbitalElements(
                    semi_major_axis=self.semi_major_axis,
                    eccentricity=0.0,  # 圆轨道
                    inclination=self.inclination_deg,
                    raan=raan,
                    arg_periapsis=0.0,  # 圆轨道时无意义
                    true_anomaly=true_anomaly
                )
                self.orbital_elements.append(orbital_elem)
                
        # 预计算向量化数组（必须在_update_all_positions之前）
        self._init_vectorized_arrays()
        # 计算初始位置
        self._update_all_positions(self.start_time)
    
    def _init_vectorized_arrays(self):
        """预计算向量化所需的常量数组，避免每步重复创建"""
        N = self.total_sats
        # 所有卫星的初始真近点角 (度)
        self._ta0 = np.array([e.true_anomaly for e in self.orbital_elements])
        # 所有卫星的RAAN (rad)
        raans = np.array([e.raan for e in self.orbital_elements])
        self._cos_O = np.cos(np.radians(raans))
        self._sin_O = np.sin(np.radians(raans))
        # 倾角 (所有卫星相同)
        i_rad = np.radians(self.inclination_deg)
        self._cos_i = np.cos(i_rad)
        self._sin_i = np.sin(i_rad)
        # 圆轨道 omega=0 => cos_w=1, sin_w=0，旋转矩阵简化
        # R列向量预计算: R @ [cos(nu)*r, sin(nu)*r, 0]
        # R[:,0] = [cos_O, sin_O, 0]  (因为cos_w=1, sin_w=0)
        # R[:,1] = [-cos_O*0 - sin_O*1*cos_i, -sin_O*0 + cos_O*1*cos_i, ... ] 简化后:
        # R[:,0] = [cos_O, sin_O, 0]
        # R[:,1] = [-sin_O*cos_i, cos_O*cos_i, sin_i]
        self._R0 = np.column_stack([self._cos_O, self._sin_O, np.zeros(N)])  # (N,3)
        self._R1 = np.column_stack([-self._sin_O * self._cos_i,
                                     self._cos_O * self._cos_i,
                                     np.full(N, self._sin_i)])  # (N,3)
        # 平均角速度
        self._mean_motion = 2 * np.pi / self.orbital_period  # rad/s
        # 半长轴 (圆轨道 r=a)
        self._a = self.semi_major_axis
        # 速度大小 v = sqrt(mu/a) for circular orbit
        self._v_circ = np.sqrt(EARTH_MU / self._a)
        
        # 预分配位置/速度数组
        self._all_pos_eci = np.zeros((N, 3))
        self._all_vel_eci = np.zeros((N, 3))
        self._all_pos_ecef = np.zeros((N, 3))
        self._all_pos_lla = np.zeros((N, 3))

    def _update_all_positions(self, time: datetime):
        """向量化更新所有卫星位置（批量计算，无Python循环）"""
        dt = (time - self.start_time).total_seconds()
        N = self.total_sats
        
        # 批量计算新真近点角 (rad)
        nu = np.radians(self._ta0) + self._mean_motion * dt  # (N,)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)
        
        # 圆轨道: r = a, pos_eci = R @ [a*cos(nu), a*sin(nu), 0]
        # = a*cos(nu)*R[:,0] + a*sin(nu)*R[:,1]
        a = self._a
        self._all_pos_eci[:] = (a * cos_nu)[:, None] * self._R0 + (a * sin_nu)[:, None] * self._R1
        
        # 速度: vel_eci = v*(-sin(nu)*R[:,0] + cos(nu)*R[:,1])
        v = self._v_circ
        self._all_vel_eci[:] = (v * (-sin_nu))[:, None] * self._R0 + (v * cos_nu)[:, None] * self._R1
        
        # ECI -> ECEF: 绕Z轴旋转GMST角
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        days_since_j2000 = (time - j2000).total_seconds() / 86400.0
        gmst_rad = np.radians((280.46061837 + 360.98564736629 * days_since_j2000) % 360.0)
        cos_g, sin_g = np.cos(gmst_rad), np.sin(gmst_rad)
        
        px, py, pz = self._all_pos_eci[:, 0], self._all_pos_eci[:, 1], self._all_pos_eci[:, 2]
        self._all_pos_ecef[:, 0] = cos_g * px + sin_g * py
        self._all_pos_ecef[:, 1] = -sin_g * px + cos_g * py
        self._all_pos_ecef[:, 2] = pz
        
        # ECEF -> LLA (球面近似，向量化)
        r = np.linalg.norm(self._all_pos_ecef, axis=1)
        self._all_pos_lla[:, 0] = np.degrees(np.arcsin(self._all_pos_ecef[:, 2] / r))  # lat
        self._all_pos_lla[:, 1] = np.degrees(np.arctan2(self._all_pos_ecef[:, 1], self._all_pos_ecef[:, 0]))  # lon
        self._all_pos_lla[:, 2] = r - EARTH_RADIUS_KM  # alt
        
        self.current_time = time
    
    def reset(self, time_offset_sec: float = 0.0):
        """
        重置星座到初始状态

        Args:
            time_offset_sec: 从 start_time 偏移的秒数，用于 episode 随机化
        """
        self.current_time = self.start_time + timedelta(seconds=float(time_offset_sec))
        self._update_all_positions(self.current_time)
    
    def propagate(self, delta_seconds: float):
        """
        向前推进仿真时间
        
        Args:
            delta_seconds: 推进的时间 (秒)
        """
        new_time = self.current_time + timedelta(seconds=delta_seconds)
        self._update_all_positions(new_time)

    def get_all_positions_ecef_at_offsets(
        self,
        offsets_seconds: np.ndarray,
    ) -> np.ndarray:
        """返回当前时刻之后多个偏移量对应的全部卫星 ECEF 位置。

        该方法只做纯计算，不修改星座当前状态，供 RVT 预测使用。

        Returns:
            形状为 ``(num_offsets, total_sats, 3)`` 的 ECEF 位置数组。
        """
        offsets = np.asarray(offsets_seconds, dtype=np.float64).reshape(-1)
        absolute_dt = (
            (self.current_time - self.start_time).total_seconds() + offsets
        )
        nu = (
            np.radians(self._ta0)[None, :]
            + self._mean_motion * absolute_dt[:, None]
        )
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        positions_eci = (
            (self._a * cos_nu)[:, :, None] * self._R0[None, :, :]
            + (self._a * sin_nu)[:, :, None] * self._R1[None, :, :]
        )

        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        base_days = (self.current_time - j2000).total_seconds() / 86400.0
        future_days = base_days + offsets / 86400.0
        gmst_rad = np.radians(
            (
                280.46061837
                + 360.98564736629 * future_days
            )
            % 360.0
        )
        cos_g = np.cos(gmst_rad)[:, None]
        sin_g = np.sin(gmst_rad)[:, None]

        positions_ecef = np.empty_like(positions_eci)
        px = positions_eci[:, :, 0]
        py = positions_eci[:, :, 1]
        positions_ecef[:, :, 0] = cos_g * px + sin_g * py
        positions_ecef[:, :, 1] = -sin_g * px + cos_g * py
        positions_ecef[:, :, 2] = positions_eci[:, :, 2]
        return positions_ecef
