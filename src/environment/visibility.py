"""
卫星可见性计算模块
实现论文中的关键功能：
1. 仰角计算
2. 可见性判断
3. 剩余可见时间 (RVT) 预测 - 用于预切换决策
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 地球参数
EARTH_RADIUS_KM = 6371.0


@dataclass
class VisibilityInfo:
    """
    单个卫星对用户的可见性信息
    """
    sat_id: int
    is_visible: bool           # 是否可见
    elevation_deg: float       # 仰角 (度)
    azimuth_deg: float         # 方位角 (度)
    distance_km: float         # 距离 (km)
    rvt_seconds: float         # 剩余可见时间 (秒)


class VisibilityCalculator:
    """
    可见性计算器
    
    实现论文中的可见性判断和剩余可见时间预测
    这是预切换机制的核心组件
    """
    
    def __init__(self, min_elevation_deg: float = 10.0):
        """
        Args:
            min_elevation_deg: 最小仰角阈值 (度)
                              低于此仰角的卫星视为不可见
        """
        self.min_elevation_deg = min_elevation_deg
        
    def lla_to_ecef(self, lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
        """
        将经纬度高度转换为ECEF坐标
        
        Args:
            lat_deg: 纬度 (度)
            lon_deg: 经度 (度)
            alt_km: 海拔高度 (km)，地面用户默认为0
            
        Returns:
            ECEF坐标 [x, y, z] (km)
        """
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        r = EARTH_RADIUS_KM + alt_km
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def compute_elevation_azimuth(
        self,
        user_pos_ecef: np.ndarray,
        sat_pos_ecef: np.ndarray,
        user_lat_deg: float,
        user_lon_deg: float
    ) -> Tuple[float, float, float]:
        """
        计算卫星相对于地面用户的仰角、方位角和距离
        
        仰角 (Elevation): 卫星方向与地平面的夹角
        方位角 (Azimuth): 卫星方向在地平面投影与正北的夹角
        
        Args:
            user_pos_ecef: 用户ECEF位置 (km)
            sat_pos_ecef: 卫星ECEF位置 (km)
            user_lat_deg: 用户纬度 (度)
            user_lon_deg: 用户经度 (度)
            
        Returns:
            (仰角, 方位角, 距离)
        """
        # 1. 计算指向卫星的向量 (ECEF系)
        vec_to_sat = sat_pos_ecef - user_pos_ecef
        distance = np.linalg.norm(vec_to_sat)
        
        if distance < 1e-6:
            return 90.0, 0.0, 0.0  # 卫星在正上方
        
        # 2. 构建ENU (East-North-Up) 坐标系
        # 这是以用户位置为原点的局部坐标系
        lat_rad = np.radians(user_lat_deg)
        lon_rad = np.radians(user_lon_deg)
        
        # ENU基向量 (在ECEF系中的表示)
        # East: 指向东方
        e_east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
        
        # North: 指向北方
        e_north = np.array([
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad)
        ])
        
        # Up: 指向天顶 (垂直于地表向上)
        e_up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])
        
        # 3. 将卫星向量投影到ENU系
        east_component = np.dot(vec_to_sat, e_east)
        north_component = np.dot(vec_to_sat, e_north)
        up_component = np.dot(vec_to_sat, e_up)
        
        # 4. 计算仰角
        horizontal_dist = np.sqrt(east_component**2 + north_component**2)
        elevation_rad = np.arctan2(up_component, horizontal_dist)
        elevation_deg = np.degrees(elevation_rad)
        
        # 5. 计算方位角 (从正北顺时针)
        azimuth_rad = np.arctan2(east_component, north_component)
        azimuth_deg = np.degrees(azimuth_rad)
        if azimuth_deg < 0:
            azimuth_deg += 360.0
            
        return elevation_deg, azimuth_deg, distance
    
    def is_visible(self, elevation_deg: float) -> bool:
        """
        判断卫星是否可见
        
        Args:
            elevation_deg: 仰角 (度)
            
        Returns:
            是否可见
        """
        return elevation_deg >= self.min_elevation_deg
    
    def estimate_rvt(
        self,
        user_pos_ecef: np.ndarray,
        sat_pos_ecef: np.ndarray,
        sat_vel_eci: np.ndarray,
        user_lat_deg: float,
        user_lon_deg: float,
        current_elevation: float,
        time_step: float = 1.0,
        max_time: float = 600.0
    ) -> float:
        """
        估算剩余可见时间 (Remaining Visible Time, RVT)
        
        这是论文中预切换机制的关键参数
        当RVT低于阈值时触发预切换
        
        方法：通过向前预测卫星轨迹来估算RVT
        
        Args:
            user_pos_ecef: 用户ECEF位置 (km)
            sat_pos_ecef: 当前卫星ECEF位置 (km)
            sat_vel_eci: 卫星ECI速度 (km/s)，用于预测
            user_lat_deg: 用户纬度
            user_lon_deg: 用户经度
            current_elevation: 当前仰角
            time_step: 预测时间步长 (秒)
            max_time: 最大预测时间 (秒)
            
        Returns:
            剩余可见时间 (秒)
        """
        if current_elevation < self.min_elevation_deg:
            return 0.0
        
        # 简化预测：假设卫星做匀速运动
        # 更精确的方法需要考虑轨道曲率
        rvt = 0.0
        predicted_sat_pos = sat_pos_ecef.copy()
        
        # 注：这里用ECI速度近似，实际应用中需要更精确的轨道传播
        # 由于时间尺度较短（几分钟），误差可接受
        sat_speed = np.linalg.norm(sat_vel_eci)
        
        # 计算卫星轨道的角速度
        orbit_radius = np.linalg.norm(sat_pos_ecef)
        angular_velocity = sat_speed / orbit_radius  # rad/s
        
        for t in np.arange(time_step, max_time, time_step):
            # 简化：假设卫星沿圆轨道运动
            # 计算卫星绕地心旋转后的位置
            angle = angular_velocity * t
            
            # 构建旋转矩阵 (绕轨道法向量旋转)
            # 这里用简化计算，实际应用中应使用精确轨道传播
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 获取轨道法向量 (假设速度方向垂直于位置向量)
            orbit_normal = np.cross(sat_pos_ecef, sat_vel_eci)
            orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)
            
            # Rodrigues旋转公式
            predicted_sat_pos = (
                sat_pos_ecef * cos_a +
                np.cross(orbit_normal, sat_pos_ecef) * sin_a +
                orbit_normal * np.dot(orbit_normal, sat_pos_ecef) * (1 - cos_a)
            )
            
            # 计算预测位置的仰角
            elev, _, _ = self.compute_elevation_azimuth(
                user_pos_ecef, predicted_sat_pos, user_lat_deg, user_lon_deg
            )
            
            if elev < self.min_elevation_deg:
                # 卫星将变为不可见
                rvt = t - time_step  # 返回上一个可见时刻
                break
            
            rvt = t
        
        return rvt
    
    def compute_visibility_for_user(
        self,
        user_lat: float,
        user_lon: float,
        satellite_positions: np.ndarray,
        satellite_velocities: np.ndarray
    ) -> List[VisibilityInfo]:
        """
        计算用户对所有卫星的可见性信息
        
        Args:
            user_lat: 用户纬度 (度)
            user_lon: 用户经度 (度)
            satellite_positions: 所有卫星ECEF位置 (N, 3)
            satellite_velocities: 所有卫星ECI速度 (N, 3)
            
        Returns:
            每颗卫星的可见性信息列表
        """
        user_pos_ecef = self.lla_to_ecef(user_lat, user_lon, 0.0)
        
        visibility_list = []
        
        for sat_id in range(len(satellite_positions)):
            sat_pos = satellite_positions[sat_id]
            sat_vel = satellite_velocities[sat_id]
            
            # 计算仰角、方位角、距离
            elevation, azimuth, distance = self.compute_elevation_azimuth(
                user_pos_ecef, sat_pos, user_lat, user_lon
            )
            
            # 判断可见性
            is_vis = self.is_visible(elevation)
            
            # 计算RVT (仅对可见卫星)
            if is_vis:
                rvt = self.estimate_rvt(
                    user_pos_ecef, sat_pos, sat_vel,
                    user_lat, user_lon, elevation
                )
            else:
                rvt = 0.0
            
            visibility_list.append(VisibilityInfo(
                sat_id=sat_id,
                is_visible=is_vis,
                elevation_deg=elevation,
                azimuth_deg=azimuth,
                distance_km=distance,
                rvt_seconds=rvt
            ))
        
        return visibility_list
    
    def get_visible_satellites(
        self,
        visibility_list: List[VisibilityInfo]
    ) -> List[VisibilityInfo]:
        """
        获取所有可见卫星的列表
        
        Args:
            visibility_list: 完整的可见性列表
            
        Returns:
            仅包含可见卫星的列表
        """
        return [v for v in visibility_list if v.is_visible]
    
    def get_best_satellite(
        self,
        visibility_list: List[VisibilityInfo],
        criterion: str = 'elevation'
    ) -> Optional[VisibilityInfo]:
        """
        根据指定准则获取最优卫星
        
        Args:
            visibility_list: 可见性列表
            criterion: 选择准则
                - 'elevation': 选择仰角最高的 (信号质量最好)
                - 'rvt': 选择RVT最长的 (服务时间最长)
                - 'distance': 选择距离最近的 (传播时延最小)
                
        Returns:
            最优卫星的可见性信息
        """
        visible_sats = self.get_visible_satellites(visibility_list)
        
        if not visible_sats:
            return None
        
        if criterion == 'elevation':
            return max(visible_sats, key=lambda x: x.elevation_deg)
        elif criterion == 'rvt':
            return max(visible_sats, key=lambda x: x.rvt_seconds)
        elif criterion == 'distance':
            return min(visible_sats, key=lambda x: x.distance_km)
        else:
            raise ValueError(f"未知准则: {criterion}")