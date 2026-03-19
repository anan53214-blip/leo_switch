"""
用户模型
实现地面用户的位置、状态和连接管理
支持在指定区域内随机生成用户
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# 地球参数
EARTH_RADIUS_KM = 6371.0


class UserState(Enum):
    """用户连接状态"""
    IDLE = 0          # 空闲，未连接任何卫星
    CONNECTED = 1     # 已连接卫星，正常服务
    HANDOVER = 2      # 正在切换中
    BLOCKED = 3       # 阻塞，无可用卫星


@dataclass
class UserPosition:
    """用户位置信息"""
    latitude: float       # 纬度 (度)
    longitude: float      # 经度 (度)
    altitude: float = 0.0 # 海拔高度 (km)，地面用户默认为0
    
    def to_ecef(self) -> np.ndarray:
        """转换为ECEF坐标"""
        lat_rad = np.radians(self.latitude)
        lon_rad = np.radians(self.longitude)
        r = EARTH_RADIUS_KM + self.altitude
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return np.array([x, y, z])


@dataclass
class User:
    """
    地面用户模型
    
    属性:
        user_id: 用户唯一标识
        position: 用户位置
        state: 当前连接状态
        serving_satellite: 当前服务卫星ID (-1表示未连接)
        candidate_satellites: 候选卫星列表
        handover_count: 切换次数统计
        service_time: 累计服务时间
    """
    user_id: int
    position: UserPosition
    state: UserState = UserState.IDLE
    serving_satellite: int = -1
    candidate_satellites: List[int] = field(default_factory=list)
    
    # 统计信息
    handover_count: int = 0
    successful_handovers: int = 0
    failed_handovers: int = 0
    total_service_time: float = 0.0
    total_blocked_time: float = 0.0
    
    # 切换相关
    handover_start_time: float = -1.0  # 切换开始时间
    last_update_time: float = 0.0      # 上次更新时间
    
    def get_ecef_position(self) -> np.ndarray:
        """获取ECEF坐标"""
        return self.position.to_ecef()
    
    def connect_to_satellite(self, sat_id: int, current_time: float):
        """
        连接到指定卫星
        
        Args:
            sat_id: 目标卫星ID
            current_time: 当前时间
        """
        self.serving_satellite = sat_id
        self.state = UserState.CONNECTED
        self.last_update_time = current_time
    
    def start_handover(self, target_sat_id: int, current_time: float):
        """
        开始切换过程
        
        Args:
            target_sat_id: 目标卫星ID
            current_time: 当前时间
        """
        self.state = UserState.HANDOVER
        self.handover_start_time = current_time
        self.handover_count += 1
    
    def complete_handover(self, new_sat_id: int, current_time: float, success: bool = True):
        """
        完成切换
        
        Args:
            new_sat_id: 新卫星ID
            current_time: 当前时间
            success: 切换是否成功
        """
        if success:
            self.serving_satellite = new_sat_id
            self.state = UserState.CONNECTED
            self.successful_handovers += 1
        else:
            self.state = UserState.BLOCKED
            self.failed_handovers += 1
        
        self.handover_start_time = -1.0
        self.last_update_time = current_time
    
    def disconnect(self, current_time: float):
        """断开连接"""
        self.serving_satellite = -1
        self.state = UserState.IDLE
        self.last_update_time = current_time
    
    def update_statistics(self, current_time: float):
        """
        更新统计信息
        
        Args:
            current_time: 当前时间
        """
        dt = current_time - self.last_update_time
        
        if self.state == UserState.CONNECTED:
            self.total_service_time += dt
        elif self.state == UserState.BLOCKED:
            self.total_blocked_time += dt
        
        self.last_update_time = current_time
    
    def get_state_vector(self) -> np.ndarray:
        """
        获取用户状态向量（用于神经网络输入）
        
        Returns:
            状态向量 [lat, lon, state, serving_sat, ...]
        """
        return np.array([
            self.position.latitude / 90.0,        # 归一化纬度
            self.position.longitude / 180.0,      # 归一化经度
            self.state.value / 3.0,               # 归一化状态
            self.serving_satellite / 100.0,       # 归一化卫星ID
        ])


class UserGenerator:
    """
    用户生成器
    在指定区域内随机生成用户
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 随机种子，用于复现实验
        """
        self.rng = np.random.default_rng(seed)
    
    def generate_users_in_circle(
        self,
        center_lat: float,
        center_lon: float,
        radius_deg: float,
        num_users: int
    ) -> List[User]:
        """
        在圆形区域内均匀随机生成用户
        
        使用极坐标方法确保用户均匀分布
        
        Args:
            center_lat: 圆心纬度 (度)
            center_lon: 圆心经度 (度)
            radius_deg: 半径 (度)
            num_users: 用户数量
            
        Returns:
            用户列表
        """
        users = []
        
        for i in range(num_users):
            # 使用极坐标生成均匀分布的点
            # r = radius * sqrt(uniform(0,1)) 确保面积均匀
            r = radius_deg * np.sqrt(self.rng.uniform(0, 1))
            theta = self.rng.uniform(0, 2 * np.pi)
            
            # 转换为经纬度偏移
            # 注意：经度需要根据纬度进行修正
            lat_offset = r * np.cos(theta)
            lon_offset = r * np.sin(theta) / np.cos(np.radians(center_lat))
            
            user_lat = center_lat + lat_offset
            user_lon = center_lon + lon_offset
            
            # 确保经纬度在有效范围内
            user_lat = np.clip(user_lat, -90, 90)
            user_lon = ((user_lon + 180) % 360) - 180  # 归一化到[-180, 180]
            
            position = UserPosition(
                latitude=user_lat,
                longitude=user_lon,
                altitude=0.0
            )
            
            user = User(
                user_id=i,
                position=position
            )
            users.append(user)
        
        return users
    
    def generate_users_in_rectangle(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        num_users: int
    ) -> List[User]:
        """
        在矩形区域内随机生成用户
        
        Args:
            lat_min, lat_max: 纬度范围
            lon_min, lon_max: 经度范围
            num_users: 用户数量
            
        Returns:
            用户列表
        """
        users = []
        
        for i in range(num_users):
            user_lat = self.rng.uniform(lat_min, lat_max)
            user_lon = self.rng.uniform(lon_min, lon_max)
            
            position = UserPosition(
                latitude=user_lat,
                longitude=user_lon,
                altitude=0.0
            )
            
            user = User(
                user_id=i,
                position=position
            )
            users.append(user)
        
        return users
    
    def generate_users_grid(
        self,
        center_lat: float,
        center_lon: float,
        spacing_deg: float,
        grid_size: int
    ) -> List[User]:
        """
        在网格上生成用户（用于规则分布测试）
        
        Args:
            center_lat: 中心纬度
            center_lon: 中心经度
            spacing_deg: 网格间距 (度)
            grid_size: 网格边长 (奇数，如5表示5x5)
            
        Returns:
            用户列表
        """
        users = []
        half_size = grid_size // 2
        user_id = 0
        
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                user_lat = center_lat + i * spacing_deg
                user_lon = center_lon + j * spacing_deg / np.cos(np.radians(center_lat))
                
                position = UserPosition(
                    latitude=user_lat,
                    longitude=user_lon,
                    altitude=0.0
                )
                
                user = User(
                    user_id=user_id,
                    position=position
                )
                users.append(user)
                user_id += 1
        
        return users


class UserManager:
    """
    用户管理器
    统一管理所有用户的状态更新和统计
    """
    
    def __init__(self, users: List[User]):
        """
        Args:
            users: 用户列表
        """
        self.users = users
        self.num_users = len(users)
        self.current_time = 0.0
    
    def get_user(self, user_id: int) -> User:
        """获取指定用户"""
        return self.users[user_id]
    
    def get_all_positions_ecef(self) -> np.ndarray:
        """
        获取所有用户的ECEF位置矩阵
        
        Returns:
            positions: shape (num_users, 3)
        """
        positions = np.zeros((self.num_users, 3))
        for i, user in enumerate(self.users):
            positions[i] = user.get_ecef_position()
        return positions
    
    def get_all_positions_lla(self) -> np.ndarray:
        """
        获取所有用户的经纬度位置矩阵
        
        Returns:
            positions: shape (num_users, 2) - [lat, lon]
        """
        positions = np.zeros((self.num_users, 2))
        for i, user in enumerate(self.users):
            positions[i, 0] = user.position.latitude
            positions[i, 1] = user.position.longitude
        return positions
    
    def get_connected_users(self) -> List[User]:
        """获取所有已连接用户"""
        return [u for u in self.users if u.state == UserState.CONNECTED]
    
    def get_idle_users(self) -> List[User]:
        """获取所有空闲用户"""
        return [u for u in self.users if u.state == UserState.IDLE]
    
    def get_blocked_users(self) -> List[User]:
        """获取所有阻塞用户"""
        return [u for u in self.users if u.state == UserState.BLOCKED]
    
    def update_all_statistics(self, current_time: float):
        """更新所有用户的统计信息"""
        for user in self.users:
            user.update_statistics(current_time)
        self.current_time = current_time
    
    def get_statistics_summary(self) -> Dict:
        """
        获取统计摘要
        
        Returns:
            包含各项统计指标的字典
        """
        total_handovers = sum(u.handover_count for u in self.users)
        successful_handovers = sum(u.successful_handovers for u in self.users)
        failed_handovers = sum(u.failed_handovers for u in self.users)
        total_service_time = sum(u.total_service_time for u in self.users)
        total_blocked_time = sum(u.total_blocked_time for u in self.users)
        
        connected_count = len(self.get_connected_users())
        blocked_count = len(self.get_blocked_users())
        
        return {
            'num_users': self.num_users,
            'connected_users': connected_count,
            'blocked_users': blocked_count,
            'total_handovers': total_handovers,
            'successful_handovers': successful_handovers,
            'failed_handovers': failed_handovers,
            'handover_success_rate': successful_handovers / max(total_handovers, 1),
            'total_service_time': total_service_time,
            'total_blocked_time': total_blocked_time,
            'service_ratio': total_service_time / max(total_service_time + total_blocked_time, 1)
        }
    
    def print_status(self):
        """打印当前状态"""
        stats = self.get_statistics_summary()
        print(f"\n用户状态统计 (t={self.current_time:.1f}s):")
        print(f"  已连接: {stats['connected_users']}/{stats['num_users']}")
        print(f"  阻塞中: {stats['blocked_users']}")
        print(f"  切换次数: {stats['total_handovers']} "
              f"(成功率: {stats['handover_success_rate']*100:.1f}%)")