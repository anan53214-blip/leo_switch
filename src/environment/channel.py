"""
卫星信道模型
实现星地链路的信道建模，包括：
1. 自由空间路径损耗 (FSPL)
2. 大气损耗
3. 信噪比 (SNR) 计算
4. 香农信道容量

参考论文：
- 宋晓勤等《基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法》
- 付一阳等《星地融合网络中基于异质图表征的多智能体协作切换方法》
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


# 物理常数
SPEED_OF_LIGHT = 3e8          # 光速 (m/s)
BOLTZMANN_CONSTANT = 1.38e-23 # 玻尔兹曼常数 (J/K)


@dataclass
class ChannelConfig:
    """
    信道配置参数
    参考论文中的典型参数设置及实际LEO系统（如Starlink）
    """
    # 频率参数
    carrier_frequency_ghz: float = 20.0      # 载波频率 (GHz)，Ka频段
    bandwidth_mhz: float = 10.0              # 信道带宽 (MHz)，增加带宽
    
    # 发射端参数 (卫星)
    satellite_tx_power_dbm: float = 40.0     # 卫星发射功率 (dBm)，10W
    satellite_antenna_gain_db: float = 34.0  # 卫星天线增益 (dB)，相控阵
    
    # 接收端参数 (用户终端)
    user_tx_power_dbm: float = 24.0          # 用户发射功率 (dBm)，2W卫星终端
    user_antenna_gain_db: float = 38.5       # 用户天线增益 (dB)，相控阵终端
    
    # 噪声参数
    noise_temperature_k: float = 354.81      # 等效噪声温度 (K)
    noise_figure_db: float = 2.0             # 噪声系数 (dB)
    
    # 大气损耗参数
    rain_attenuation_db: float = 0.0         # 雨衰 (dB)，晴天为0
    atmospheric_loss_db: float = 0.3         # 大气吸收损耗 (dB)，降低
    
    # 其他损耗
    pointing_loss_db: float = 0.5            # 指向损耗 (dB)，相控阵更精准
    polarization_loss_db: float = 0.3        # 极化损耗 (dB)
    implementation_loss_db: float = 1.0      # 实现损耗 (dB)


class SatelliteChannel:
    """
    卫星信道模型
    
    实现星地链路的完整信道模型，用于计算：
    - 上行链路（用户→卫星）：任务数据上传
    - 下行链路（卫星→用户）：计算结果返回
    
    信道容量公式（香农公式）：
    C = B * log2(1 + SNR)
    
    其中 SNR = P_tx * G_tx * G_rx / (L_total * N_0 * B)
    """
    
    def __init__(self, config: Optional[ChannelConfig] = None):
        """
        初始化信道模型
        
        Args:
            config: 信道配置参数，为None时使用默认值
        """
        self.config = config or ChannelConfig()
        
        # 预计算常用值
        self._precompute_constants()
    
    def _precompute_constants(self):
        """预计算常用常数"""
        # 转换单位
        self.carrier_freq_hz = self.config.carrier_frequency_ghz * 1e9
        self.bandwidth_hz = self.config.bandwidth_mhz * 1e6
        
        # 波长 (m)
        self.wavelength = SPEED_OF_LIGHT / self.carrier_freq_hz
        
        # 功率转换 (dBm -> W)
        self.sat_tx_power_w = self._dbm_to_watt(self.config.satellite_tx_power_dbm)
        self.user_tx_power_w = self._dbm_to_watt(self.config.user_tx_power_dbm)
        
        # 噪声功率谱密度 N0 (W/Hz)
        noise_figure_linear = self._db_to_linear(self.config.noise_figure_db)
        self.noise_psd = BOLTZMANN_CONSTANT * self.config.noise_temperature_k * noise_figure_linear
        
        # 噪声功率 (W)
        self.noise_power_w = self.noise_psd * self.bandwidth_hz
        
        # 天线增益 (线性值)
        self.sat_antenna_gain = self._db_to_linear(self.config.satellite_antenna_gain_db)
        self.user_antenna_gain = self._db_to_linear(self.config.user_antenna_gain_db)
        
        # 其他损耗 (线性值)
        self.atmospheric_loss = self._db_to_linear(self.config.atmospheric_loss_db)
        self.rain_loss = self._db_to_linear(self.config.rain_attenuation_db)
        self.pointing_loss = self._db_to_linear(self.config.pointing_loss_db)
        self.polarization_loss = self._db_to_linear(self.config.polarization_loss_db)
        self.implementation_loss = self._db_to_linear(self.config.implementation_loss_db)
        
        # 总附加损耗
        self.total_additional_loss = (
            self.atmospheric_loss * 
            self.rain_loss * 
            self.pointing_loss * 
            self.polarization_loss * 
            self.implementation_loss
        )
    
    @staticmethod
    def _db_to_linear(db_value: float) -> float:
        """dB转线性值"""
        return 10 ** (db_value / 10)
    
    @staticmethod
    def _linear_to_db(linear_value: float) -> float:
        """线性值转dB"""
        return 10 * np.log10(max(linear_value, 1e-20))
    
    @staticmethod
    def _dbm_to_watt(dbm_value: float) -> float:
        """dBm转瓦特"""
        return 10 ** ((dbm_value - 30) / 10)
    
    @staticmethod
    def _watt_to_dbm(watt_value: float) -> float:
        """瓦特转dBm"""
        return 10 * np.log10(max(watt_value, 1e-20)) + 30
    
    def compute_free_space_path_loss(self, distance_km: float) -> float:
        """
        计算自由空间路径损耗 (FSPL)
        
        FSPL = (4πd/λ)² = (4πdf/c)²
        
        FSPL_dB = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
                = 20*log10(d_km) + 20*log10(f_GHz) + 92.45
        
        Args:
            distance_km: 星地距离 (km)
            
        Returns:
            路径损耗 (线性值，非dB)
        """
        if distance_km <= 0:
            return 1.0
        
        # 距离转换为米
        distance_m = distance_km * 1000
        
        # FSPL = (4πd/λ)²
        fspl_linear = (4 * np.pi * distance_m / self.wavelength) ** 2
        
        return fspl_linear
    
    def compute_free_space_path_loss_db(self, distance_km: float) -> float:
        """
        计算自由空间路径损耗 (dB)
        
        Args:
            distance_km: 星地距离 (km)
            
        Returns:
            路径损耗 (dB)
        """
        if distance_km <= 0:
            return 0.0
        
        # FSPL_dB = 20*log10(d_km) + 20*log10(f_GHz) + 92.45
        fspl_db = (
            20 * np.log10(distance_km) + 
            20 * np.log10(self.config.carrier_frequency_ghz) + 
            92.45
        )
        
        return fspl_db
    
    def compute_total_path_loss(self, distance_km: float, elevation_deg: float) -> float:
        """
        计算总路径损耗
        
        包括：FSPL + 大气损耗 + 雨衰 + 其他损耗
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            
        Returns:
            总路径损耗 (线性值)
        """
        # 自由空间路径损耗
        fspl = self.compute_free_space_path_loss(distance_km)
        
        # 仰角相关的大气损耗修正
        # 低仰角时大气路径更长，损耗更大
        elevation_factor = self._compute_elevation_factor(elevation_deg)
        
        # 总损耗
        total_loss = fspl * self.total_additional_loss * elevation_factor
        
        return total_loss
    
    def _compute_elevation_factor(self, elevation_deg: float) -> float:
        """
        计算仰角相关的损耗因子
        
        低仰角时信号穿过大气层路径更长
        
        Args:
            elevation_deg: 仰角 (度)
            
        Returns:
            仰角损耗因子 (线性值)
        """
        if elevation_deg <= 0:
            return 100.0  # 地平线以下，极大损耗
        
        if elevation_deg >= 90:
            return 1.0
        
        # 大气路径长度与sin(elevation)成反比
        # 使用简化模型：factor = 1/sin(elev) 的修正
        elev_rad = np.radians(elevation_deg)
        
        # 限制最大因子（避免低仰角时损耗过大）
        sin_elev = max(np.sin(elev_rad), 0.1)
        
        # 附加损耗因子 (仅对大气损耗部分有效)
        atmospheric_path_factor = 1.0 / sin_elev
        
        # 限制在合理范围内
        return min(atmospheric_path_factor, 10.0)
    
    def compute_snr_uplink(
        self,
        distance_km: float,
        elevation_deg: float,
        user_tx_power_dbm: Optional[float] = None
    ) -> float:
        """
        计算上行链路SNR（用户→卫星）
        
        SNR = P_tx * G_tx * G_rx / (L_path * N_0 * B)
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            user_tx_power_dbm: 用户发射功率 (dBm)，None时使用默认值
            
        Returns:
            SNR (线性值)
        """
        # 用户发射功率
        if user_tx_power_dbm is not None:
            tx_power_w = self._dbm_to_watt(user_tx_power_dbm)
        else:
            tx_power_w = self.user_tx_power_w
        
        # 路径损耗
        path_loss = self.compute_total_path_loss(distance_km, elevation_deg)
        
        # 接收功率
        # P_rx = P_tx * G_tx * G_rx / L
        received_power = (
            tx_power_w * 
            self.user_antenna_gain * 
            self.sat_antenna_gain / 
            path_loss
        )
        
        # SNR = P_rx / N
        snr = received_power / self.noise_power_w
        
        return max(snr, 1e-10)  # 避免0值
    
    def compute_snr_downlink(
        self,
        distance_km: float,
        elevation_deg: float,
        sat_tx_power_dbm: Optional[float] = None
    ) -> float:
        """
        计算下行链路SNR（卫星→用户）
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            sat_tx_power_dbm: 卫星发射功率 (dBm)，None时使用默认值
            
        Returns:
            SNR (线性值)
        """
        # 卫星发射功率
        if sat_tx_power_dbm is not None:
            tx_power_w = self._dbm_to_watt(sat_tx_power_dbm)
        else:
            tx_power_w = self.sat_tx_power_w
        
        # 路径损耗
        path_loss = self.compute_total_path_loss(distance_km, elevation_deg)
        
        # 接收功率
        received_power = (
            tx_power_w * 
            self.sat_antenna_gain * 
            self.user_antenna_gain / 
            path_loss
        )
        
        # SNR
        snr = received_power / self.noise_power_w
        
        return max(snr, 1e-10)
    
    def compute_snr_db(
        self,
        distance_km: float,
        elevation_deg: float,
        link_type: str = 'uplink'
    ) -> float:
        """
        计算SNR (dB)
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            link_type: 'uplink' 或 'downlink'
            
        Returns:
            SNR (dB)
        """
        if link_type == 'uplink':
            snr_linear = self.compute_snr_uplink(distance_km, elevation_deg)
        else:
            snr_linear = self.compute_snr_downlink(distance_km, elevation_deg)
        
        return self._linear_to_db(snr_linear)
    
    def compute_channel_capacity(
        self,
        distance_km: float,
        elevation_deg: float,
        link_type: str = 'uplink',
        bandwidth_mhz: Optional[float] = None
    ) -> float:
        """
        计算香农信道容量
        
        C = B * log2(1 + SNR)
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            link_type: 'uplink' 或 'downlink'
            bandwidth_mhz: 带宽 (MHz)，None时使用默认值
            
        Returns:
            信道容量 (bps)
        """
        # SNR
        if link_type == 'uplink':
            snr = self.compute_snr_uplink(distance_km, elevation_deg)
        else:
            snr = self.compute_snr_downlink(distance_km, elevation_deg)
        
        # 带宽
        if bandwidth_mhz is not None:
            bandwidth_hz = bandwidth_mhz * 1e6
        else:
            bandwidth_hz = self.bandwidth_hz
        
        # 香农容量
        capacity_bps = bandwidth_hz * np.log2(1 + snr)
        
        return capacity_bps
    
    def compute_data_rate_mbps(
        self,
        distance_km: float,
        elevation_deg: float,
        link_type: str = 'uplink',
        bandwidth_mhz: Optional[float] = None
    ) -> float:
        """
        计算数据传输速率 (Mbps)
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            link_type: 'uplink' 或 'downlink'
            bandwidth_mhz: 带宽 (MHz)
            
        Returns:
            数据速率 (Mbps)
        """
        capacity_bps = self.compute_channel_capacity(
            distance_km, elevation_deg, link_type, bandwidth_mhz
        )
        return capacity_bps / 1e6
    
    def compute_transmission_delay(
        self,
        data_size_bits: float,
        distance_km: float,
        elevation_deg: float,
        link_type: str = 'uplink'
    ) -> float:
        """
        计算传输时延
        
        T_trans = D / R + T_prop
        
        其中：
        - D: 数据量 (bits)
        - R: 传输速率 (bps)
        - T_prop: 传播时延 (s)
        
        Args:
            data_size_bits: 数据量 (bits)
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            link_type: 链路类型
            
        Returns:
            传输时延 (秒)
        """
        # 传输速率
        data_rate_bps = self.compute_channel_capacity(
            distance_km, elevation_deg, link_type
        )
        
        # 传输时延
        if data_rate_bps > 0:
            transmission_delay = data_size_bits / data_rate_bps
        else:
            transmission_delay = float('inf')
        
        # 传播时延
        propagation_delay = self.compute_propagation_delay(distance_km)
        
        return transmission_delay + propagation_delay
    
    def compute_propagation_delay(self, distance_km: float) -> float:
        """
        计算传播时延
        
        T_prop = d / c
        
        Args:
            distance_km: 星地距离 (km)
            
        Returns:
            传播时延 (秒)
        """
        distance_m = distance_km * 1000
        return distance_m / SPEED_OF_LIGHT
    
    def compute_propagation_delay_ms(self, distance_km: float) -> float:
        """计算传播时延 (毫秒)"""
        return self.compute_propagation_delay(distance_km) * 1000
    
    def get_link_budget(
        self,
        distance_km: float,
        elevation_deg: float,
        link_type: str = 'uplink'
    ) -> Dict[str, float]:
        """
        获取完整的链路预算
        
        Args:
            distance_km: 星地距离 (km)
            elevation_deg: 仰角 (度)
            link_type: 链路类型
            
        Returns:
            包含所有链路参数的字典
        """
        # 根据链路类型选择参数
        if link_type == 'uplink':
            tx_power_dbm = self.config.user_tx_power_dbm
            tx_gain_db = self.config.user_antenna_gain_db
            rx_gain_db = self.config.satellite_antenna_gain_db
            snr = self.compute_snr_uplink(distance_km, elevation_deg)
        else:
            tx_power_dbm = self.config.satellite_tx_power_dbm
            tx_gain_db = self.config.satellite_antenna_gain_db
            rx_gain_db = self.config.user_antenna_gain_db
            snr = self.compute_snr_downlink(distance_km, elevation_deg)
        
        # 计算各项
        fspl_db = self.compute_free_space_path_loss_db(distance_km)
        total_loss_db = self._linear_to_db(
            self.compute_total_path_loss(distance_km, elevation_deg)
        )
        
        return {
            'link_type': link_type,
            'distance_km': distance_km,
            'elevation_deg': elevation_deg,
            'tx_power_dbm': tx_power_dbm,
            'tx_antenna_gain_db': tx_gain_db,
            'rx_antenna_gain_db': rx_gain_db,
            'fspl_db': fspl_db,
            'atmospheric_loss_db': self.config.atmospheric_loss_db,
            'rain_attenuation_db': self.config.rain_attenuation_db,
            'total_path_loss_db': total_loss_db,
            'noise_power_dbm': self._watt_to_dbm(self.noise_power_w),
            'snr_db': self._linear_to_db(snr),
            'snr_linear': snr,
            'channel_capacity_mbps': self.compute_data_rate_mbps(
                distance_km, elevation_deg, link_type
            ),
            'propagation_delay_ms': self.compute_propagation_delay_ms(distance_km),
        }
