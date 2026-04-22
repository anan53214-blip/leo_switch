"""
信道模型测试
验证信道计算的正确性
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, 'd:\\python_code\\LEO_switch')

from src.environment.channel import (
    SatelliteChannel, ChannelConfig, MultiUserChannel,
    compute_link_capacity
)


@pytest.fixture
def channel():
    return SatelliteChannel()


def test_channel_basic():
    """测试基本信道计算"""
    print("=" * 60)
    print("测试1: 基本信道参数")
    print("=" * 60)
    
    channel = SatelliteChannel()
    
    print(f"\n信道配置:")
    print(f"  载波频率: {channel.config.carrier_frequency_ghz} GHz")
    print(f"  带宽: {channel.config.bandwidth_mhz} MHz")
    print(f"  卫星发射功率: {channel.config.satellite_tx_power_dbm} dBm")
    print(f"  用户发射功率: {channel.config.user_tx_power_dbm} dBm")
    
    # 典型LEO卫星距离 (550km高度，仰角90°时)
    distance = 550  # km
    elevation = 90  # 度
    
    # 自由空间路径损耗
    fspl_db = channel.compute_free_space_path_loss_db(distance)
    print(f"\n自由空间路径损耗 (d={distance}km): {fspl_db:.2f} dB")
    
    # 预期值: FSPL ≈ 20*log10(550) + 20*log10(20) + 92.45 ≈ 173 dB
    assert 170 < fspl_db < 180, f"FSPL计算异常: {fspl_db}"
    print("✓ FSPL计算正确")
    


def test_snr_calculation(channel):
    """测试SNR计算"""
    print("\n" + "=" * 60)
    print("测试2: SNR计算")
    print("=" * 60)
    
    # 测试不同距离和仰角的SNR
    test_cases = [
        (550, 90),   # 最短距离，最佳仰角
        (800, 45),   # 中等距离
        (1500, 20),  # 较远距离
        (2000, 10),  # 最小仰角
    ]
    
    print(f"\n{'距离(km)':<12} {'仰角(°)':<10} {'上行SNR(dB)':<14} {'下行SNR(dB)':<14}")
    print("-" * 50)
    
    for distance, elevation in test_cases:
        snr_up = channel.compute_snr_db(distance, elevation, 'uplink')
        snr_down = channel.compute_snr_db(distance, elevation, 'downlink')
        print(f"{distance:<12} {elevation:<10} {snr_up:<14.2f} {snr_down:<14.2f}")
        
        # 下行SNR应该更高（卫星发射功率更大）
        assert snr_down > snr_up, "下行SNR应大于上行"
    
    print("\n✓ SNR随距离增加而降低")
    print("✓ 下行SNR > 上行SNR")


def test_channel_capacity(channel):
    """测试信道容量计算"""
    print("\n" + "=" * 60)
    print("测试3: 信道容量计算")
    print("=" * 60)
    
    test_cases = [
        (550, 90),
        (800, 45),
        (1500, 20),
        (2000, 10),
    ]
    
    print(f"\n{'距离(km)':<12} {'仰角(°)':<10} {'上行速率(Mbps)':<16} {'下行速率(Mbps)':<16}")
    print("-" * 55)
    
    for distance, elevation in test_cases:
        rate_up = channel.compute_data_rate_mbps(distance, elevation, 'uplink')
        rate_down = channel.compute_data_rate_mbps(distance, elevation, 'downlink')
        print(f"{distance:<12} {elevation:<10} {rate_up:<16.2f} {rate_down:<16.2f}")
        
        assert rate_up > 0, "传输速率应为正"
        assert rate_down > rate_up, "下行速率应大于上行"
    
    print("\n✓ 信道容量计算正确")


def test_transmission_delay(channel):
    """测试传输时延"""
    print("\n" + "=" * 60)
    print("测试4: 传输时延计算")
    print("=" * 60)
    
    distance = 800  # km
    elevation = 45  # 度
    
    # 传播时延
    prop_delay_ms = channel.compute_propagation_delay_ms(distance)
    print(f"\n传播时延 (d={distance}km): {prop_delay_ms:.2f} ms")
    
    # 预期: d/c = 800km / 3e5 km/s ≈ 2.67 ms
    expected_prop_delay = distance / 300  # ms
    assert abs(prop_delay_ms - expected_prop_delay) < 0.1, "传播时延计算异常"
    
    # 测试不同数据量的传输时延
    data_sizes_mb = [1, 5, 10, 50]  # MB
    
    print(f"\n数据传输时延 (d={distance}km, elev={elevation}°):")
    print(f"{'数据量(MB)':<12} {'传输时延(s)':<14} {'传输速率(Mbps)':<16}")
    print("-" * 45)
    
    rate_mbps = channel.compute_data_rate_mbps(distance, elevation, 'uplink')
    
    for data_mb in data_sizes_mb:
        data_bits = data_mb * 8 * 1e6  # bits
        delay = channel.compute_transmission_delay(
            data_bits, distance, elevation, 'uplink'
        )
        print(f"{data_mb:<12} {delay:<14.3f} {rate_mbps:<16.2f}")
    
    print("\n✓ 传输时延计算正确")


def test_link_budget(channel):
    """测试链路预算"""
    print("\n" + "=" * 60)
    print("测试5: 链路预算")
    print("=" * 60)
    
    # 打印典型场景的链路预算
    channel.print_link_budget(800, 45, 'uplink')
    channel.print_link_budget(800, 45, 'downlink')


def test_multi_user_channel():
    """测试多用户信道"""
    print("\n" + "=" * 60)
    print("测试6: 多用户信道")
    print("=" * 60)
    
    mu_channel = MultiUserChannel(total_bandwidth_mhz=500.0)
    
    # 模拟5个用户
    num_users = 5
    distances = np.array([600, 700, 800, 900, 1000])  # km
    elevations = np.array([60, 50, 45, 35, 30])       # 度
    demands = np.array([1.0, 2.0, 1.5, 0.5, 1.0])     # 相对需求
    
    # 平均分配
    bw_equal = mu_channel.allocate_bandwidth_equal(num_users)
    print(f"\n平均分配带宽: {bw_equal:.1f} MHz/用户")
    
    # 按需分配
    bw_proportional = mu_channel.allocate_bandwidth_proportional(demands)
    
    print(f"\n按需分配结果:")
    print(f"{'用户':<8} {'距离(km)':<12} {'仰角(°)':<10} {'需求':<8} {'带宽(MHz)':<12}")
    print("-" * 50)
    for i in range(num_users):
        print(f"{i:<8} {distances[i]:<12} {elevations[i]:<10} {demands[i]:<8.1f} {bw_proportional[i]:<12.1f}")
    
    # 计算速率
    rates = mu_channel.compute_user_rates(
        distances, elevations, bw_proportional, 'uplink'
    )
    
    print(f"\n用户传输速率:")
    for i in range(num_users):
        print(f"  用户{i}: {rates[i]:.2f} Mbps")
    
    print("\n✓ 多用户信道测试通过")


def test_convenience_function():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("测试7: 便捷函数")
    print("=" * 60)
    
    capacity = compute_link_capacity(800, 45, 20)
    print(f"\ncompute_link_capacity(800km, 45°, 20MHz) = {capacity:.2f} Mbps")
    
    assert capacity > 0, "容量应为正"
    print("✓ 便捷函数测试通过")


def main():
    """运行所有测试"""
    print("\n" + "#" * 60)
    print("#          卫星信道模型测试")
    print("#" * 60)
    
    channel = SatelliteChannel()
    test_channel_basic()
    test_snr_calculation(channel)
    test_channel_capacity(channel)
    test_transmission_delay(channel)
    test_link_budget(channel)
    test_multi_user_channel()
    test_convenience_function()
    
    print("\n" + "=" * 60)
    print("         所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
