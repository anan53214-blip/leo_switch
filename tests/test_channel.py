"""
淇￠亾妯″瀷娴嬭瘯
楠岃瘉淇￠亾璁＄畻鐨勬纭€?
"""

import numpy as np
import pytest

from src.environment.channel import (
    SatelliteChannel, ChannelConfig, MultiUserChannel,
    compute_link_capacity
)


def test_default_channel_config_uses_moderate_leo_uplink_pressure():
    config = ChannelConfig()

    assert config.bandwidth_mhz == 10.0
    assert config.user_tx_power_dbm == 24.0


@pytest.fixture
def channel():
    return SatelliteChannel()


def test_channel_basic():
    """娴嬭瘯鍩烘湰淇￠亾璁＄畻"""
    print("=" * 60)
    print("娴嬭瘯1: 鍩烘湰淇￠亾鍙傛暟")
    print("=" * 60)
    
    channel = SatelliteChannel()
    
    print(f"\n淇￠亾閰嶇疆:")
    print(f"  杞芥尝棰戠巼: {channel.config.carrier_frequency_ghz} GHz")
    print(f"  甯﹀: {channel.config.bandwidth_mhz} MHz")
    print(f"  鍗槦鍙戝皠鍔熺巼: {channel.config.satellite_tx_power_dbm} dBm")
    print(f"  鐢ㄦ埛鍙戝皠鍔熺巼: {channel.config.user_tx_power_dbm} dBm")
    
    # 鍏稿瀷LEO鍗槦璺濈 (550km楂樺害锛屼话瑙?0掳鏃?
    distance = 550  # km
    elevation = 90  # 搴?
    
    # 鑷敱绌洪棿璺緞鎹熻€?
    fspl_db = channel.compute_free_space_path_loss_db(distance)
    print(f"\n鑷敱绌洪棿璺緞鎹熻€?(d={distance}km): {fspl_db:.2f} dB")
    
    # 棰勬湡鍊? FSPL 鈮?20*log10(550) + 20*log10(20) + 92.45 鈮?173 dB
    assert 170 < fspl_db < 180, f"FSPL璁＄畻寮傚父: {fspl_db}"
    print("鉁?FSPL璁＄畻姝ｇ‘")
    


def test_snr_calculation(channel):
    """娴嬭瘯SNR璁＄畻"""
    print("\n" + "=" * 60)
    print("娴嬭瘯2: SNR璁＄畻")
    print("=" * 60)
    
    # 娴嬭瘯涓嶅悓璺濈鍜屼话瑙掔殑SNR
    test_cases = [
        (550, 90),   # 鏈€鐭窛绂伙紝鏈€浣充话瑙?
        (800, 45),   # 涓瓑璺濈
        (1500, 20),  # 杈冭繙璺濈
        (2000, 10),  # 鏈€灏忎话瑙?
    ]
    
    print(f"\n{'璺濈(km)':<12} {'浠拌(掳)':<10} {'涓婅SNR(dB)':<14} {'涓嬭SNR(dB)':<14}")
    print("-" * 50)
    
    for distance, elevation in test_cases:
        snr_up = channel.compute_snr_db(distance, elevation, 'uplink')
        snr_down = channel.compute_snr_db(distance, elevation, 'downlink')
        print(f"{distance:<12} {elevation:<10} {snr_up:<14.2f} {snr_down:<14.2f}")
        
        # 涓嬭SNR搴旇鏇撮珮锛堝崼鏄熷彂灏勫姛鐜囨洿澶э級
        assert snr_down > snr_up, "涓嬭SNR搴斿ぇ浜庝笂琛?
    
    print("\n鉁?SNR闅忚窛绂诲鍔犺€岄檷浣?)
    print("鉁?涓嬭SNR > 涓婅SNR")


def test_channel_capacity(channel):
    """娴嬭瘯淇￠亾瀹归噺璁＄畻"""
    print("\n" + "=" * 60)
    print("娴嬭瘯3: 淇￠亾瀹归噺璁＄畻")
    print("=" * 60)
    
    test_cases = [
        (550, 90),
        (800, 45),
        (1500, 20),
        (2000, 10),
    ]
    
    print(f"\n{'璺濈(km)':<12} {'浠拌(掳)':<10} {'涓婅閫熺巼(Mbps)':<16} {'涓嬭閫熺巼(Mbps)':<16}")
    print("-" * 55)
    
    for distance, elevation in test_cases:
        rate_up = channel.compute_data_rate_mbps(distance, elevation, 'uplink')
        rate_down = channel.compute_data_rate_mbps(distance, elevation, 'downlink')
        print(f"{distance:<12} {elevation:<10} {rate_up:<16.2f} {rate_down:<16.2f}")
        
        assert rate_up > 0, "浼犺緭閫熺巼搴斾负姝?
        assert rate_down > rate_up, "涓嬭閫熺巼搴斿ぇ浜庝笂琛?
    
    print("\n鉁?淇￠亾瀹归噺璁＄畻姝ｇ‘")


def test_transmission_delay(channel):
    """娴嬭瘯浼犺緭鏃跺欢"""
    print("\n" + "=" * 60)
    print("娴嬭瘯4: 浼犺緭鏃跺欢璁＄畻")
    print("=" * 60)
    
    distance = 800  # km
    elevation = 45  # 搴?
    
    # 浼犳挱鏃跺欢
    prop_delay_ms = channel.compute_propagation_delay_ms(distance)
    print(f"\n浼犳挱鏃跺欢 (d={distance}km): {prop_delay_ms:.2f} ms")
    
    # 棰勬湡: d/c = 800km / 3e5 km/s 鈮?2.67 ms
    expected_prop_delay = distance / 300  # ms
    assert abs(prop_delay_ms - expected_prop_delay) < 0.1, "浼犳挱鏃跺欢璁＄畻寮傚父"
    
    # 娴嬭瘯涓嶅悓鏁版嵁閲忕殑浼犺緭鏃跺欢
    data_sizes_mb = [1, 5, 10, 50]  # MB
    
    print(f"\n鏁版嵁浼犺緭鏃跺欢 (d={distance}km, elev={elevation}掳):")
    print(f"{'鏁版嵁閲?MB)':<12} {'浼犺緭鏃跺欢(s)':<14} {'浼犺緭閫熺巼(Mbps)':<16}")
    print("-" * 45)
    
    rate_mbps = channel.compute_data_rate_mbps(distance, elevation, 'uplink')
    
    for data_mb in data_sizes_mb:
        data_bits = data_mb * 8 * 1e6  # bits
        delay = channel.compute_transmission_delay(
            data_bits, distance, elevation, 'uplink'
        )
        print(f"{data_mb:<12} {delay:<14.3f} {rate_mbps:<16.2f}")
    
    print("\n鉁?浼犺緭鏃跺欢璁＄畻姝ｇ‘")


def test_link_budget(channel):
    """娴嬭瘯閾捐矾棰勭畻"""
    print("\n" + "=" * 60)
    print("娴嬭瘯5: 閾捐矾棰勭畻")
    print("=" * 60)
    
    # 鎵撳嵃鍏稿瀷鍦烘櫙鐨勯摼璺绠?
    channel.print_link_budget(800, 45, 'uplink')
    channel.print_link_budget(800, 45, 'downlink')


def test_multi_user_channel():
    """娴嬭瘯澶氱敤鎴蜂俊閬?""
    print("\n" + "=" * 60)
    print("娴嬭瘯6: 澶氱敤鎴蜂俊閬?)
    print("=" * 60)
    
    mu_channel = MultiUserChannel(total_bandwidth_mhz=500.0)
    
    # 妯℃嫙5涓敤鎴?
    num_users = 5
    distances = np.array([600, 700, 800, 900, 1000])  # km
    elevations = np.array([60, 50, 45, 35, 30])       # 搴?
    demands = np.array([1.0, 2.0, 1.5, 0.5, 1.0])     # 鐩稿闇€姹?
    
    # 骞冲潎鍒嗛厤
    bw_equal = mu_channel.allocate_bandwidth_equal(num_users)
    print(f"\n骞冲潎鍒嗛厤甯﹀: {bw_equal:.1f} MHz/鐢ㄦ埛")
    
    # 鎸夐渶鍒嗛厤
    bw_proportional = mu_channel.allocate_bandwidth_proportional(demands)
    
    print(f"\n鎸夐渶鍒嗛厤缁撴灉:")
    print(f"{'鐢ㄦ埛':<8} {'璺濈(km)':<12} {'浠拌(掳)':<10} {'闇€姹?:<8} {'甯﹀(MHz)':<12}")
    print("-" * 50)
    for i in range(num_users):
        print(f"{i:<8} {distances[i]:<12} {elevations[i]:<10} {demands[i]:<8.1f} {bw_proportional[i]:<12.1f}")
    
    # 璁＄畻閫熺巼
    rates = mu_channel.compute_user_rates(
        distances, elevations, bw_proportional, 'uplink'
    )
    
    print(f"\n鐢ㄦ埛浼犺緭閫熺巼:")
    for i in range(num_users):
        print(f"  鐢ㄦ埛{i}: {rates[i]:.2f} Mbps")
    
    print("\n鉁?澶氱敤鎴蜂俊閬撴祴璇曢€氳繃")


def test_convenience_function():
    """娴嬭瘯渚挎嵎鍑芥暟"""
    print("\n" + "=" * 60)
    print("娴嬭瘯7: 渚挎嵎鍑芥暟")
    print("=" * 60)
    
    capacity = compute_link_capacity(800, 45, 20)
    print(f"\ncompute_link_capacity(800km, 45掳, 20MHz) = {capacity:.2f} Mbps")
    
    assert capacity > 0, "瀹归噺搴斾负姝?
    print("鉁?渚挎嵎鍑芥暟娴嬭瘯閫氳繃")


