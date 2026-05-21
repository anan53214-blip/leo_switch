"""
MEC妯″瀷娴嬭瘯
楠岃瘉浠诲姟鍗歌浇璁＄畻鐨勬纭€?
"""

import numpy as np
import pytest

from src.environment.mec import (
    MECConfig, MECServer, MECManager,
    OffloadingCalculator, ComputeResult,
    compute_task_delay, compute_task_energy
)


def test_default_mec_config_keeps_20ghz_satellite_capacity_with_tighter_queue():
    config = MECConfig()

    assert config.satellite_cpu_freq_ghz == 5.0
    assert config.satellite_num_cores == 4
    assert config.satellite_cpu_freq_ghz * config.satellite_num_cores == 20.0
    assert config.max_queue_size == 6
    assert config.user_cpu_freq_ghz == 1.0
    assert config.user_max_cpu_freq_ghz == 1.5
    assert config.user_idle_power_w == 0.05


@pytest.fixture
def calc():
    return OffloadingCalculator()


def test_mec_config():
    """娴嬭瘯MEC閰嶇疆"""
    print("=" * 60)
    print("娴嬭瘯1: MEC閰嶇疆鍙傛暟")
    print("=" * 60)
    
    config = MECConfig()
    
    print(f"\n鍗槦MEC鍙傛暟:")
    print(f"  CPU棰戠巼: {config.satellite_cpu_freq_ghz} GHz")
    print(f"  鏈€澶PU棰戠巼: {config.satellite_max_cpu_freq_ghz} GHz")
    print(f"  鏍稿績鏁? {config.satellite_num_cores}")
    print(f"  鏈€澶ч槦鍒? {config.max_queue_size}")
    
    print(f"\n鐢ㄦ埛璁惧鍙傛暟:")
    print(f"  CPU棰戠巼: {config.user_cpu_freq_ghz} GHz")
    print(f"  鑳借€楃郴鏁拔? {config.kappa}")
    
    print("\n鉁?閰嶇疆鍙傛暟鍔犺浇姝ｇ‘")


def test_mec_server():
    """娴嬭瘯MEC鏈嶅姟鍣?""
    print("\n" + "=" * 60)
    print("娴嬭瘯2: MEC鏈嶅姟鍣?)
    print("=" * 60)
    
    server = MECServer(satellite_id=0)
    
    print(f"\n鍒濆鐘舵€?")
    print(f"  鍗槦ID: {server.satellite_id}")
    print(f"  CPU棰戠巼: {server.cpu_freq_ghz} GHz")
    print(f"  鍙敤棰戠巼: {server.available_freq_ghz} GHz")
    print(f"  鍒╃敤鐜? {server.utilization:.2%}")
    print(f"  闃熷垪闀垮害: {server.queue_length}")
    
    # 娣诲姞鐢ㄦ埛
    server.add_user(0)
    server.add_user(1)
    server.add_user(2)
    print(f"\n娣诲姞3涓敤鎴峰悗:")
    print(f"  宸茶繛鎺ョ敤鎴? {server.connected_users}")
    
    # 鍒嗛厤璧勬簮
    allocated = server.allocate_compute_resource(0, 3.0)
    print(f"\n涓虹敤鎴?鍒嗛厤3GHz: 瀹為檯鍒嗛厤 {allocated} GHz")
    print(f"  鍓╀綑鍙敤: {server.available_freq_ghz} GHz")
    print(f"  鍒╃敤鐜? {server.utilization:.2%}")
    
    # 璁＄畻鏃跺欢
    cycles = 1e9  # 1G cycles
    delay = server.compute_processing_delay(cycles, allocated)
    print(f"\n澶勭悊1G cycles (鍒嗛厤{allocated}GHz):")
    print(f"  澶勭悊鏃跺欢: {delay*1000:.2f} ms")
    
    # 鐘舵€佸悜閲?
    state = server.get_state_vector()
    print(f"\n鐘舵€佸悜閲? {state}")
    
    print("\n鉁?MEC鏈嶅姟鍣ㄦ祴璇曢€氳繃")


def test_offloading_calculator():
    """娴嬭瘯鍗歌浇璁＄畻鍣?""
    print("\n" + "=" * 60)
    print("娴嬭瘯3: 鍗歌浇璁＄畻鍣?)
    print("=" * 60)
    
    calc = OffloadingCalculator()
    
    # 娴嬭瘯鍙傛暟
    data_bits = 5 * 8 * 1e6     # 5 MB
    compute_cycles = 2 * 1e9    # 2 G cycles
    max_delay = 2.0             # 2绉?
    distance = 800              # km
    elevation = 45              # 搴?
    
    print(f"\n浠诲姟鍙傛暟:")
    print(f"  鏁版嵁閲? 5 MB")
    print(f"  璁＄畻閲? 2 G cycles")
    print(f"  鏈€澶ф椂寤? {max_delay} s")
    print(f"  鍗槦璺濈: {distance} km")
    print(f"  浠拌: {elevation}掳")
    
    # 娴嬭瘯涓嶅悓鍗歌浇姣斾緥
    print(f"\n涓嶅悓鍗歌浇姣斾緥鐨勭粨鏋?")
    print("-" * 80)
    print(f"{'位':<6} {'鏈湴鏃跺欢(ms)':<14} {'涓婁紶(ms)':<12} {'鍗槦璁＄畻(ms)':<14} "
          f"{'鎬绘椂寤?ms)':<12} {'鑳借€?mJ)':<10} {'婊¤冻绾︽潫':<10}")
    print("-" * 80)
    
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = calc.compute_offloading_result(
            data_size_bits=data_bits,
            computation_cycles=compute_cycles,
            max_delay=max_delay,
            offload_ratio=ratio,
            distance_km=distance,
            elevation_deg=elevation
        )
        
        print(f"{ratio:<6.2f} {result.local_compute_delay*1000:<14.2f} "
              f"{result.upload_delay*1000:<12.2f} {result.satellite_compute_delay*1000:<14.2f} "
              f"{result.total_delay*1000:<12.2f} {result.total_energy*1000:<10.4f} "
              f"{'鉁? if result.deadline_met else '鉁?:<10}")
    
    print("-" * 80)
    print("\n鉁?鍗歌浇璁＄畻娴嬭瘯閫氳繃")
    


def test_optimal_offloading(calc):
    """娴嬭瘯鏈€浼樺嵏杞芥瘮渚嬫悳绱?""
    print("\n" + "=" * 60)
    print("娴嬭瘯4: 鏈€浼樺嵏杞芥瘮渚嬫悳绱?)
    print("=" * 60)
    
    # 浠诲姟鍙傛暟
    data_bits = 10 * 8 * 1e6    # 10 MB
    compute_cycles = 5 * 1e9    # 5 G cycles
    max_delay = 3.0             # 3绉?
    distance = 800
    elevation = 45
    
    print(f"\n浠诲姟: 10MB鏁版嵁, 5G cycles璁＄畻, 鏈€澶ф椂寤?s")
    
    # 鏈€灏忓寲鏃跺欢
    ratio_delay, result_delay = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='delay'
    )
    print(f"\n鏈€灏忓寲鏃跺欢:")
    print(f"  鏈€浼樺嵏杞芥瘮渚? {ratio_delay:.2f}")
    print(f"  鎬绘椂寤? {result_delay.total_delay*1000:.2f} ms")
    print(f"  鎬昏兘鑰? {result_delay.total_energy*1000:.4f} mJ")
    
    # 鏈€灏忓寲鑳借€?
    ratio_energy, result_energy = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='energy'
    )
    print(f"\n鏈€灏忓寲鑳借€?")
    print(f"  鏈€浼樺嵏杞芥瘮渚? {ratio_energy:.2f}")
    print(f"  鎬绘椂寤? {result_energy.total_delay*1000:.2f} ms")
    print(f"  鎬昏兘鑰? {result_energy.total_energy*1000:.4f} mJ")
    
    # 鍔犳潈浼樺寲
    ratio_weighted, result_weighted = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='weighted'
    )
    print(f"\n鍔犳潈浼樺寲 (鏃跺欢+鑳借€?:")
    print(f"  鏈€浼樺嵏杞芥瘮渚? {ratio_weighted:.2f}")
    print(f"  鎬绘椂寤? {result_weighted.total_delay*1000:.2f} ms")
    print(f"  鎬昏兘鑰? {result_weighted.total_energy*1000:.4f} mJ")
    
    print("\n鉁?鏈€浼樺嵏杞芥悳绱㈡祴璇曢€氳繃")


def test_mec_manager():
    """娴嬭瘯MEC绠＄悊鍣?""
    print("\n" + "=" * 60)
    print("娴嬭瘯5: MEC绠＄悊鍣?)
    print("=" * 60)
    
    manager = MECManager(num_satellites=66)
    
    print(f"\n鍒濆鍖?{manager.num_satellites} 棰楀崼鏄熺殑MEC鏈嶅姟鍣?)
    
    # 妯℃嫙涓€浜涘崼鏄熺殑璐熻浇
    for sat_id in [0, 1, 2, 5, 10]:
        server = manager.get_server(sat_id)
        server.allocate_compute_resource(0, 3.0)  # 鍒嗛厤涓€浜涜祫婧?
    
    # 鑾峰彇缁熻淇℃伅
    utils = manager.get_all_utilizations()
    print(f"\n鍓?0棰楀崼鏄熷埄鐢ㄧ巼:")
    for i in range(10):
        print(f"  鍗槦{i}: {utils[i]:.2%}")
    
    # 鎵炬渶浣冲崼鏄?
    candidates = [0, 1, 2, 3, 4, 5]
    distances = {i: 600 + i * 50 for i in candidates}
    elevations = {i: 60 - i * 5 for i in candidates}
    
    best = manager.find_best_satellite(candidates, distances, elevations)
    print(f"\n鍊欓€夊崼鏄?{candidates} 涓渶浣抽€夋嫨: 鍗槦{best}")
    
    # 缁熻淇℃伅
    stats = manager.get_statistics()
    print(f"\n缁熻淇℃伅:")
    print(f"  骞冲潎鍒╃敤鐜? {stats['average_utilization']:.2%}")
    print(f"  杩囪浇鍗槦鏁? {stats['num_overloaded']}")
    
    print("\n鉁?MEC绠＄悊鍣ㄦ祴璇曢€氳繃")


def test_convenience_functions():
    """娴嬭瘯渚挎嵎鍑芥暟"""
    print("\n" + "=" * 60)
    print("娴嬭瘯6: 渚挎嵎鍑芥暟")
    print("=" * 60)
    
    # 娴嬭瘯鍙傛暟
    data_mb = 5
    compute_gcycles = 2
    offload_ratio = 0.5
    distance = 800
    elevation = 45
    
    delay = compute_task_delay(data_mb, compute_gcycles, offload_ratio, distance, elevation)
    energy = compute_task_energy(data_mb, compute_gcycles, offload_ratio, distance, elevation)
    
    print(f"\n浠诲姟: {data_mb}MB, {compute_gcycles}G cycles, 位={offload_ratio}")
    print(f"  compute_task_delay() = {delay*1000:.2f} ms")
    print(f"  compute_task_energy() = {energy*1000:.4f} mJ")
    
    print("\n鉁?渚挎嵎鍑芥暟娴嬭瘯閫氳繃")


def test_delay_energy_tradeoff():
    """娴嬭瘯鏃跺欢-鑳借€楁潈琛?""
    print("\n" + "=" * 60)
    print("娴嬭瘯7: 鏃跺欢-鑳借€楁潈琛″垎鏋?)
    print("=" * 60)
    
    calc = OffloadingCalculator()
    
    # 鍥哄畾鍙傛暟
    data_bits = 10 * 8 * 1e6
    compute_cycles = 5 * 1e9
    max_delay = 5.0
    distance = 800
    elevation = 45
    
    print(f"\n浠诲姟: 10MB, 5G cycles")
    print(f"\n{'鍗歌浇姣斾緥':<10} {'鏃跺欢(ms)':<12} {'鑳借€?mJ)':<12} {'绛栫暐':<20}")
    print("-" * 55)
    
    for ratio in np.linspace(0, 1, 11):
        result = calc.compute_offloading_result(
            data_bits, compute_cycles, max_delay, ratio, distance, elevation
        )
        
        if ratio == 0:
            strategy = "瀹屽叏鏈湴"
        elif ratio == 1:
            strategy = "瀹屽叏鍗歌浇"
        else:
            strategy = f"閮ㄥ垎鍗歌浇({ratio:.0%})"
        
        print(f"{ratio:<10.1f} {result.total_delay*1000:<12.2f} "
              f"{result.total_energy*1000:<12.4f} {strategy:<20}")
    
    print("-" * 55)
    print("\n瑙傚療: 瀹屽叏鍗歌浇鏃跺欢鏈€灏忥紝瀹屽叏鏈湴鑳借€楁渶灏?)
    print("鉁?鏃跺欢-鑳借€楁潈琛″垎鏋愬畬鎴?)


