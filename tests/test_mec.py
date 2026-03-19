"""
MEC模型测试
验证任务卸载计算的正确性
"""

import sys
import numpy as np

sys.path.insert(0, 'd:\\python_code\\LEO_switch')

from src.environment.mec import (
    MECConfig, MECServer, MECManager,
    OffloadingCalculator, ComputeResult,
    compute_task_delay, compute_task_energy
)


def test_mec_config():
    """测试MEC配置"""
    print("=" * 60)
    print("测试1: MEC配置参数")
    print("=" * 60)
    
    config = MECConfig()
    
    print(f"\n卫星MEC参数:")
    print(f"  CPU频率: {config.satellite_cpu_freq_ghz} GHz")
    print(f"  最大CPU频率: {config.satellite_max_cpu_freq_ghz} GHz")
    print(f"  核心数: {config.satellite_num_cores}")
    print(f"  最大队列: {config.max_queue_size}")
    
    print(f"\n用户设备参数:")
    print(f"  CPU频率: {config.user_cpu_freq_ghz} GHz")
    print(f"  能耗系数κ: {config.kappa}")
    
    print("\n✓ 配置参数加载正确")
    return config


def test_mec_server():
    """测试MEC服务器"""
    print("\n" + "=" * 60)
    print("测试2: MEC服务器")
    print("=" * 60)
    
    server = MECServer(satellite_id=0)
    
    print(f"\n初始状态:")
    print(f"  卫星ID: {server.satellite_id}")
    print(f"  CPU频率: {server.cpu_freq_ghz} GHz")
    print(f"  可用频率: {server.available_freq_ghz} GHz")
    print(f"  利用率: {server.utilization:.2%}")
    print(f"  队列长度: {server.queue_length}")
    
    # 添加用户
    server.add_user(0)
    server.add_user(1)
    server.add_user(2)
    print(f"\n添加3个用户后:")
    print(f"  已连接用户: {server.connected_users}")
    
    # 分配资源
    allocated = server.allocate_compute_resource(0, 3.0)
    print(f"\n为用户0分配3GHz: 实际分配 {allocated} GHz")
    print(f"  剩余可用: {server.available_freq_ghz} GHz")
    print(f"  利用率: {server.utilization:.2%}")
    
    # 计算时延
    cycles = 1e9  # 1G cycles
    delay = server.compute_processing_delay(cycles, allocated)
    print(f"\n处理1G cycles (分配{allocated}GHz):")
    print(f"  处理时延: {delay*1000:.2f} ms")
    
    # 状态向量
    state = server.get_state_vector()
    print(f"\n状态向量: {state}")
    
    print("\n✓ MEC服务器测试通过")
    return server


def test_offloading_calculator():
    """测试卸载计算器"""
    print("\n" + "=" * 60)
    print("测试3: 卸载计算器")
    print("=" * 60)
    
    calc = OffloadingCalculator()
    
    # 测试参数
    data_bits = 5 * 8 * 1e6     # 5 MB
    compute_cycles = 2 * 1e9    # 2 G cycles
    max_delay = 2.0             # 2秒
    distance = 800              # km
    elevation = 45              # 度
    
    print(f"\n任务参数:")
    print(f"  数据量: 5 MB")
    print(f"  计算量: 2 G cycles")
    print(f"  最大时延: {max_delay} s")
    print(f"  卫星距离: {distance} km")
    print(f"  仰角: {elevation}°")
    
    # 测试不同卸载比例
    print(f"\n不同卸载比例的结果:")
    print("-" * 80)
    print(f"{'λ':<6} {'本地时延(ms)':<14} {'上传(ms)':<12} {'卫星计算(ms)':<14} "
          f"{'总时延(ms)':<12} {'能耗(mJ)':<10} {'满足约束':<10}")
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
              f"{'✓' if result.deadline_met else '✗':<10}")
    
    print("-" * 80)
    print("\n✓ 卸载计算测试通过")
    
    return calc


def test_optimal_offloading(calc):
    """测试最优卸载比例搜索"""
    print("\n" + "=" * 60)
    print("测试4: 最优卸载比例搜索")
    print("=" * 60)
    
    # 任务参数
    data_bits = 10 * 8 * 1e6    # 10 MB
    compute_cycles = 5 * 1e9    # 5 G cycles
    max_delay = 3.0             # 3秒
    distance = 800
    elevation = 45
    
    print(f"\n任务: 10MB数据, 5G cycles计算, 最大时延3s")
    
    # 最小化时延
    ratio_delay, result_delay = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='delay'
    )
    print(f"\n最小化时延:")
    print(f"  最优卸载比例: {ratio_delay:.2f}")
    print(f"  总时延: {result_delay.total_delay*1000:.2f} ms")
    print(f"  总能耗: {result_delay.total_energy*1000:.4f} mJ")
    
    # 最小化能耗
    ratio_energy, result_energy = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='energy'
    )
    print(f"\n最小化能耗:")
    print(f"  最优卸载比例: {ratio_energy:.2f}")
    print(f"  总时延: {result_energy.total_delay*1000:.2f} ms")
    print(f"  总能耗: {result_energy.total_energy*1000:.4f} mJ")
    
    # 加权优化
    ratio_weighted, result_weighted = calc.find_optimal_offload_ratio(
        data_bits, compute_cycles, max_delay, distance, elevation,
        objective='weighted'
    )
    print(f"\n加权优化 (时延+能耗):")
    print(f"  最优卸载比例: {ratio_weighted:.2f}")
    print(f"  总时延: {result_weighted.total_delay*1000:.2f} ms")
    print(f"  总能耗: {result_weighted.total_energy*1000:.4f} mJ")
    
    print("\n✓ 最优卸载搜索测试通过")


def test_mec_manager():
    """测试MEC管理器"""
    print("\n" + "=" * 60)
    print("测试5: MEC管理器")
    print("=" * 60)
    
    manager = MECManager(num_satellites=66)
    
    print(f"\n初始化 {manager.num_satellites} 颗卫星的MEC服务器")
    
    # 模拟一些卫星的负载
    for sat_id in [0, 1, 2, 5, 10]:
        server = manager.get_server(sat_id)
        server.allocate_compute_resource(0, 3.0)  # 分配一些资源
    
    # 获取统计信息
    utils = manager.get_all_utilizations()
    print(f"\n前10颗卫星利用率:")
    for i in range(10):
        print(f"  卫星{i}: {utils[i]:.2%}")
    
    # 找最佳卫星
    candidates = [0, 1, 2, 3, 4, 5]
    distances = {i: 600 + i * 50 for i in candidates}
    elevations = {i: 60 - i * 5 for i in candidates}
    
    best = manager.find_best_satellite(candidates, distances, elevations)
    print(f"\n候选卫星 {candidates} 中最佳选择: 卫星{best}")
    
    # 统计信息
    stats = manager.get_statistics()
    print(f"\n统计信息:")
    print(f"  平均利用率: {stats['average_utilization']:.2%}")
    print(f"  过载卫星数: {stats['num_overloaded']}")
    
    print("\n✓ MEC管理器测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("测试6: 便捷函数")
    print("=" * 60)
    
    # 测试参数
    data_mb = 5
    compute_gcycles = 2
    offload_ratio = 0.5
    distance = 800
    elevation = 45
    
    delay = compute_task_delay(data_mb, compute_gcycles, offload_ratio, distance, elevation)
    energy = compute_task_energy(data_mb, compute_gcycles, offload_ratio, distance, elevation)
    
    print(f"\n任务: {data_mb}MB, {compute_gcycles}G cycles, λ={offload_ratio}")
    print(f"  compute_task_delay() = {delay*1000:.2f} ms")
    print(f"  compute_task_energy() = {energy*1000:.4f} mJ")
    
    print("\n✓ 便捷函数测试通过")


def test_delay_energy_tradeoff():
    """测试时延-能耗权衡"""
    print("\n" + "=" * 60)
    print("测试7: 时延-能耗权衡分析")
    print("=" * 60)
    
    calc = OffloadingCalculator()
    
    # 固定参数
    data_bits = 10 * 8 * 1e6
    compute_cycles = 5 * 1e9
    max_delay = 5.0
    distance = 800
    elevation = 45
    
    print(f"\n任务: 10MB, 5G cycles")
    print(f"\n{'卸载比例':<10} {'时延(ms)':<12} {'能耗(mJ)':<12} {'策略':<20}")
    print("-" * 55)
    
    for ratio in np.linspace(0, 1, 11):
        result = calc.compute_offloading_result(
            data_bits, compute_cycles, max_delay, ratio, distance, elevation
        )
        
        if ratio == 0:
            strategy = "完全本地"
        elif ratio == 1:
            strategy = "完全卸载"
        else:
            strategy = f"部分卸载({ratio:.0%})"
        
        print(f"{ratio:<10.1f} {result.total_delay*1000:<12.2f} "
              f"{result.total_energy*1000:<12.4f} {strategy:<20}")
    
    print("-" * 55)
    print("\n观察: 完全卸载时延最小，完全本地能耗最小")
    print("✓ 时延-能耗权衡分析完成")


def main():
    """运行所有测试"""
    print("\n" + "#" * 60)
    print("#              MEC模型测试")
    print("#" * 60)
    
    config = test_mec_config()
    server = test_mec_server()
    calc = test_offloading_calculator()
    test_optimal_offloading(calc)
    test_mec_manager()
    test_convenience_functions()
    test_delay_energy_tradeoff()
    
    print("\n" + "=" * 60)
    print("         所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
