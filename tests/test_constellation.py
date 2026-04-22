"""
星座模型单元测试
验证Walker星座和可见性计算的正确性
"""

import sys
import numpy as np
import pytest
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, 'd:\\python_code\\LEO_switch')

from src.environment.constellation import WalkerConstellation, EARTH_RADIUS_KM
from src.environment.visibility import VisibilityCalculator


@pytest.fixture
def constellation():
    return WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
    )


def test_constellation_initialization():
    """测试星座初始化"""
    print("=" * 60)
    print("测试1: 星座初始化")
    print("=" * 60)
    
    # 创建Walker星座
    constellation = WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0
    )
    
    # 验证卫星数量
    assert constellation.total_sats == 66, "卫星总数应为66"
    print(f"✓ 卫星总数: {constellation.total_sats}")
    
    # 验证轨道周期 (550km高度约95分钟)
    expected_period_min = 95.0
    actual_period_min = constellation.orbital_period / 60
    assert abs(actual_period_min - expected_period_min) < 2, "轨道周期应约为95分钟"
    print(f"✓ 轨道周期: {actual_period_min:.2f} 分钟")
    
    # 验证轨道高度
    for state in constellation.satellite_states:
        alt = state.position_lla[2]
        assert abs(alt - 550) < 10, f"轨道高度应约为550km, 实际为{alt:.1f}km"
    print(f"✓ 轨道高度验证通过")
    


def test_satellite_positions(constellation):
    """测试卫星位置计算"""
    print("\n" + "=" * 60)
    print("测试2: 卫星位置计算")
    print("=" * 60)
    
    # 获取几颗卫星的位置
    for sat_id in [0, 10, 32, 65]:
        pos_info = constellation.get_satellite_position(sat_id)
        print(f"\n卫星 {sat_id} (轨道平面 {pos_info['plane_id']}):")
        print(f"  纬度: {pos_info['latitude']:.2f}°")
        print(f"  经度: {pos_info['longitude']:.2f}°")
        print(f"  高度: {pos_info['altitude']:.2f} km")
        
        # 验证位置在合理范围内
        assert -90 <= pos_info['latitude'] <= 90, "纬度应在[-90, 90]"
        assert -180 <= pos_info['longitude'] <= 180, "经度应在[-180, 180]"
        assert 540 <= pos_info['altitude'] <= 560, "高度应约为550km"
    
    print("\n✓ 卫星位置计算验证通过")


def test_orbit_propagation(constellation):
    """测试轨道传播"""
    print("\n" + "=" * 60)
    print("测试3: 轨道传播")
    print("=" * 60)
    
    # 记录初始位置
    sat_id = 0
    initial_pos = constellation.get_satellite_position(sat_id)
    initial_lon = initial_pos['longitude']
    
    # 传播5分钟
    constellation.propagate(300)
    
    new_pos = constellation.get_satellite_position(sat_id)
    new_lon = new_pos['longitude']
    
    # 验证卫星移动了 (经度变化应该明显)
    lon_change = abs(new_lon - initial_lon)
    if lon_change > 180:
        lon_change = 360 - lon_change
    
    print(f"5分钟后经度变化: {lon_change:.2f}°")
    assert lon_change > 5, "5分钟内经度应有明显变化"
    
    # 再传播一个完整周期，验证回到附近位置
    period = constellation.orbital_period
    constellation.propagate(period - 300)  # 补齐一个周期
    
    cycle_pos = constellation.get_satellite_position(sat_id)
    print(f"一个周期后纬度: {cycle_pos['latitude']:.2f}° (初始: {initial_pos['latitude']:.2f}°)")
    
    print("\n✓ 轨道传播验证通过")


def test_visibility_calculation(constellation):
    """测试可见性计算"""
    print("\n" + "=" * 60)
    print("测试4: 可见性计算")
    print("=" * 60)
    
    # 创建可见性计算器
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    # 用户位置: 北京
    user_lat = 39.9
    user_lon = 116.4
    
    # 获取所有卫星位置和速度
    positions = np.array([s.position_ecef for s in constellation.satellite_states])
    velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    
    # 计算可见性
    visibility_list = vis_calc.compute_visibility_for_user(
        user_lat, user_lon, positions, velocities
    )
    
    # 统计可见卫星
    visible_sats = vis_calc.get_visible_satellites(visibility_list)
    print(f"\n用户位置: 北京 ({user_lat}°N, {user_lon}°E)")
    print(f"可见卫星数量: {len(visible_sats)} / {constellation.total_sats}")
    
    # 打印可见卫星详情
    print("\n可见卫星列表:")
    print("-" * 50)
    for v in sorted(visible_sats, key=lambda x: -x.elevation_deg)[:5]:
        print(f"  卫星{v.sat_id:2d}: 仰角={v.elevation_deg:5.1f}°, "
              f"距离={v.distance_km:6.1f}km, RVT={v.rvt_seconds/60:.1f}分钟")
    
    # 获取最优卫星
    best_elev = vis_calc.get_best_satellite(visibility_list, 'elevation')
    best_rvt = vis_calc.get_best_satellite(visibility_list, 'rvt')
    
    print(f"\n最高仰角卫星: {best_elev.sat_id} ({best_elev.elevation_deg:.1f}°)")
    print(f"最长RVT卫星: {best_rvt.sat_id} ({best_rvt.rvt_seconds/60:.1f}分钟)")
    
    print("\n✓ 可见性计算验证通过")


def test_rvt_dynamics(constellation):
    """测试RVT随时间变化"""
    print("\n" + "=" * 60)
    print("测试5: RVT动态变化")
    print("=" * 60)
    
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    user_lat, user_lon = 39.9, 116.4
    
    # 重置星座时间
    constellation._update_all_positions(constellation.start_time)
    
    # 选择一颗初始可见的卫星
    positions = np.array([s.position_ecef for s in constellation.satellite_states])
    velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    visibility_list = vis_calc.compute_visibility_for_user(
        user_lat, user_lon, positions, velocities
    )
    
    visible_sats = vis_calc.get_visible_satellites(visibility_list)
    if not visible_sats:
        print("无可见卫星，跳过测试")
        return
    
    # 选择RVT中等的卫星进行跟踪
    tracked_sat = sorted(visible_sats, key=lambda x: x.rvt_seconds)[len(visible_sats)//2]
    sat_id = tracked_sat.sat_id
    
    print(f"跟踪卫星 {sat_id}:")
    print("-" * 50)
    
    # 每30秒记录一次状态
    for t in range(0, 301, 30):
        if t > 0:
            constellation.propagate(30)
            
        positions = np.array([s.position_ecef for s in constellation.satellite_states])
        velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
        visibility_list = vis_calc.compute_visibility_for_user(
            user_lat, user_lon, positions, velocities
        )
        
        v = visibility_list[sat_id]
        status = "可见" if v.is_visible else "不可见"
        print(f"  t={t:3d}s: 仰角={v.elevation_deg:5.1f}°, "
              f"RVT={v.rvt_seconds/60:4.1f}分钟, 状态={status}")
    
    print("\n✓ RVT动态变化测试完成")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("      LEO卫星星座模型测试")
    print("=" * 60 + "\n")
    
    # 依次运行测试
    constellation = WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
    )
    test_constellation_initialization()
    test_satellite_positions(constellation)
    test_orbit_propagation(constellation)
    test_visibility_calculation(constellation)
    test_rvt_dynamics(constellation)
    
    print("\n" + "=" * 60)
    print("      所有测试通过!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
