"""
用户和任务模型测试
验证用户生成和任务创建的正确性
"""

import sys
import numpy as np

sys.path.insert(0, 'd:\\python_code\\LEO_switch')

from src.environment.user import (
    User, UserPosition, UserState, 
    UserGenerator, UserManager
)
from src.environment.task import (
    Task, TaskType, TaskGenerator, TaskManager, TaskConfig
)
from src.environment.constellation import WalkerConstellation
from src.environment.visibility import VisibilityCalculator


def test_user_generation():
    """测试用户生成"""
    print("=" * 60)
    print("测试1: 用户生成")
    print("=" * 60)
    
    # 创建用户生成器
    generator = UserGenerator(seed=42)
    
    # 按照要求：以(0, 0)为圆心，半径2°的圆内生成用户
    center_lat = 0.0
    center_lon = 0.0
    radius_deg = 2.0
    num_users = 10
    
    users = generator.generate_users_in_circle(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_deg=radius_deg,
        num_users=num_users
    )
    
    print(f"\n生成区域: 圆心({center_lat}°, {center_lon}°), 半径{radius_deg}°")
    print(f"生成用户数: {len(users)}")
    print("\n用户位置列表:")
    print("-" * 50)
    print(f"{'用户ID':^8} {'纬度(°)':^12} {'经度(°)':^12} {'到圆心距离(°)':^14}")
    print("-" * 50)
    
    for user in users:
        # 计算到圆心的距离
        dist = np.sqrt(
            (user.position.latitude - center_lat)**2 +
            (user.position.longitude - center_lon)**2
        )
        print(f"{user.user_id:^8} {user.position.latitude:^12.4f} "
              f"{user.position.longitude:^12.4f} {dist:^14.4f}")
        
        # 验证在圆内
        assert dist <= radius_deg * 1.1, f"用户{user.user_id}超出圆形范围"
    
    print("-" * 50)
    print("✓ 用户生成测试通过")
    
    return users


def test_user_manager(users):
    """测试用户管理器"""
    print("\n" + "=" * 60)
    print("测试2: 用户管理器")
    print("=" * 60)
    
    manager = UserManager(users)
    
    # 获取所有位置
    positions_ecef = manager.get_all_positions_ecef()
    positions_lla = manager.get_all_positions_lla()
    
    print(f"\n用户数量: {manager.num_users}")
    print(f"ECEF位置矩阵形状: {positions_ecef.shape}")
    print(f"LLA位置矩阵形状: {positions_lla.shape}")
    
    # 模拟连接
    users[0].connect_to_satellite(sat_id=5, current_time=0.0)
    users[1].connect_to_satellite(sat_id=10, current_time=0.0)
    users[2].state = UserState.BLOCKED
    
    print(f"\n已连接用户: {len(manager.get_connected_users())}")
    print(f"空闲用户: {len(manager.get_idle_users())}")
    print(f"阻塞用户: {len(manager.get_blocked_users())}")
    
    # 模拟切换
    users[0].start_handover(target_sat_id=15, current_time=1.0)
    users[0].complete_handover(new_sat_id=15, current_time=1.5, success=True)
    
    # 更新统计
    manager.update_all_statistics(current_time=10.0)
    manager.print_status()
    
    print("\n✓ 用户管理器测试通过")
    
    return manager


def test_task_generation(users):
    """测试任务生成"""
    print("\n" + "=" * 60)
    print("测试3: 任务生成")
    print("=" * 60)
    
    # 创建任务生成器
    task_gen = TaskGenerator(seed=42)
    
    # 为所有用户生成任务
    tasks = task_gen.generate_tasks_for_users(
        user_ids=[u.user_id for u in users],
        current_time=0.0
    )
    
    print(f"\n生成任务数: {len(tasks)}")
    print("\n任务详情:")
    print("-" * 70)
    print(f"{'任务ID':^8} {'用户ID':^8} {'类型':^8} {'数据量(MB)':^12} "
          f"{'计算量(GC)':^12} {'时延限制(s)':^12}")
    print("-" * 70)
    
    type_count = {TaskType.LIGHT: 0, TaskType.MEDIUM: 0, TaskType.HEAVY: 0}
    
    for task in tasks:
        type_name = task.task_type.name
        type_count[task.task_type] += 1
        print(f"{task.task_id:^8} {task.user_id:^8} {type_name:^8} "
              f"{task.get_data_size_MB():^12.2f} "
              f"{task.get_computation_GCycles():^12.2f} "
              f"{task.max_delay:^12.2f}")
    
    print("-" * 70)
    print(f"\n任务类型分布:")
    print(f"  轻量级: {type_count[TaskType.LIGHT]}")
    print(f"  中等: {type_count[TaskType.MEDIUM]}")
    print(f"  重型: {type_count[TaskType.HEAVY]}")
    
    print("\n✓ 任务生成测试通过")
    
    return tasks


def test_task_manager(tasks):
    """测试任务管理器"""
    print("\n" + "=" * 60)
    print("测试4: 任务管理器")
    print("=" * 60)
    
    manager = TaskManager()
    
    # 添加任务
    for task in tasks:
        manager.add_task(task)
    
    print(f"\n添加任务数: {len(tasks)}")
    
    # 模拟任务处理
    for i, task in enumerate(tasks[:5]):
        manager.start_task(task.task_id)
        
        # 模拟完成
        if i < 3:
            task.total_delay = 0.5 + i * 0.3
            task.total_energy = 0.1 + i * 0.05
            manager.complete_task(task.task_id, current_time=1.0 + i)
        else:
            manager.fail_task(task.task_id, current_time=10.0)
    
    manager.print_status()
    
    print("\n✓ 任务管理器测试通过")


def test_integrated_scenario():
    """集成测试：用户-星座-可见性"""
    print("\n" + "=" * 60)
    print("测试5: 集成场景测试")
    print("=" * 60)
    
    # 1. 创建星座
    constellation = WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0
    )
    
    # 2. 在(0, 0)附近生成用户
    user_gen = UserGenerator(seed=42)
    users = user_gen.generate_users_in_circle(
        center_lat=0.0,
        center_lon=0.0,
        radius_deg=2.0,
        num_users=10
    )
    user_manager = UserManager(users)
    
    # 3. 创建可见性计算器
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    # 4. 获取卫星位置
    sat_positions = np.array([s.position_ecef for s in constellation.satellite_states])
    sat_velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    
    print(f"\n区域: 赤道附近 (0°N, 0°E)，半径2°")
    print(f"用户数: {len(users)}")
    print(f"卫星数: {constellation.total_sats}")
    
    # 5. 为每个用户计算可见卫星
    print("\n用户可见性分析:")
    print("-" * 60)
    
    for user in users:
        visibility = vis_calc.compute_visibility_for_user(
            user.position.latitude,
            user.position.longitude,
            sat_positions,
            sat_velocities
        )
        
        visible_sats = vis_calc.get_visible_satellites(visibility)
        
        if visible_sats:
            best_sat = vis_calc.get_best_satellite(visibility, 'elevation')
            print(f"用户{user.user_id}: 可见{len(visible_sats)}颗卫星, "
                  f"最佳卫星{best_sat.sat_id}(仰角{best_sat.elevation_deg:.1f}°, "
                  f"RVT={best_sat.rvt_seconds/60:.1f}分钟)")
            
            # 连接到最佳卫星
            user.connect_to_satellite(best_sat.sat_id, current_time=0.0)
        else:
            print(f"用户{user.user_id}: 无可见卫星")
            user.state = UserState.BLOCKED
    
    print("-" * 60)
    
    # 6. 打印统计
    user_manager.update_all_statistics(0.0)
    stats = user_manager.get_statistics_summary()
    print(f"\n连接成功: {stats['connected_users']}/{stats['num_users']}")
    
    print("\n✓ 集成场景测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("        用户和任务模型测试")
    print("=" * 60 + "\n")
    
    users = test_user_generation()
    test_user_manager(users)
    tasks = test_task_generation(users)
    test_task_manager(tasks)
    test_integrated_scenario()
    
    print("\n" + "=" * 60)
    print("        所有测试通过!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()