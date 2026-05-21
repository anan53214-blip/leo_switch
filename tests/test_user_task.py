"""
鐢ㄦ埛鍜屼换鍔℃ā鍨嬫祴璇?
楠岃瘉鐢ㄦ埛鐢熸垚鍜屼换鍔″垱寤虹殑姝ｇ‘鎬?
"""

import numpy as np
import pytest

from src.environment.user import (
    User, UserPosition, UserState, 
    UserGenerator, UserManager
)
from src.environment.task import (
    Task, TaskType, TaskGenerator, TaskManager, TaskConfig
)
from src.environment.constellation import WalkerConstellation
from src.environment.visibility import VisibilityCalculator


@pytest.fixture
def users():
    generator = UserGenerator(seed=42)
    return generator.generate_users_in_circle(
        center_lat=0.0,
        center_lon=0.0,
        radius_deg=2.0,
        num_users=10,
    )


@pytest.fixture
def tasks(users):
    task_gen = TaskGenerator(seed=42)
    return task_gen.generate_tasks_for_users(
        user_ids=[u.user_id for u in users],
        current_time=0.0,
    )


def test_user_generation():
    """娴嬭瘯鐢ㄦ埛鐢熸垚"""
    print("=" * 60)
    print("娴嬭瘯1: 鐢ㄦ埛鐢熸垚")
    print("=" * 60)
    
    # 鍒涘缓鐢ㄦ埛鐢熸垚鍣?
    generator = UserGenerator(seed=42)
    
    # 鎸夌収瑕佹眰锛氫互(0, 0)涓哄渾蹇冿紝鍗婂緞2掳鐨勫渾鍐呯敓鎴愮敤鎴?
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
    
    print(f"\n鐢熸垚鍖哄煙: 鍦嗗績({center_lat}掳, {center_lon}掳), 鍗婂緞{radius_deg}掳")
    print(f"鐢熸垚鐢ㄦ埛鏁? {len(users)}")
    print("\n鐢ㄦ埛浣嶇疆鍒楄〃:")
    print("-" * 50)
    print(f"{'鐢ㄦ埛ID':^8} {'绾害(掳)':^12} {'缁忓害(掳)':^12} {'鍒板渾蹇冭窛绂?掳)':^14}")
    print("-" * 50)
    
    for user in users:
        # 璁＄畻鍒板渾蹇冪殑璺濈
        dist = np.sqrt(
            (user.position.latitude - center_lat)**2 +
            (user.position.longitude - center_lon)**2
        )
        print(f"{user.user_id:^8} {user.position.latitude:^12.4f} "
              f"{user.position.longitude:^12.4f} {dist:^14.4f}")
        
        # 楠岃瘉鍦ㄥ渾鍐?
        assert dist <= radius_deg * 1.1, f"鐢ㄦ埛{user.user_id}瓒呭嚭鍦嗗舰鑼冨洿"
    
    print("-" * 50)
    print("鉁?鐢ㄦ埛鐢熸垚娴嬭瘯閫氳繃")
    


def test_user_manager(users):
    """娴嬭瘯鐢ㄦ埛绠＄悊鍣?""
    print("\n" + "=" * 60)
    print("娴嬭瘯2: 鐢ㄦ埛绠＄悊鍣?)
    print("=" * 60)
    
    manager = UserManager(users)
    
    # 鑾峰彇鎵€鏈変綅缃?
    positions_ecef = manager.get_all_positions_ecef()
    positions_lla = manager.get_all_positions_lla()
    
    print(f"\n鐢ㄦ埛鏁伴噺: {manager.num_users}")
    print(f"ECEF浣嶇疆鐭╅樀褰㈢姸: {positions_ecef.shape}")
    print(f"LLA浣嶇疆鐭╅樀褰㈢姸: {positions_lla.shape}")
    
    # 妯℃嫙杩炴帴
    users[0].connect_to_satellite(sat_id=5, current_time=0.0)
    users[1].connect_to_satellite(sat_id=10, current_time=0.0)
    users[2].state = UserState.BLOCKED
    
    print(f"\n宸茶繛鎺ョ敤鎴? {len(manager.get_connected_users())}")
    print(f"绌洪棽鐢ㄦ埛: {len(manager.get_idle_users())}")
    print(f"闃诲鐢ㄦ埛: {len(manager.get_blocked_users())}")
    
    # 妯℃嫙鍒囨崲
    users[0].start_handover(target_sat_id=15, current_time=1.0)
    users[0].complete_handover(new_sat_id=15, current_time=1.5, success=True)
    
    # 鏇存柊缁熻
    manager.update_all_statistics(current_time=10.0)
    manager.print_status()
    
    print("\n鉁?鐢ㄦ埛绠＄悊鍣ㄦ祴璇曢€氳繃")
    


def test_task_generation(users):
    """娴嬭瘯浠诲姟鐢熸垚"""
    print("\n" + "=" * 60)
    print("娴嬭瘯3: 浠诲姟鐢熸垚")
    print("=" * 60)
    
    # 鍒涘缓浠诲姟鐢熸垚鍣?
    task_gen = TaskGenerator(seed=42)
    
    # 涓烘墍鏈夌敤鎴风敓鎴愪换鍔?
    tasks = task_gen.generate_tasks_for_users(
        user_ids=[u.user_id for u in users],
        current_time=0.0
    )
    
    print(f"\n鐢熸垚浠诲姟鏁? {len(tasks)}")
    print("\n浠诲姟璇︽儏:")
    print("-" * 70)
    print(f"{'浠诲姟ID':^8} {'鐢ㄦ埛ID':^8} {'绫诲瀷':^8} {'鏁版嵁閲?MB)':^12} "
          f"{'璁＄畻閲?GC)':^12} {'鏃跺欢闄愬埗(s)':^12}")
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
    print(f"\n浠诲姟绫诲瀷鍒嗗竷:")
    print(f"  杞婚噺绾? {type_count[TaskType.LIGHT]}")
    print(f"  涓瓑: {type_count[TaskType.MEDIUM]}")
    print(f"  閲嶅瀷: {type_count[TaskType.HEAVY]}")
    
    print("\n鉁?浠诲姟鐢熸垚娴嬭瘯閫氳繃")
    


def test_task_manager(tasks):
    """娴嬭瘯浠诲姟绠＄悊鍣?""
    print("\n" + "=" * 60)
    print("娴嬭瘯4: 浠诲姟绠＄悊鍣?)
    print("=" * 60)
    
    manager = TaskManager()
    
    # 娣诲姞浠诲姟
    for task in tasks:
        manager.add_task(task)
    
    print(f"\n娣诲姞浠诲姟鏁? {len(tasks)}")
    
    # 妯℃嫙浠诲姟澶勭悊
    for i, task in enumerate(tasks[:5]):
        manager.start_task(task.task_id)
        
        # 妯℃嫙瀹屾垚
        if i < 3:
            task.total_delay = 0.5 + i * 0.3
            task.total_energy = 0.1 + i * 0.05
            manager.complete_task(task.task_id, current_time=1.0 + i)
        else:
            manager.fail_task(task.task_id, current_time=10.0)
    
    manager.print_status()
    
    print("\n鉁?浠诲姟绠＄悊鍣ㄦ祴璇曢€氳繃")


def test_integrated_scenario():
    """闆嗘垚娴嬭瘯锛氱敤鎴?鏄熷骇-鍙鎬?""
    print("\n" + "=" * 60)
    print("娴嬭瘯5: 闆嗘垚鍦烘櫙娴嬭瘯")
    print("=" * 60)
    
    # 1. 鍒涘缓鏄熷骇
    constellation = WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0
    )
    
    # 2. 鍦?0, 0)闄勮繎鐢熸垚鐢ㄦ埛
    user_gen = UserGenerator(seed=42)
    users = user_gen.generate_users_in_circle(
        center_lat=0.0,
        center_lon=0.0,
        radius_deg=2.0,
        num_users=10
    )
    user_manager = UserManager(users)
    
    # 3. 鍒涘缓鍙鎬ц绠楀櫒
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    # 4. 鑾峰彇鍗槦浣嶇疆
    sat_positions = np.array([s.position_ecef for s in constellation.satellite_states])
    sat_velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    
    print(f"\n鍖哄煙: 璧ら亾闄勮繎 (0掳N, 0掳E)锛屽崐寰?掳")
    print(f"鐢ㄦ埛鏁? {len(users)}")
    print(f"鍗槦鏁? {constellation.total_sats}")
    
    # 5. 涓烘瘡涓敤鎴疯绠楀彲瑙佸崼鏄?
    print("\n鐢ㄦ埛鍙鎬у垎鏋?")
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
            print(f"鐢ㄦ埛{user.user_id}: 鍙{len(visible_sats)}棰楀崼鏄? "
                  f"鏈€浣冲崼鏄焮best_sat.sat_id}(浠拌{best_sat.elevation_deg:.1f}掳, "
                  f"RVT={best_sat.rvt_seconds/60:.1f}鍒嗛挓)")
            
            # 杩炴帴鍒版渶浣冲崼鏄?
            user.connect_to_satellite(best_sat.sat_id, current_time=0.0)
        else:
            print(f"鐢ㄦ埛{user.user_id}: 鏃犲彲瑙佸崼鏄?)
            user.state = UserState.BLOCKED
    
    print("-" * 60)
    
    # 6. 鎵撳嵃缁熻
    user_manager.update_all_statistics(0.0)
    stats = user_manager.get_statistics_summary()
    print(f"\n杩炴帴鎴愬姛: {stats['connected_users']}/{stats['num_users']}")
    
    print("\n鉁?闆嗘垚鍦烘櫙娴嬭瘯閫氳繃")


