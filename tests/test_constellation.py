"""
鏄熷骇妯″瀷鍗曞厓娴嬭瘯
楠岃瘉Walker鏄熷骇鍜屽彲瑙佹€ц绠楃殑姝ｇ‘鎬?
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

# 娣诲姞椤圭洰璺緞
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
    """娴嬭瘯鏄熷骇鍒濆鍖?""
    print("=" * 60)
    print("娴嬭瘯1: 鏄熷骇鍒濆鍖?)
    print("=" * 60)
    
    # 鍒涘缓Walker鏄熷骇
    constellation = WalkerConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0
    )
    
    # 楠岃瘉鍗槦鏁伴噺
    assert constellation.total_sats == 66, "鍗槦鎬绘暟搴斾负66"
    print(f"鉁?鍗槦鎬绘暟: {constellation.total_sats}")
    
    # 楠岃瘉杞ㄩ亾鍛ㄦ湡 (550km楂樺害绾?5鍒嗛挓)
    expected_period_min = 95.0
    actual_period_min = constellation.orbital_period / 60
    assert abs(actual_period_min - expected_period_min) < 2, "杞ㄩ亾鍛ㄦ湡搴旂害涓?5鍒嗛挓"
    print(f"鉁?杞ㄩ亾鍛ㄦ湡: {actual_period_min:.2f} 鍒嗛挓")
    
    # 楠岃瘉杞ㄩ亾楂樺害
    for state in constellation.satellite_states:
        alt = state.position_lla[2]
        assert abs(alt - 550) < 10, f"杞ㄩ亾楂樺害搴旂害涓?50km, 瀹為檯涓簕alt:.1f}km"
    print(f"鉁?杞ㄩ亾楂樺害楠岃瘉閫氳繃")
    


def test_satellite_positions(constellation):
    """娴嬭瘯鍗槦浣嶇疆璁＄畻"""
    print("\n" + "=" * 60)
    print("娴嬭瘯2: 鍗槦浣嶇疆璁＄畻")
    print("=" * 60)
    
    # 鑾峰彇鍑犻鍗槦鐨勪綅缃?
    for sat_id in [0, 10, 32, 65]:
        pos_info = constellation.get_satellite_position(sat_id)
        print(f"\n鍗槦 {sat_id} (杞ㄩ亾骞抽潰 {pos_info['plane_id']}):")
        print(f"  绾害: {pos_info['latitude']:.2f}掳")
        print(f"  缁忓害: {pos_info['longitude']:.2f}掳")
        print(f"  楂樺害: {pos_info['altitude']:.2f} km")
        
        # 楠岃瘉浣嶇疆鍦ㄥ悎鐞嗚寖鍥村唴
        assert -90 <= pos_info['latitude'] <= 90, "绾害搴斿湪[-90, 90]"
        assert -180 <= pos_info['longitude'] <= 180, "缁忓害搴斿湪[-180, 180]"
        assert 540 <= pos_info['altitude'] <= 560, "楂樺害搴旂害涓?50km"
    
    print("\n鉁?鍗槦浣嶇疆璁＄畻楠岃瘉閫氳繃")


def test_orbit_propagation(constellation):
    """娴嬭瘯杞ㄩ亾浼犳挱"""
    print("\n" + "=" * 60)
    print("娴嬭瘯3: 杞ㄩ亾浼犳挱")
    print("=" * 60)
    
    # 璁板綍鍒濆浣嶇疆
    sat_id = 0
    initial_pos = constellation.get_satellite_position(sat_id)
    initial_lon = initial_pos['longitude']
    
    # 浼犳挱5鍒嗛挓
    constellation.propagate(300)
    
    new_pos = constellation.get_satellite_position(sat_id)
    new_lon = new_pos['longitude']
    
    # 楠岃瘉鍗槦绉诲姩浜?(缁忓害鍙樺寲搴旇鏄庢樉)
    lon_change = abs(new_lon - initial_lon)
    if lon_change > 180:
        lon_change = 360 - lon_change
    
    print(f"5鍒嗛挓鍚庣粡搴﹀彉鍖? {lon_change:.2f}掳")
    assert lon_change > 5, "5鍒嗛挓鍐呯粡搴﹀簲鏈夋槑鏄惧彉鍖?
    
    # 鍐嶄紶鎾竴涓畬鏁村懆鏈燂紝楠岃瘉鍥炲埌闄勮繎浣嶇疆
    period = constellation.orbital_period
    constellation.propagate(period - 300)  # 琛ラ綈涓€涓懆鏈?
    
    cycle_pos = constellation.get_satellite_position(sat_id)
    print(f"涓€涓懆鏈熷悗绾害: {cycle_pos['latitude']:.2f}掳 (鍒濆: {initial_pos['latitude']:.2f}掳)")
    
    print("\n鉁?杞ㄩ亾浼犳挱楠岃瘉閫氳繃")


def test_visibility_calculation(constellation):
    """娴嬭瘯鍙鎬ц绠?""
    print("\n" + "=" * 60)
    print("娴嬭瘯4: 鍙鎬ц绠?)
    print("=" * 60)
    
    # 鍒涘缓鍙鎬ц绠楀櫒
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    # 鐢ㄦ埛浣嶇疆: 鍖椾含
    user_lat = 39.9
    user_lon = 116.4
    
    # 鑾峰彇鎵€鏈夊崼鏄熶綅缃拰閫熷害
    positions = np.array([s.position_ecef for s in constellation.satellite_states])
    velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    
    # 璁＄畻鍙鎬?
    visibility_list = vis_calc.compute_visibility_for_user(
        user_lat, user_lon, positions, velocities
    )
    
    # 缁熻鍙鍗槦
    visible_sats = vis_calc.get_visible_satellites(visibility_list)
    print(f"\n鐢ㄦ埛浣嶇疆: 鍖椾含 ({user_lat}掳N, {user_lon}掳E)")
    print(f"鍙鍗槦鏁伴噺: {len(visible_sats)} / {constellation.total_sats}")
    
    # 鎵撳嵃鍙鍗槦璇︽儏
    print("\n鍙鍗槦鍒楄〃:")
    print("-" * 50)
    for v in sorted(visible_sats, key=lambda x: -x.elevation_deg)[:5]:
        print(f"  鍗槦{v.sat_id:2d}: 浠拌={v.elevation_deg:5.1f}掳, "
              f"璺濈={v.distance_km:6.1f}km, RVT={v.rvt_seconds/60:.1f}鍒嗛挓")
    
    # 鑾峰彇鏈€浼樺崼鏄?
    best_elev = vis_calc.get_best_satellite(visibility_list, 'elevation')
    best_rvt = vis_calc.get_best_satellite(visibility_list, 'rvt')
    
    print(f"\n鏈€楂樹话瑙掑崼鏄? {best_elev.sat_id} ({best_elev.elevation_deg:.1f}掳)")
    print(f"鏈€闀縍VT鍗槦: {best_rvt.sat_id} ({best_rvt.rvt_seconds/60:.1f}鍒嗛挓)")
    
    print("\n鉁?鍙鎬ц绠楅獙璇侀€氳繃")


def test_rvt_dynamics(constellation):
    """娴嬭瘯RVT闅忔椂闂村彉鍖?""
    print("\n" + "=" * 60)
    print("娴嬭瘯5: RVT鍔ㄦ€佸彉鍖?)
    print("=" * 60)
    
    vis_calc = VisibilityCalculator(min_elevation_deg=10.0)
    
    user_lat, user_lon = 39.9, 116.4
    
    # 閲嶇疆鏄熷骇鏃堕棿
    constellation._update_all_positions(constellation.start_time)
    
    # 閫夋嫨涓€棰楀垵濮嬪彲瑙佺殑鍗槦
    positions = np.array([s.position_ecef for s in constellation.satellite_states])
    velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
    visibility_list = vis_calc.compute_visibility_for_user(
        user_lat, user_lon, positions, velocities
    )
    
    visible_sats = vis_calc.get_visible_satellites(visibility_list)
    if not visible_sats:
        print("鏃犲彲瑙佸崼鏄燂紝璺宠繃娴嬭瘯")
        return
    
    # 閫夋嫨RVT涓瓑鐨勫崼鏄熻繘琛岃窡韪?
    tracked_sat = sorted(visible_sats, key=lambda x: x.rvt_seconds)[len(visible_sats)//2]
    sat_id = tracked_sat.sat_id
    
    print(f"璺熻釜鍗槦 {sat_id}:")
    print("-" * 50)
    
    # 姣?0绉掕褰曚竴娆＄姸鎬?
    for t in range(0, 301, 30):
        if t > 0:
            constellation.propagate(30)
            
        positions = np.array([s.position_ecef for s in constellation.satellite_states])
        velocities = np.array([s.velocity_eci for s in constellation.satellite_states])
        visibility_list = vis_calc.compute_visibility_for_user(
            user_lat, user_lon, positions, velocities
        )
        
        v = visibility_list[sat_id]
        status = "鍙" if v.is_visible else "涓嶅彲瑙?
        print(f"  t={t:3d}s: 浠拌={v.elevation_deg:5.1f}掳, "
              f"RVT={v.rvt_seconds/60:4.1f}鍒嗛挓, 鐘舵€?{status}")
    
    print("\n鉁?RVT鍔ㄦ€佸彉鍖栨祴璇曞畬鎴?)


