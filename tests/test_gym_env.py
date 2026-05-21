"""
Gymnasium鐜娴嬭瘯
楠岃瘉寮哄寲瀛︿範鐜鐨勬纭€?
"""

import numpy as np
import pytest


# 妫€鏌ymnasium鏄惁瀹夎
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("璀﹀憡: gymnasium鏈畨瑁咃紝璇疯繍琛? pip install gymnasium")


def test_default_env_config_uses_20_user_competitive_scenario():
    from src.environment.gym_env import EnvConfig

    config = EnvConfig()

    assert config.num_users == 20
    assert config.task_arrival_prob == 0.35
    assert config.altitude_km == 550.0
    assert config.handover_delay_sec == 0.6
    assert config.rvt_threshold_sec == 60.0


@pytest.fixture
def env():
    from src.environment.gym_env import LEOSatelliteEnv, EnvConfig

    env_obj = LEOSatelliteEnv(EnvConfig(num_users=5, max_steps=100, seed=42))
    yield env_obj
    env_obj.close()


def test_env_creation():
    """娴嬭瘯鐜鍒涘缓"""
    print("=" * 60)
    print("娴嬭瘯1: 鐜鍒涘缓")
    print("=" * 60)
    
    from src.environment.gym_env import LEOSatelliteEnv, EnvConfig
    
    config = EnvConfig(
        num_users=5,
        max_steps=100,
        seed=42
    )
    
    env = LEOSatelliteEnv(config)
    
    print(f"\n鐜閰嶇疆:")
    print(f"  鍗槦鏁伴噺: {env.num_satellites}")
    print(f"  鐢ㄦ埛鏁伴噺: {env.num_users}")
    print(f"  鏈€澶ф鏁? {env.config.max_steps}")
    
    print(f"\n瑙傛祴绌洪棿: {env.observation_space.shape}")
    print(f"鍔ㄤ綔绌洪棿: {env.action_space.shape}")
    
    print("\n鉁?鐜鍒涘缓鎴愬姛")


def test_env_reset(env):
    """娴嬭瘯鐜閲嶇疆"""
    print("\n" + "=" * 60)
    print("娴嬭瘯2: 鐜閲嶇疆")
    print("=" * 60)
    
    obs, info = env.reset(seed=42)
    
    print(f"\n鍒濆瑙傛祴褰㈢姸: {obs.shape}")
    print(f"鍒濆淇℃伅: {info}")
    
    # 妫€鏌ョ敤鎴峰垵濮嬭繛鎺ョ姸鎬?
    connected = sum(1 for u in env.user_manager.users 
                   if u.serving_satellite >= 0)
    print(f"\n鍒濆杩炴帴鐢ㄦ埛鏁? {connected}/{env.num_users}")
    
    # 妫€鏌ヨ娴嬪€艰寖鍥?
    print(f"\n瑙傛祴鍊艰寖鍥? [{obs.min():.4f}, {obs.max():.4f}]")
    
    print("\n鉁?鐜閲嶇疆鎴愬姛")


def test_random_actions(env):
    """娴嬭瘯闅忔満鍔ㄤ綔"""
    print("\n" + "=" * 60)
    print("娴嬭瘯3: 闅忔満鍔ㄤ綔鎵ц")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    total_reward = 0
    num_steps = 10
    
    print(f"\n鎵ц {num_steps} 姝ラ殢鏈哄姩浣?")
    print("-" * 60)
    
    for step in range(num_steps):
        # 鐢熸垚闅忔満鍔ㄤ綔
        actions = np.random.rand(env.num_users, 2)
        actions[:, 0] *= env.handover_action_dim  # 鍒囨崲鍔ㄤ綔
        actions[:, 1] = np.clip(actions[:, 1], 0, 1)  # 鍗歌浇姣斾緥
        
        # 鎵ц鍔ㄤ綔
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        print(f"Step {step+1}: reward={reward:.4f}, "
              f"tasks={info['stats']['total_tasks']}, "
              f"handovers={info['stats']['total_handovers']}")
        
        if terminated or truncated:
            break
    
    print("-" * 60)
    print(f"鎬诲鍔? {total_reward:.4f}")
    print(f"骞冲潎濂栧姳: {total_reward/num_steps:.4f}")
    
    print("\n鉁?闅忔満鍔ㄤ綔鎵ц鎴愬姛")


def test_specific_actions(env):
    """娴嬭瘯鐗瑰畾鍔ㄤ綔"""
    print("\n" + "=" * 60)
    print("娴嬭瘯4: 鐗瑰畾鍔ㄤ綔娴嬭瘯")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 娴嬭瘯涓嶅悓鍗歌浇姣斾緥
    offload_ratios = [0.0, 0.5, 1.0]
    
    print("\n娴嬭瘯涓嶅悓鍗歌浇姣斾緥:")
    print("-" * 50)
    
    for ratio in offload_ratios:
        obs, _ = env.reset(seed=42)
        
        # 鎵€鏈夌敤鎴蜂娇鐢ㄧ浉鍚屽嵏杞芥瘮渚嬶紝涓嶅垏鎹?
        actions = np.zeros((env.num_users, 2))
        actions[:, 0] = 0  # 涓嶅垏鎹?
        actions[:, 1] = ratio  # 鍗歌浇姣斾緥
        
        obs, reward, _, _, info = env.step(actions)
        
        print(f"位={ratio:.1f}: reward={reward:.4f}, "
              f"tasks={info['stats']['completed_tasks']}/{info['stats']['total_tasks']}")
    
    print("-" * 50)
    print("\n鉁?鐗瑰畾鍔ㄤ綔娴嬭瘯鎴愬姛")


def test_handover_actions(env):
    """娴嬭瘯鍒囨崲鍔ㄤ綔"""
    print("\n" + "=" * 60)
    print("娴嬭瘯5: 鍒囨崲鍔ㄤ綔娴嬭瘯")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 鑾峰彇鐢ㄦ埛0鐨勫彲瑙佸崼鏄?
    user = env.user_manager.users[0]
    visible_sats = env._get_visible_satellites(user)
    
    print(f"\n鐢ㄦ埛0鐨勫彲瑙佸崼鏄?")
    for i, sat in enumerate(visible_sats[:5]):
        print(f"  {i+1}. 鍗槦{sat.sat_id}: 浠拌={sat.elevation_deg:.1f}掳, "
              f"璺濈={sat.distance_km:.1f}km, RVT={sat.rvt_seconds:.0f}s")
    
    # 鎵ц鍒囨崲鍒扮涓€涓彲瑙佸崼鏄?
    actions = np.zeros((env.num_users, 2))
    actions[0, 0] = 1  # 鐢ㄦ埛0鍒囨崲鍒扮1涓彲瑙佸崼鏄?
    actions[:, 1] = 0.5  # 鍗歌浇姣斾緥
    
    obs, reward, _, _, info = env.step(actions)
    
    print(f"\n鍒囨崲鍚?")
    print(f"  鐢ㄦ埛0鏈嶅姟鍗槦: {env.user_manager.users[0].serving_satellite}")
    print(f"  鍒囨崲缁熻: {info['stats']['successful_handovers']}/{info['stats']['total_handovers']}")
    
    print("\n鉁?鍒囨崲鍔ㄤ綔娴嬭瘯鎴愬姛")


def test_episode_run(env):
    """娴嬭瘯瀹屾暣episode"""
    print("\n" + "=" * 60)
    print("娴嬭瘯6: 瀹屾暣Episode杩愯")
    print("=" * 60)
    
    # 浣跨敤杈冪煭鐨別pisode
    env.config.max_steps = 50
    
    obs, _ = env.reset(seed=42)
    
    total_reward = 0
    step = 0
    
    while True:
        # 绠€鍗曠瓥鐣ワ細涓嶅垏鎹紝50%鍗歌浇
        actions = np.zeros((env.num_users, 2))
        actions[:, 0] = 0
        actions[:, 1] = 0.5
        
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode瀹屾垚:")
    print(f"  鎬绘鏁? {step}")
    print(f"  鎬诲鍔? {total_reward:.4f}")
    print(f"  骞冲潎濂栧姳: {total_reward/step:.4f}")
    print(f"\n鏈€缁堢粺璁?")
    for key, value in info['stats'].items():
        print(f"  {key}: {value}")
    
    print("\n鉁?瀹屾暣Episode杩愯鎴愬姛")


def test_graph_state(env):
    """娴嬭瘯鍥剧姸鎬佽幏鍙?""
    print("\n" + "=" * 60)
    print("娴嬭瘯7: 鍥剧姸鎬佽幏鍙?(鐢ㄤ簬GNN)")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 鑾峰彇鍥剧姸鎬?
    graph_state = env.get_state_for_graph()
    
    print(f"\n鍥剧姸鎬佷俊鎭?")
    print(f"  鍗槦鑺傜偣鏁? {len(graph_state['satellites'])}")
    print(f"  鐢ㄦ埛鑺傜偣鏁? {len(graph_state['users'])}")
    print(f"  杈规暟閲? {len(graph_state['edges'])}")
    
    # 鏄剧ず涓€浜涜竟淇℃伅
    print(f"\n閮ㄥ垎鐢ㄦ埛-鍗槦杈?")
    for edge in graph_state['edges'][:5]:
        print(f"  鐢ㄦ埛{edge['user_id']} 鈫?鍗槦{edge['satellite_id']}: "
              f"璺濈={edge['distance']:.1f}km, 浠拌={edge['elevation']:.1f}掳")
    
    print("\n鉁?鍥剧姸鎬佽幏鍙栨垚鍔?)


def test_render(env):
    """娴嬭瘯娓叉煋"""
    print("\n" + "=" * 60)
    print("娴嬭瘯8: 鐜娓叉煋")
    print("=" * 60)
    
    env.render_mode = "human"
    obs, _ = env.reset(seed=42)
    
    # 鎵ц鍑犳骞舵覆鏌?
    for _ in range(3):
        actions = np.random.rand(env.num_users, 2)
        actions[:, 0] *= env.handover_action_dim
        obs, reward, _, _, _ = env.step(actions)
        env.render()
    
    print("\n鉁?鐜娓叉煋娴嬭瘯鎴愬姛")


