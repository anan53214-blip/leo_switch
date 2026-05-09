"""
Gymnasium环境测试
验证强化学习环境的正确性
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, 'd:\\python_code\\LEO_switch')

# 检查gymnasium是否安装
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("警告: gymnasium未安装，请运行: pip install gymnasium")


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
    """测试环境创建"""
    print("=" * 60)
    print("测试1: 环境创建")
    print("=" * 60)
    
    from src.environment.gym_env import LEOSatelliteEnv, EnvConfig
    
    config = EnvConfig(
        num_users=5,
        max_steps=100,
        seed=42
    )
    
    env = LEOSatelliteEnv(config)
    
    print(f"\n环境配置:")
    print(f"  卫星数量: {env.num_satellites}")
    print(f"  用户数量: {env.num_users}")
    print(f"  最大步数: {env.config.max_steps}")
    
    print(f"\n观测空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.shape}")
    
    print("\n✓ 环境创建成功")


def test_env_reset(env):
    """测试环境重置"""
    print("\n" + "=" * 60)
    print("测试2: 环境重置")
    print("=" * 60)
    
    obs, info = env.reset(seed=42)
    
    print(f"\n初始观测形状: {obs.shape}")
    print(f"初始信息: {info}")
    
    # 检查用户初始连接状态
    connected = sum(1 for u in env.user_manager.users 
                   if u.serving_satellite >= 0)
    print(f"\n初始连接用户数: {connected}/{env.num_users}")
    
    # 检查观测值范围
    print(f"\n观测值范围: [{obs.min():.4f}, {obs.max():.4f}]")
    
    print("\n✓ 环境重置成功")


def test_random_actions(env):
    """测试随机动作"""
    print("\n" + "=" * 60)
    print("测试3: 随机动作执行")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    total_reward = 0
    num_steps = 10
    
    print(f"\n执行 {num_steps} 步随机动作:")
    print("-" * 60)
    
    for step in range(num_steps):
        # 生成随机动作
        actions = np.random.rand(env.num_users, 2)
        actions[:, 0] *= env.handover_action_dim  # 切换动作
        actions[:, 1] = np.clip(actions[:, 1], 0, 1)  # 卸载比例
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        print(f"Step {step+1}: reward={reward:.4f}, "
              f"tasks={info['stats']['total_tasks']}, "
              f"handovers={info['stats']['total_handovers']}")
        
        if terminated or truncated:
            break
    
    print("-" * 60)
    print(f"总奖励: {total_reward:.4f}")
    print(f"平均奖励: {total_reward/num_steps:.4f}")
    
    print("\n✓ 随机动作执行成功")


def test_specific_actions(env):
    """测试特定动作"""
    print("\n" + "=" * 60)
    print("测试4: 特定动作测试")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 测试不同卸载比例
    offload_ratios = [0.0, 0.5, 1.0]
    
    print("\n测试不同卸载比例:")
    print("-" * 50)
    
    for ratio in offload_ratios:
        obs, _ = env.reset(seed=42)
        
        # 所有用户使用相同卸载比例，不切换
        actions = np.zeros((env.num_users, 2))
        actions[:, 0] = 0  # 不切换
        actions[:, 1] = ratio  # 卸载比例
        
        obs, reward, _, _, info = env.step(actions)
        
        print(f"λ={ratio:.1f}: reward={reward:.4f}, "
              f"tasks={info['stats']['completed_tasks']}/{info['stats']['total_tasks']}")
    
    print("-" * 50)
    print("\n✓ 特定动作测试成功")


def test_handover_actions(env):
    """测试切换动作"""
    print("\n" + "=" * 60)
    print("测试5: 切换动作测试")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 获取用户0的可见卫星
    user = env.user_manager.users[0]
    visible_sats = env._get_visible_satellites(user)
    
    print(f"\n用户0的可见卫星:")
    for i, sat in enumerate(visible_sats[:5]):
        print(f"  {i+1}. 卫星{sat.sat_id}: 仰角={sat.elevation_deg:.1f}°, "
              f"距离={sat.distance_km:.1f}km, RVT={sat.rvt_seconds:.0f}s")
    
    # 执行切换到第一个可见卫星
    actions = np.zeros((env.num_users, 2))
    actions[0, 0] = 1  # 用户0切换到第1个可见卫星
    actions[:, 1] = 0.5  # 卸载比例
    
    obs, reward, _, _, info = env.step(actions)
    
    print(f"\n切换后:")
    print(f"  用户0服务卫星: {env.user_manager.users[0].serving_satellite}")
    print(f"  切换统计: {info['stats']['successful_handovers']}/{info['stats']['total_handovers']}")
    
    print("\n✓ 切换动作测试成功")


def test_episode_run(env):
    """测试完整episode"""
    print("\n" + "=" * 60)
    print("测试6: 完整Episode运行")
    print("=" * 60)
    
    # 使用较短的episode
    env.config.max_steps = 50
    
    obs, _ = env.reset(seed=42)
    
    total_reward = 0
    step = 0
    
    while True:
        # 简单策略：不切换，50%卸载
        actions = np.zeros((env.num_users, 2))
        actions[:, 0] = 0
        actions[:, 1] = 0.5
        
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode完成:")
    print(f"  总步数: {step}")
    print(f"  总奖励: {total_reward:.4f}")
    print(f"  平均奖励: {total_reward/step:.4f}")
    print(f"\n最终统计:")
    for key, value in info['stats'].items():
        print(f"  {key}: {value}")
    
    print("\n✓ 完整Episode运行成功")


def test_graph_state(env):
    """测试图状态获取"""
    print("\n" + "=" * 60)
    print("测试7: 图状态获取 (用于GNN)")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    # 获取图状态
    graph_state = env.get_state_for_graph()
    
    print(f"\n图状态信息:")
    print(f"  卫星节点数: {len(graph_state['satellites'])}")
    print(f"  用户节点数: {len(graph_state['users'])}")
    print(f"  边数量: {len(graph_state['edges'])}")
    
    # 显示一些边信息
    print(f"\n部分用户-卫星边:")
    for edge in graph_state['edges'][:5]:
        print(f"  用户{edge['user_id']} → 卫星{edge['satellite_id']}: "
              f"距离={edge['distance']:.1f}km, 仰角={edge['elevation']:.1f}°")
    
    print("\n✓ 图状态获取成功")


def test_render(env):
    """测试渲染"""
    print("\n" + "=" * 60)
    print("测试8: 环境渲染")
    print("=" * 60)
    
    env.render_mode = "human"
    obs, _ = env.reset(seed=42)
    
    # 执行几步并渲染
    for _ in range(3):
        actions = np.random.rand(env.num_users, 2)
        actions[:, 0] *= env.handover_action_dim
        obs, reward, _, _, _ = env.step(actions)
        env.render()
    
    print("\n✓ 环境渲染测试成功")


def main():
    """运行所有测试"""
    if not GYM_AVAILABLE:
        print("\n请先安装gymnasium: pip install gymnasium")
        return
    
    print("\n" + "#" * 60)
    print("#          Gymnasium环境测试")
    print("#" * 60)
    
    test_env_creation()

    from src.environment.gym_env import LEOSatelliteEnv, EnvConfig

    env = LEOSatelliteEnv(EnvConfig(num_users=5, max_steps=100, seed=42))
    try:
        test_env_reset(env)
        test_random_actions(env)
        test_specific_actions(env)
        test_handover_actions(env)
        test_episode_run(env)
        test_graph_state(env)
        test_render(env)
    finally:
        env.close()
    
    print("\n" + "=" * 60)
    print("         所有测试通过! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
