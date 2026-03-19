"""测试HAN集成"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from scripts.train import HANMAPPOTrainer, TrainConfig
import numpy as np

# 测试配置
config = TrainConfig()
config.num_users = 3
config.max_steps = 20
config.total_timesteps = 128
config.n_steps = 32
config.device = 'cpu'

print('Creating trainer...')
trainer = HANMAPPOTrainer(config)

print('Testing HAN encoding...')
obs, info = trainer.env.reset()
observations, global_state, available_actions = trainer._get_observations(obs)

print(f'Raw obs shape: {obs.shape}')
print(f'HAN encoded obs shape: {observations.shape}')  
print(f'Global state shape: {global_state.shape}')
print(f'Available actions shape: {available_actions.shape}')

print('Testing MAPPO.act()...')
actions, log_probs, value = trainer.mappo.act(observations, global_state, available_actions)

print(f'Handover actions: {actions["handover"]}')
print(f'Offload ratios: {actions["offload"]}')
print(f'State value: {value:.4f}')

print('HAN integration test PASSED!')
