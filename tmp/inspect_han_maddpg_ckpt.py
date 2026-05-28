import torch
from pathlib import Path
base=Path('results/baseline_compare/20260525_220324/learned_baselines/han_maddpg')
for name in ['best_model.pt','final_model.pt','checkpoint_200000.pt','checkpoint_800000.pt']:
    d=torch.load(base/name,map_location='cpu',weights_only=False)
    print(name, 'total_steps=', d.get('total_steps'), 'episodes=', d.get('episodes'), 'algorithm_train_step=', d.get('algorithm_train_step'), 'best_score=', d.get('best_model_score'), 'best_reward=', d.get('best_reward'))

