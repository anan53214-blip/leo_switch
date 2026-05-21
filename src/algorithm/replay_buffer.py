from collections import deque
import random
from typing import Dict

import numpy as np
import torch


class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, action_feature_dim, mask_dim, device="cpu"):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.mask_dim = int(mask_dim)
        self.device = torch.device(device)
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

    def _normalize_reward(self, reward) -> np.ndarray:
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.ndim == 0:
            return np.full(self.num_agents, float(reward_array), dtype=np.float32)
        if reward_array.shape != (self.num_agents,):
            raise ValueError(
                f"Expected scalar reward or shape ({self.num_agents},), got {reward_array.shape}"
            )
        return reward_array.copy()

    def add(self, obs, action_features, reward, next_obs, done, masks, next_masks):
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                np.asarray(action_features, dtype=np.float32).copy(),
                self._normalize_reward(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
                np.asarray(masks, dtype=bool).copy(),
                np.asarray(next_masks, dtype=bool).copy(),
            )
        )

    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        if len(self.buffer) < int(batch_size):
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}")
        batch = random.sample(self.buffer, int(batch_size))
        return {
            "obs": torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=self.device),
            "dones": torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device),
            "masks": torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool, device=self.device),
            "next_masks": torch.tensor(np.stack([item[6] for item in batch]), dtype=torch.bool, device=self.device),
        }
