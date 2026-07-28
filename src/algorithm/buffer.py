
import torch
import numpy as np
from typing import Any, Dict, Optional, Generator


class MultiAgentRolloutBuffer:
    
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        global_state_dim: int,
        max_candidates: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.max_candidates = max_candidates
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.pos = 0
        self.full = False
        


        self.observations = np.zeros(
            (buffer_size, num_agents, obs_dim), dtype=np.float32
        )
        

        self.actions_discrete = np.zeros(
            (buffer_size, num_agents), dtype=np.int64
        )
        

        self.actions_continuous = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        

        self.candidate_masks = np.ones(
            (buffer_size, num_agents, max_candidates + 1), dtype=np.float32
        )
        

        self.rewards = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        

        self.dones = np.zeros(buffer_size, dtype=np.float32)
        

        self.log_probs = np.zeros(
            (buffer_size, num_agents), dtype=np.float32
        )
        


        self.global_states = np.zeros(
            (buffer_size, global_state_dim), dtype=np.float32
        )



        self.satellite_embeddings = None


        self.candidate_sat_ids = np.full(
            (buffer_size, num_agents, max_candidates),
            -1,
            dtype=np.int64,
        )


        self.values = np.zeros(buffer_size, dtype=np.float32)
        


        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.graph_snapshots = [None] * buffer_size
    
    def reset(self):
        for index in range(self.pos):
            self.graph_snapshots[index] = None
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        satellite_embeddings: Optional[np.ndarray],
        actions_discrete: np.ndarray,
        actions_continuous: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        value: float,
        log_probs: np.ndarray,
        candidate_masks: Optional[np.ndarray] = None,
        candidate_sat_ids: Optional[np.ndarray] = None,
        graph_snapshot: Optional[Any] = None,
    ):
        self.observations[self.pos] = obs
        self.global_states[self.pos] = global_state

        if satellite_embeddings is not None:
            if self.satellite_embeddings is None:
                num_satellites = satellite_embeddings.shape[0]
                embed_dim = satellite_embeddings.shape[1]
                self.satellite_embeddings = np.zeros(
                    (self.buffer_size, num_satellites, embed_dim), dtype=np.float32
                )
            self.satellite_embeddings[self.pos] = satellite_embeddings

        self.actions_discrete[self.pos] = actions_discrete
        self.actions_continuous[self.pos] = actions_continuous
        

        if np.isscalar(rewards):
            self.rewards[self.pos] = np.full(self.num_agents, rewards)
        else:
            self.rewards[self.pos] = rewards
        
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_probs
        
        if candidate_masks is not None:
            self.candidate_masks[self.pos] = candidate_masks

        if candidate_sat_ids is not None:
            self.candidate_sat_ids[self.pos] = candidate_sat_ids

        self.graph_snapshots[self.pos] = graph_snapshot
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool
    ):
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                # dones[step] belongs to transition (s_t, a_t, r_t).
                # Using dones[step + 1] leaks advantages across episode resets.
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            

            delta = (
                self.rewards[step]  # (num_agents,)
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos, np.newaxis]
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Dict[str, torch.Tensor], None, None]:

        total_samples = self.pos * self.num_agents
        

        flat_obs = self.observations[:self.pos].reshape(-1, self.obs_dim)
        

        flat_actions_discrete = self.actions_discrete[:self.pos].reshape(-1)
        flat_actions_continuous = self.actions_continuous[:self.pos].reshape(-1)
        

        flat_masks = self.candidate_masks[:self.pos].reshape(
            -1, self.max_candidates + 1
        )
        

        flat_log_probs = self.log_probs[:self.pos].reshape(-1)
        

        flat_advantages = self.advantages[:self.pos].reshape(-1)

        flat_returns = np.repeat(self.returns[:self.pos].mean(axis=1), self.num_agents)
        flat_values = np.repeat(self.values[:self.pos], self.num_agents)
        

        flat_global_states = np.repeat(
            self.global_states[:self.pos], self.num_agents, axis=0
        )


        flat_satellite_embeddings = None
        if self.satellite_embeddings is not None:
            flat_satellite_embeddings = np.repeat(
                self.satellite_embeddings[:self.pos], self.num_agents, axis=0
            )


        flat_candidate_sat_ids = self.candidate_sat_ids[:self.pos].reshape(
            -1,
            self.max_candidates,
        )
        flat_time_indices = np.repeat(
            np.arange(self.pos, dtype=np.int64),
            self.num_agents,
        )
        flat_agent_indices = np.tile(
            np.arange(self.num_agents, dtype=np.int64),
            self.pos,
        )


        if shuffle:
            indices = np.random.permutation(total_samples)
        else:
            indices = np.arange(total_samples)
        

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            yield {
                'observations': torch.tensor(
                    flat_obs[batch_indices], device=self.device, dtype=torch.float32
                ),
                'global_states': torch.tensor(
                    flat_global_states[batch_indices], device=self.device, dtype=torch.float32
                ),
                'satellite_embeddings': None if flat_satellite_embeddings is None else torch.tensor(
                    flat_satellite_embeddings[batch_indices], device=self.device, dtype=torch.float32
                ),
                'actions_discrete': torch.tensor(
                    flat_actions_discrete[batch_indices], device=self.device, dtype=torch.long
                ),
                'actions_continuous': torch.tensor(
                    flat_actions_continuous[batch_indices], device=self.device, dtype=torch.float32
                ),
                'candidate_masks': torch.tensor(
                    flat_masks[batch_indices], device=self.device, dtype=torch.float32
                ),
                'candidate_sat_ids': torch.tensor(
                    flat_candidate_sat_ids[batch_indices], device=self.device, dtype=torch.long
                ),
                'time_indices': torch.tensor(
                    flat_time_indices[batch_indices], device=self.device, dtype=torch.long
                ),
                'agent_indices': torch.tensor(
                    flat_agent_indices[batch_indices], device=self.device, dtype=torch.long
                ),
                'old_log_probs': torch.tensor(
                    flat_log_probs[batch_indices], device=self.device, dtype=torch.float32
                ),
                'advantages': torch.tensor(
                    flat_advantages[batch_indices], device=self.device, dtype=torch.float32
                ),
                'returns': torch.tensor(
                    flat_returns[batch_indices], device=self.device, dtype=torch.float32
                ),
                'values': torch.tensor(
                    flat_values[batch_indices], device=self.device, dtype=torch.float32
                )
            }
