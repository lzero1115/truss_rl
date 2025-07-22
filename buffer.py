import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from torch.cuda import Stream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class TrussRolloutDataset:
    """Dataset class for truss training data following assembly project structure"""

    def __init__(self, transform=None, pre_transform=None, batch_size=100):
        self._indices = None
        self.batch_size = batch_size
        self.nstates = 0
        self.nstreams = 5
        self.tos = Stream(device)

    def __getitem__(self, idx):
        # Raise IndexError for indices beyond dataset length to stop iteration  
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
            
        if idx < len(self) - 1:
            return self.batch_data_from_to(idx * self.batch_size, (idx + 1) * self.batch_size)
        else:
            return self.batch_data_from_to(idx * self.batch_size, self.nstates)

    def __len__(self):
        if hasattr(self, "design_states"):
            length = int(np.ceil(len(self.design_states) / self.batch_size))
            return length
        else:
            return 0

    def add_states(self, design_states): # observation space encoding
        """Add observations to dataset"""
        self.design_states = design_states
        self.nstates = len(design_states)

    def batch_data_from_to(self, idx_start, idx_end):
        """Get batch data from start to end index with CUDA streaming"""

        with torch.cuda.stream(self.tos):

            compliance_batch = self.compliances[idx_start:idx_end].to(device, non_blocking=True)
            force_directions_batch = self.force_directions[idx_start:idx_end].to(device, non_blocking=True)
            force_node_indices_batch = self.force_node_indices[idx_start:idx_end].to(device, non_blocking=True)

            return (
                self.design_states[idx_start:idx_end].to(device,non_blocking=True),
                compliance_batch,
                force_node_indices_batch,
                force_directions_batch,
                self.actions[idx_start:idx_end].to(device, non_blocking=True),
                self.masks[idx_start:idx_end].to(device, non_blocking=True),
                self.logprobs[idx_start:idx_end].to(device, non_blocking=True),
                self.rewards[idx_start:idx_end].to(device, non_blocking=True),
                self.advantages[idx_start:idx_end].to(device, non_blocking=True),
                self.weights[idx_start:idx_end].to(device, non_blocking=True),
                self.entropy_weights[idx_start:idx_end].to(device, non_blocking=True),
                self.curriculum_id[idx_start:idx_end].to(device, non_blocking=True)
            )


class TrussRolloutBuffer:
    """Rollout buffer for truss optimization following assembly project structure exactly"""

    def __init__(self,
                 gamma: float = 0.99,
                 base_entropy_weight: float = 0.01,
                 entropy_weight_increase: float = 0.001,
                 max_entropy_weight: float = 0.05,
                 per_alpha: float = 0.8,
                 per_beta: float = 0.1,
                 per_num_anneal: int = 500):

        self.gamma = gamma
        self.base_entropy_weight = base_entropy_weight
        self.entropy_weight_increase = entropy_weight_increase
        self.max_entropy_weight = max_entropy_weight

        # Prioritized experience replay
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increase = (1.0 - self.per_beta) / per_num_anneal
        self.num_step_anneal = per_num_anneal

        self.clear_replay_buffer()
        self.reset_curriculum(0)

    def update_per_beta(self):
        """Update PER beta parameter"""
        if self.num_step_anneal > 0:
            self.num_step_anneal -= 1
            self.per_beta += self.per_beta_increase

    def modify_entropy(self):
        """
        🎯 RESTORED: Difficulty-based adaptive entropy for curriculum learning
        
        ADAPTIVE BEHAVIOR:
        - Failed curriculum items → increase entropy → more exploration for hard cases
        - Successful curriculum items → reset to base entropy → less exploration for easy cases
        - This creates natural curriculum difficulty adaptation
        """
        # Increase entropy for failed curriculum items (up to max limit)
        increasable = self.entropy_weights[self.num_failed] < self.max_entropy_weight
        self.entropy_weights[self.num_failed[increasable]] += self.entropy_weight_increase
        
        # Reset entropy for successful curriculum items to base level
        self.entropy_weights[self.num_success] = self.base_entropy_weight

    def reset_curriculum(self, n_curriculums: int):
        """Reset curriculum tracking - exact same as assembly project"""
        self.num_failed = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.num_success = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.curriculum_visited = torch.zeros(n_curriculums, device=device, dtype=torch.bool)
        self.curriculum_cumsurloss = torch.zeros(n_curriculums, device=device, dtype=floatType)
        self.curriculum_rank = torch.arange(n_curriculums, dtype=intType, device=device)
        self.entropy_weights = self.base_entropy_weight * torch.ones((n_curriculums), dtype=floatType, device=device)

    def clear_replay_buffer(self, num_env: int = 0):
        """Clear replay buffer - simplified to remove force nodes and directions"""
        self.actions = [[] for _ in range(num_env)]
        self.logprobs = [[] for _ in range(num_env)]
        self.state_values = [[] for _ in range(num_env)]
        self.masks = [[] for _ in range(num_env)]

        # Truss-specific equivalents to assembly project fields
        self.design_states = [[] for _ in range(num_env)]  # equivalent to part_states
        self.next_design_states = [[] for _ in range(num_env)]  # equivalent to next_states
        self.next_stability = [[] for _ in range(num_env)]  # equivalent to next_stability
        self.rewards_per_step = [[] for _ in range(num_env)]  # equivalent to rewards_per_step

        # Additional truss-specific data (removed force_nodes and force_dirs)
        self.optimal_compliances = [[] for _ in range(num_env)]
        self.compliances = [[] for _ in range(num_env)]
        self.force_directions = [[] for _ in range(num_env)]
        self.force_node_indices = [[] for _ in range(num_env)]

        self.num_envs = num_env
        self.rewards = None  # one reward for each env
        self.weights = None  # one weight for each env


    # def truncate(self, env_id: int, step_id: int):
    #     """Truncate episode data at given step - exact same as assembly project"""
    #     names = ["actions", "states", "logprobs", "state_values", "masks",
    #              "design_states", "next_design_states", "next_stability", "rewards_per_step",
    #              "force_nodes", "force_dirs", "optimal_compliances", "compliances"]
    #     for name in names:
    #         self.__dict__[name][env_id] = self.__dict__[name][env_id][:step_id]

    def sample_curriculum(self, num_rollouts):
        """Sample curriculum indices - exact same logic as assembly project"""
        visited = self.curriculum_visited
        rank = self.curriculum_rank
        sample_new_curriculum = (visited == False).sum() > num_rollouts

        if sample_new_curriculum:
            sample_weights = torch.zeros_like(visited).to(floatType)
            sample_weights[visited == False] = 1.0 / (visited == False).sum()
            curriculum_inds = torch.multinomial(sample_weights, num_rollouts, False)
        else:
            inds = torch.arange(1, rank.shape[0] + 1, device=device, dtype=floatType)
            sample_weights = torch.pow(inds, -self.per_alpha) / torch.pow(inds, -self.per_alpha).sum()
            curriculum_inds = torch.multinomial(sample_weights[rank], num_rollouts, True)

        return curriculum_inds.cpu(), sample_new_curriculum

    def get_valid_env_inds(self):
        """Get valid environment indices for the next step. Simple filtering based on rewards."""
        env_inds = []
        rewards = []
        for env_id in range(self.num_envs):
            data = self.rewards_per_step[env_id]
            if len(data) == 0:
                rewards.append(0)
            else:
                if data[-1] == 0:  # Still running (reward 0 means continue)
                    env_inds.append(env_id)
                rewards.append(data[-1])
        return np.array(env_inds), np.array(rewards)

    def add(self, name: str, vals: List, env_inds: np.ndarray):

        for ival, ienv in enumerate(env_inds):
            if torch.is_tensor(vals[ival]):
                val = vals[ival].detach().cpu()
            else:
                val = vals[ival]
            self.__dict__[name][ienv].append(val)

    def build_dataset(self, batch_size: int = 512) :
        """Build dataset for training with optimized GPU usage"""
        dataset = TrussRolloutDataset(batch_size=batch_size)
        actions = []
        logprobs = []
        state_values = []
        masks = []
        design_states = []
        compliances = []
        force_node_indices = []
        force_directions = []
        rewards = []
        curriculum_id = []

        for ienv in range(self.num_envs):
            discounted_reward = self.rewards[ienv]

            for istep in range(len(self.actions[ienv]) - 1, -1, -1):
                # Collect dataset
                rewards.append(discounted_reward)
                design_states.append(self.design_states[ienv][istep])
                # Collect compliance data if available

                compliances.append(self.compliances[ienv][istep])
                force_node_indices.append(self.force_node_indices[ienv][istep])
                force_directions.append(self.force_directions[ienv][istep])

                actions.append(self.actions[ienv][istep])
                logprobs.append(self.logprobs[ienv][istep])
                state_values.append(self.state_values[ienv][istep])
                masks.append(self.masks[ienv][istep])
                curriculum_id.append(self.curriculum_inds[ienv])
                discounted_reward = self.gamma * discounted_reward

        if len(masks) > 0:
            # Convert to tensors with optimized GPU transfers
            state_values_tensor = torch.tensor(state_values, dtype=floatType, device="cpu")
            actions_tensor = torch.tensor(actions, dtype=intType, device="cpu")
            masks_tensor = torch.vstack(masks)
            logprobs_tensor = torch.tensor(logprobs, dtype=floatType, device="cpu")
            rewards_tensor = torch.tensor(rewards, dtype=floatType, device="cpu")
            curriculum_id_tensor = torch.hstack(curriculum_id)
            design_states_tensor = torch.stack([torch.tensor(ds, dtype=floatType) for ds in design_states])

            compliances_tensor = torch.tensor(compliances, dtype=floatType, device="cpu")
            force_directions_tensor = torch.tensor(force_directions, dtype=intType, device="cpu")
            force_node_indices_tensor = torch.tensor(force_node_indices, dtype=intType, device="cpu")

            # Dataset tensors
            dataset.actions = actions_tensor.detach()
            dataset.masks = masks_tensor
            dataset.logprobs = logprobs_tensor.detach()
            dataset.rewards = rewards_tensor.detach()

            dataset.compliances = compliances_tensor.detach()
            dataset.force_directions = force_directions_tensor.detach()
            dataset.force_node_indices = force_node_indices_tensor.detach()

            adv = dataset.rewards - state_values_tensor.detach()
            dataset.advantages = ((adv-adv.mean())/(adv.std(unbiased=False)+1e-7))

            # Add states (tensor, will be processed during training)
            dataset.add_states(design_states_tensor.detach())
            dataset.curriculum_id = curriculum_id_tensor

            # Optimize weights calculation
            dataset.weights = torch.pow(1 + self.curriculum_rank[dataset.curriculum_id],
                                        self.per_alpha - self.per_beta).cpu()
            dataset.weights /= dataset.weights.max()
            dataset.entropy_weights = self.entropy_weights[dataset.curriculum_id].cpu()

        return dataset
