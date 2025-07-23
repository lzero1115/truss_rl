import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from torch.cuda import Stream
from torch_geometric.data import Dataset, Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

class TrussRolloutDataset(Dataset):
    """Dataset class for truss training data for GNN-based workflow, inherits from torch_geometric.data.Dataset"""
    def __init__(self, transform=None, pre_transform=None, batch_size=100):
        super().__init__(None, transform, pre_transform)
        self._indices = None
        self.batch_size = batch_size
        self.nstates = 0
        self.nstreams = 5
        self.tos = Stream(device)
        #self.graph_data = []  # List of HeteroData objects

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        if idx < len(self) - 1:
            return self.batch_data_from_to(idx * self.batch_size, (idx + 1) * self.batch_size)
        else:
            return self.batch_data_from_to(idx * self.batch_size, self.nstates)

    def __len__(self):
        return int(np.ceil(len(self.graph_data) / self.batch_size)) if hasattr(self, "graph_data") else 0

    def add_graph_data(self, graph_data):
        self.graph_data = graph_data
        self.nstates = len(graph_data)

    def batch_data_from_to(self, idx_start, idx_end):
        # Build batched graph from stored HeteroData objects
        datalist = self.graph_data[idx_start:idx_end]
        batch_graph = Batch.from_data_list(datalist)
        with torch.cuda.stream(self.tos):
            return (
                batch_graph,
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
    """Rollout buffer for truss optimization, GNN-compatible. Stores HeteroData objects directly."""
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
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increase = (1.0 - self.per_beta) / per_num_anneal
        self.num_step_anneal = per_num_anneal
        self.clear_replay_buffer()
        self.reset_curriculum(0)

    def update_per_beta(self):
        if self.num_step_anneal > 0:
            self.num_step_anneal -= 1
            self.per_beta += self.per_beta_increase

    def modify_entropy(self):
        increasable = self.entropy_weights[self.num_failed] < self.max_entropy_weight
        self.entropy_weights[self.num_failed[increasable]] += self.entropy_weight_increase
        self.entropy_weights[self.num_success] = self.base_entropy_weight

    def reset_curriculum(self, n_curriculums: int):
        self.num_failed = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.num_success = torch.zeros(n_curriculums, dtype=intType, device=device)
        self.curriculum_visited = torch.zeros(n_curriculums, device=device, dtype=torch.bool)
        self.curriculum_cumsurloss = torch.zeros(n_curriculums, device=device, dtype=floatType)
        self.curriculum_rank = torch.arange(n_curriculums, dtype=intType, device=device)
        self.entropy_weights = self.base_entropy_weight * torch.ones((n_curriculums), dtype=floatType, device=device)

    def clear_replay_buffer(self, num_env: int = 0):
        self.actions = [[] for _ in range(num_env)]
        self.logprobs = [[] for _ in range(num_env)]
        self.state_values = [[] for _ in range(num_env)]
        self.masks = [[] for _ in range(num_env)]
        self.graph_data = [[] for _ in range(num_env)]  # List of HeteroData objects per env
        self.next_design_states = [[] for _ in range(num_env)]
        self.next_stability = [[] for _ in range(num_env)]
        self.rewards_per_step = [[] for _ in range(num_env)]
        self.optimal_compliances = [[] for _ in range(num_env)]
        self.num_envs = num_env
        self.rewards = None
        self.weights = None

    def sample_curriculum(self, num_rollouts):
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
        env_inds = []
        rewards = []
        for env_id in range(self.num_envs):
            data = self.rewards_per_step[env_id]
            if len(data) == 0:
                rewards.append(0)
            else:
                if data[-1] == 0:
                    env_inds.append(env_id)
                rewards.append(data[-1])
        return np.array(env_inds), np.array(rewards)

    def add(self, name: str, vals: List, env_inds: np.ndarray):
        for ival, ienv in enumerate(env_inds):
            if torch.is_tensor(vals[ival]):
                val = vals[ival].detach().cpu()
            else:
                val = vals[ival]
            if name == 'graph_data':
                self.graph_data[ienv].append(val)  # val should be a HeteroData object
            else:
                self.__dict__[name][ienv].append(val)

    def build_dataset(self, batch_size: int = 512):
        """Build dataset for training with optimized GPU usage and GNN compatibility"""
        dataset = TrussRolloutDataset(batch_size=batch_size)
        actions = []
        logprobs = []
        state_values = []
        masks = []
        graph_data = []
        rewards = []
        curriculum_id = []
        for ienv in range(self.num_envs):
            discounted_reward = self.rewards[ienv]
            for istep in range(len(self.actions[ienv]) - 1, -1, -1):
                rewards.append(discounted_reward)
                graph_data.append(self.graph_data[ienv][istep])
                actions.append(self.actions[ienv][istep])
                logprobs.append(self.logprobs[ienv][istep])
                state_values.append(self.state_values[ienv][istep])
                masks.append(self.masks[ienv][istep])
                curriculum_id.append(self.curriculum_inds[ienv])
                discounted_reward = self.gamma * discounted_reward
        if len(masks) > 0:
            actions_tensor = torch.tensor(actions, dtype=intType, device="cpu")
            masks_tensor = torch.vstack(masks)
            logprobs_tensor = torch.tensor(logprobs, dtype=floatType, device="cpu")
            rewards_tensor = torch.tensor(rewards, dtype=floatType, device="cpu")
            curriculum_id_tensor = torch.hstack(curriculum_id)
            dataset.actions = actions_tensor.detach()
            dataset.masks = masks_tensor
            dataset.logprobs = logprobs_tensor.detach()
            dataset.rewards = rewards_tensor.detach()
            adv = dataset.rewards - torch.tensor(state_values, dtype=floatType, device="cpu")
            dataset.advantages = ((adv-adv.mean())/(adv.std(unbiased=False)+1e-7))
            dataset.add_graph_data(graph_data)
            dataset.curriculum_id = curriculum_id_tensor
            dataset.weights = torch.pow(1 + self.curriculum_rank[dataset.curriculum_id],
                                        self.per_alpha - self.per_beta).cpu()
            dataset.weights /= dataset.weights.max()
            dataset.entropy_weights = self.entropy_weights[dataset.curriculum_id].cpu()
        return dataset
