import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from time import perf_counter
import copy
import pickle
from buffer import TrussRolloutBuffer
from env import TrussEnv
from policy import MLPActorCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class TrussPPO:
    def __init__(self, truss, settings: Dict):
        self.truss = copy.deepcopy(truss)
        self.n_bars = len(self.truss.all_edges)
        self.free_nodes = self.truss.n_nodes - len(self.truss.fixed_nodes)


        ppo_config = settings.get("ppo", {})
        obs_dim = ppo_config.get("obs_dim", 256)
        hidden_sizes = ppo_config.get("hidden_sizes", (512, 512, 256))

        self.policy = MLPActorCritic(
            n_bars=self.n_bars,
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes
        ).to(device)

        self.policy_old = MLPActorCritic(
            n_bars=self.n_bars,
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes
        ).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.eps_clip = ppo_config.get("eps_clip", 0.2)
        self.lr_actor = ppo_config.get("lr_actor", 2e-3)
        self.betas_actor = tuple(ppo_config.get("betas_actor", [0.95, 0.999]))

        self.lr_milestones = ppo_config.get("lr_milestones", [100, 300])
        self.base_entropy_weight = ppo_config.get("base_entropy_weight", 0.005)
        self.entropy_weight_increase = ppo_config.get("entropy_weight_increase", 0.001)
        self.max_entropy_weight = ppo_config.get("max_entropy_weight", 0.01)
        self.per_alpha = ppo_config.get("per_alpha", 0.8)
        self.per_beta = ppo_config.get("per_beta", 0.1)
        self.per_num_anneal = ppo_config.get("per_num_anneal", 500)
        self.gamma = ppo_config.get("gamma", 0.95)

        self.optimizer_params = {
            'params': self.policy.parameters(),
            'lr': self.lr_actor,
            'weight_decay': 0,
            'betas': self.betas_actor
        }

        self.optimizer = torch.optim.Adam([self.optimizer_params])

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.lr_milestones
        )

        self.buffer = TrussRolloutBuffer(
            gamma=self.gamma,
            base_entropy_weight=self.base_entropy_weight,
            entropy_weight_increase=self.entropy_weight_increase,
            max_entropy_weight=self.max_entropy_weight,
            per_alpha=self.per_alpha,
            per_beta=self.per_beta,
            per_num_anneal=self.per_num_anneal
        )

        self.episode = 0
        self.deterministic = False
        self.MseLoss = nn.MSELoss(reduction='none')
        self.accuracy_of_sample_curriculum = 0
        self.accuracy_of_entire_curriculum = 0

    def select_action(self, design_states, compliances, force_nodes, force_dirs, masks: np.ndarray, env_inds: np.ndarray):

        env_masks = torch.tensor(masks, dtype=floatType, device=device)
        design_states_tensor = torch.stack([torch.tensor(ds, dtype=floatType) for ds in design_states]).to(device)

        compliances_tensor = torch.tensor(np.array(compliances), dtype=floatType, device=device)

        force_nodes_tensor = torch.tensor(force_nodes, dtype=torch.long, device=device)

        force_dirs_tensor = torch.tensor(force_dirs, dtype=torch.long, device=device)


        self.policy_old.eval()
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(design_states_tensor,
                                                                    compliances_tensor,
                                                                    force_nodes_tensor,
                                                                    force_dirs_tensor,
                                                                    env_masks,
                                                                    self.deterministic)

        self.buffer.add('design_states', design_states, env_inds)
        self.buffer.add('actions', action, env_inds)
        self.buffer.add('logprobs', action_logprob, env_inds)
        self.buffer.add("state_values", state_val, env_inds)
        self.buffer.add("masks", env_masks.cpu(), env_inds)

        self.buffer.add("compliances", compliances.tolist(), env_inds)
        self.buffer.add("force_directions", force_dirs.tolist(), env_inds)
        self.buffer.add("force_node_indices", force_nodes.tolist(), env_inds)

        return action.cpu().numpy()

    def update_policy(self, batch_size: int = None, update_iter: int = 5):

        dataset = self.buffer.build_dataset(batch_size=batch_size)

        self.policy.train(True)
        self.optimizer = torch.optim.Adam([self.optimizer_params])

        total_loss = 0
        total_val_loss = 0
        total_sur_loss = 0
        total_entropy = 0
        loss = []
        entropy = []

        for epoch_index in range(update_iter):
            loss = 0
            epoch_sur_loss = 0
            epoch_val_loss = 0
            entropy = 0
            self.optimizer.zero_grad()

            for i, batch in enumerate(dataset):
                if batch is not None and len(batch) > 0:
                    loss_batch, sur_loss_batch, val_loss_batch, entropy_batch = self.train_one_epoch(
                        *batch, dataset.nstates, first_batch=(i == 0)
                    )
                    epoch_sur_loss += sur_loss_batch
                    epoch_val_loss += val_loss_batch
                    loss += loss_batch
                    entropy += entropy_batch

            self.optimizer.step()

            total_loss += loss
            total_sur_loss += epoch_sur_loss
            total_val_loss += epoch_val_loss
            total_entropy += entropy

        sampled, n = torch.unique(dataset.curriculum_id.to(device), return_counts=True)
        self.buffer.curriculum_cumsurloss[sampled] /= (n * update_iter)
        self.buffer.curriculum_rank = torch.argsort(torch.argsort(self.buffer.curriculum_cumsurloss, descending=True))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear_replay_buffer()

        # ✅ Step the scheduler and update internal LR tracking via optimizer_params
        self.scheduler.step()
        self.optimizer_params['lr'], *_ = self.scheduler.get_last_lr()
        #print(f"[INFO] Learning rate after scheduler step: {self.optimizer_params['lr']:.6e}")

        return loss, epoch_val_loss, epoch_sur_loss, entropy


    def train_one_epoch(self,
                        old_design_states,
                        old_compliances,
                        old_force_node_indices,
                        old_force_dirs,
                        old_actions,
                        old_masks,
                        old_logprobs,
                        rewards,
                        advantages,
                        weights,
                        entropy_weights,
                        curriculum_id,
                        nstates,
                        first_batch):

        design_states = old_design_states
        compliances = old_compliances  # Use the compliance data

        batch_percentage = float(len(old_actions)) / nstates
        logprobs, state_values, dist_entropy = self.policy.evaluate(design_states,
                                                                    old_actions,
                                                                    compliances,
                                                                    old_force_node_indices,
                                                                    old_force_dirs,
                                                                    old_masks)

        state_values = torch.squeeze(state_values).reshape(-1)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        value_loss = self.MseLoss(state_values, rewards)
        surrogate_loss = -torch.min(surr1, surr2)

        loss = (surrogate_loss * weights).mean() + 0.5 * (value_loss * weights).mean() - (
                entropy_weights * dist_entropy * weights).mean()
        self.buffer.curriculum_visited[curriculum_id] = True
        self.buffer.curriculum_cumsurloss = self.buffer.curriculum_cumsurloss.scatter_reduce(0, curriculum_id,
                                                                                             torch.abs(
                                                                                                 surrogate_loss.detach()),
                                                                                             reduce='sum',
                                                                                             include_self=not first_batch)

        entropy_mean = dist_entropy.mean().item()
        

        loss *= batch_percentage
        loss_mean = loss.item()
        loss.mean().backward()

        return loss_mean, batch_percentage * surrogate_loss.mean().item(), batch_percentage * value_loss.mean().item(), batch_percentage * entropy_mean

    def save(self, checkpoint_path: str, settings: Dict):
        d = {
            'settings': settings,
            'policy_state_dict': self.policy_old.state_dict(),

        }

        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


