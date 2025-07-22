import torch
import torch.nn as nn
from torch.distributions import Categorical
from src.Truss import TrussStructure
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPActorCritic(nn.Module):
    def __init__(self, truss: TrussStructure, obs_dim=256, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.truss = copy.deepcopy(truss)
        self.n_bars = len(self.truss.all_edges)
        self.n_nodes = self.truss.n_nodes

        # === Static parts of observation ===
        self.fixed_onehot = torch.zeros(self.n_nodes, dtype=torch.float32, device=device)
        for j in self.truss.fixed_nodes:
            self.fixed_onehot[j] = 1.0

        self.connectivity_flat = self.get_connectivity_matrix().flatten().to(device)

        # === Total input dimension ===
        # design + compliance (2*n_bars) + fixed (n_nodes) + force_node (n_nodes) +
        # force_direction (8 classes) + connectivity (n_nodes^2)
        self.raw_obs_dim = (
            2 * self.n_bars +         # design + compliance
            self.n_nodes +           # fixed node one-hot
            self.n_nodes +           # force node one-hot
            8 +                      # force direction one-hot
            self.n_nodes * self.n_nodes  # flattened connectivity
        )

        # === Observation encoder ===
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.raw_obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.Sigmoid(),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.Sigmoid(),
            nn.Linear(obs_dim, obs_dim)
        )

        # === Policy network ===
        self.policy_net = self._build_net(obs_dim, hidden_sizes, self.n_bars, final_activation=True)

        # === Value network ===
        self.value_net = self._build_net(obs_dim, hidden_sizes, 1, final_activation=False)

        self.mask_prob = 1e-9
        self.to(device)

    def get_connectivity_matrix(self):
        """
        Computes the adjacency matrix from self.truss.all_edges and self.truss.n_nodes.
        Returns a [n_nodes x n_nodes] torch.FloatTensor on the correct device.
        """
        conn = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.float32, device=device)
        for i, j in self.truss.all_edges:
            conn[i, j] = 1.0
            conn[j, i] = 1.0  # undirected connection
        return conn

    def _build_net(self, input_dim, hidden_sizes, output_dim, final_activation=False):
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.Sigmoid()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation:
            layers += [nn.LayerNorm(output_dim), nn.Sigmoid()]
        else:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def encode_obs(self, design_state, compliance, force_node_idx, force_direction_class):
        B = design_state.shape[0] # batch size

        # === Static encodings ===
        fixed_onehot = self.fixed_onehot.unsqueeze(0).repeat(B, 1)  # [B, n_nodes]
        conn_flat = self.connectivity_flat.unsqueeze(0).repeat(B, 1)  # [B, n_nodes^2]

        # === Force node one-hot ===
        force_node_onehot = torch.zeros((B, self.n_nodes), dtype=torch.float32, device=device)
        force_node_onehot[torch.arange(B), force_node_idx] = 1.0

        # === Force direction one-hot (8 directions)
        force_dir_onehot = torch.zeros((B, 8), dtype=torch.float32, device=device)
        force_dir_onehot[torch.arange(B), force_direction_class] = 1.0

        # === Bar features ===
        bar_feat = torch.cat([design_state, compliance], dim=-1)  # [B, 2*n_bars]

        # === Final observation ===
        obs = torch.cat([
            bar_feat,             # [B, 2*n_bars]
            fixed_onehot,         # [B, n_nodes]
            force_node_onehot,    # [B, n_nodes]
            force_dir_onehot,     # [B, 8]
            conn_flat             # [B, n_nodes^2]
        ], dim=-1)

        return self.encoder_mlp(obs)

    def act(self, design_state, compliance, force_node_idx, force_direction_class, mask=None, deterministic=False):
        obs = self.encode_obs(design_state, compliance, force_node_idx, force_direction_class)
        act_prob = self.policy_net(obs)
        if mask is not None:
            act_prob = mask * act_prob + mask * self.mask_prob
        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)
        action = torch.argmax(act_prob, dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, design_state, action, compliance, force_node_idx, force_direction_class, mask=None):
        obs = self.encode_obs(design_state, compliance, force_node_idx, force_direction_class)
        act_prob = self.policy_net(obs)
        if mask is not None:
            act_prob = mask * act_prob + mask * self.mask_prob
        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, value, entropy
