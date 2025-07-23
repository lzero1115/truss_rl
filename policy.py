import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool
from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

class TrussGNNEncoder(nn.Module):
    """
    GNN encoder for truss graphs. Processes HeteroData and outputs (probs, value).
    - probs: action probabilities for each bar (after sigmoid)
    - value: scalar value for the whole graph (after tanh)
    Joint node features: [x, y, is_fixed, is_loaded, cos(theta), sin(theta)] (6 features)
    """
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Joint encoder: [x, y, is_fixed, is_loaded, cos(theta), sin(theta)]
        self.joint_encoder = nn.Linear(6, hidden_dim)
        self.bar_encoder = nn.Linear(1 + 1 + 2, hidden_dim)         # [design_state, compliance, endpoint indices]
        convs = []
        for _ in range(num_layers):
            conv = HeteroConv({
                ('bar', 'connects', 'joint'): GATConv(hidden_dim, hidden_dim, add_self_loops=False),
                ('joint', 'rev_connects', 'bar'): GATConv(hidden_dim, hidden_dim, add_self_loops=False),
            }, aggr='sum')
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        # Actor head: outputs probability for each bar (sigmoid)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability directly
        )
        # Critic head: outputs a value for the whole graph, with tanh activation
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Ensure value is in [-1, 1]
        )

    def forward(self, data):
        # Joint node features: [x, y, is_fixed, is_loaded, cos(theta), sin(theta)]
        joint_features = data['joint'].x  # [n_joints, 6]
        x_dict = {
            'joint': self.joint_encoder(joint_features),
            'bar': self.bar_encoder(data['bar'].x)
        }
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        bar_feats = x_dict['bar']  # [total_bars_in_batch, hidden_dim]
        probs = self.actor_head(bar_feats).squeeze(-1)  # [total_bars_in_batch], in [0,1]
        batch = data['bar'].batch if hasattr(data['bar'], 'batch') else torch.zeros(bar_feats.size(0), dtype=torch.long, device=bar_feats.device)
        pooled = global_mean_pool(bar_feats, batch)
        value = self.critic_head(pooled).squeeze(-1)
        return probs, value

class TrussGNNActorCritic(nn.Module):
    """
    Actor-critic policy for truss RL using a GNN encoder.
    - The encoder returns (probs, value) directly.
    - The actor outputs probabilities (not logits) for each bar.
    - The critic outputs a value for the whole graph.
    """
    def __init__(self, hidden_dim=64, num_layers=3, num_actions=None, num_force_dirs=8, force_dir_embed_dim=4):
        super().__init__()
        self.encoder = TrussGNNEncoder(hidden_dim=hidden_dim, num_layers=num_layers)
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.mask_prob = 1E-9
        self.to(device)

    def forward(self, data):
        # Returns (probs, value)
        return self.encoder(data)

    def act(self, batch_graph, mask=None, deterministic=False):
        probs, value = self.forward(batch_graph)
        batch_size = batch_graph.num_graphs
        n_bars = self.num_actions
        # Reshape to [batch_size, n_bars]
        probs = probs.view(batch_size, n_bars)
        if mask is not None:
            mask = mask.view(batch_size, n_bars)
            probs = mask * probs + mask * self.mask_prob
        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, batch_graph, action, mask=None):
        probs, value = self.forward(batch_graph)
        batch_size = batch_graph.num_graphs
        n_bars = self.num_actions
        probs = probs.view(batch_size, n_bars)
        if mask is not None:
            mask = mask.view(batch_size, n_bars)
            probs = mask * probs + mask * self.mask_prob
        dist = Categorical(probs)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, value, entropy
