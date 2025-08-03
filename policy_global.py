import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.Truss import TrussStructure
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.long  # Changed from torch.int32 to torch.long for PyTorch indexing compatibility


class TrussEncoder(nn.Module):
    """Global encoder for truss structure observations - treats entire truss as unified system"""
    
    def __init__(self, truss: TrussStructure, 
                 obs_dim=256):
        super().__init__()
        
        self.truss = truss
        self.n_nodes = truss.n_nodes
        self.n_bars = len(truss.all_edges)
        self.fixed_nodes = set(truss.fixed_nodes)
        
        # Pre-compute static features (no encoding needed)
        self.node_positions = torch.tensor(truss.nodes[:, :2], dtype=floatType, device=device)  # [n_nodes, 2]
        self.node_fixed = torch.tensor([1.0 if i in self.fixed_nodes else 0.0 for i in range(self.n_nodes)], 
                                      dtype=floatType, device=device)  # [n_nodes]
        
        # Connectivity matrix (static)
        self.connectivity_matrix = torch.zeros(len(truss.all_edges), truss.n_nodes, dtype=floatType, device=device)
        for i, (node1, node2) in enumerate(truss.all_edges):
            self.connectivity_matrix[i, node1] = 1.0
            self.connectivity_matrix[i, node2] = 1.0
        
        # Global encoder to process entire truss state
        total_input_dim = (
            self.n_bars +                    # design_state
            self.n_bars +                    # compliance
            self.n_nodes +                   # force_node_onehot
            2 +                              # force_direction [cos, sin]
            self.n_nodes * 2 +              # node_positions
            self.n_nodes +                   # node_fixed
            self.n_bars * self.n_nodes      # connectivity
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(total_input_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim)
        )
        
        self.to(device)
    
    def encode_observation(self, design_state, compliance, force_node, force_dir):
        """Encode entire truss as global observation"""
        batch_size = design_state.shape[0]
        
        # Ensure inputs are on correct device
        device = next(self.parameters()).device
        design_state = design_state.to(device)
        compliance = compliance.to(device)
        force_node = force_node.to(device)
        force_dir = force_dir.to(device)
        
        # Normalize compliance values
        compliance_sum = torch.sum(compliance, dim=1, keepdim=True) + 1e-8
        normalized_compliance = compliance / compliance_sum
        
        # Convert force direction to [cos(theta), sin(theta)]
        directions = torch.linspace(0, 2 * torch.pi, 8, device=device)
        angles = directions[force_dir]  # [B]
        force_vectors = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=-1)  # [B, 2]
        
        # Create force node one-hot encoding
        force_node_onehot = torch.zeros(batch_size, self.n_nodes, dtype=floatType, device=device)
        force_node_onehot.scatter_(1, force_node.unsqueeze(1), 1.0)  # [B, n_nodes]
        
        # Static features (same for all batches)
        node_positions = self.node_positions.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_nodes, 2]
        node_positions = node_positions.flatten(1)  # [B, n_nodes * 2]
        
        node_fixed = self.node_fixed.unsqueeze(0).expand(batch_size, -1)  # [B, n_nodes]
        
        connectivity = self.connectivity_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_bars, n_nodes]
        connectivity = connectivity.flatten(1)  # [B, n_bars * n_nodes]
        
        # Combine all features into global truss state
        global_features = torch.cat([
            design_state,                    # [B, n_bars]
            normalized_compliance,           # [B, n_bars]
            force_node_onehot,              # [B, n_nodes]
            force_vectors,                   # [B, 2]
            node_positions,                 # [B, n_nodes * 2]
            node_fixed,                     # [B, n_nodes]
            connectivity                    # [B, n_bars * n_nodes]
        ], dim=-1)
        
        # Global encoding
        encoded_obs = self.global_encoder(global_features)
        
        return encoded_obs


class MLPActorCritic(nn.Module):
    def __init__(self, truss: TrussStructure, 
                 obs_dim=256,
                 hidden_sizes=(512, 512, 256)):
        super().__init__()
        
        # Initialize truss encoder
        self.truss_encoder = TrussEncoder(truss, obs_dim=obs_dim)
        
        # Action dimension (number of bars)
        action_dim = len(truss.all_edges)
        
        # === Policy Network ===
        policy_layers = []
        prev_dim = obs_dim
        for hidden_size in hidden_sizes:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ])
            prev_dim = hidden_size
        policy_layers.append(nn.Linear(prev_dim, action_dim))
        policy_layers.append(nn.Softmax(dim=-1))

        self.policy_net = nn.Sequential(*policy_layers)

        # === Value Network ===
        value_layers = []
        prev_dim = obs_dim
        for hidden_size in hidden_sizes:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ])
            prev_dim = hidden_size
        value_layers.append(nn.Linear(prev_dim, 1))
        value_layers.append(nn.Tanh())

        self.value_net = nn.Sequential(*value_layers)

        self.mask_prob = 1E-9
        self.to(device)

    def encode_obs(self, design_state, compliance, force_node, force_dir):
        """Encode observation using truss encoder"""
        return self.truss_encoder.encode_observation(design_state, compliance, force_node, force_dir)

    def act(self, design_state, compliance, force_node, force_dir, mask=None, deterministic=False):
        """Get action, log probability, and value"""
        obs = self.encode_obs(design_state, compliance, force_node, force_dir)
        
        act_prob = self.policy_net(obs)

        if mask is not None:
            # Convert mask to tensor if it's a numpy array
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32, device=act_prob.device)
            act_prob = mask * act_prob + mask * self.mask_prob

        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)

        if deterministic:
            action = torch.argmax(act_prob, dim=-1)
        else:
            action = dist.sample()

        logprob = dist.log_prob(action)

        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, design_state, action, compliance, force_node, force_dir, mask=None):
        """Evaluate action for training"""
        obs = self.encode_obs(design_state, compliance, force_node, force_dir)

        act_prob = self.policy_net(obs)

        if mask is not None:
            # Convert mask to tensor if it's a numpy array
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32, device=act_prob.device)
            act_prob = mask * act_prob + mask * self.mask_prob

        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return logprob, value, entropy 