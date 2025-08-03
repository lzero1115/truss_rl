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
    """Element-based encoder for truss structure observations with dynamic encoding and static raw features"""
    
    def __init__(self, truss: TrussStructure, 
                 bar_embed_dim=32,
                 force_embed_dim=8,
                 obs_dim=256):
        super().__init__()
        
        self.truss = truss
        self.n_nodes = truss.n_nodes
        self.n_bars = len(truss.all_edges)
        self.fixed_nodes = set(truss.fixed_nodes)
        
        # Bar encoder (dynamic features: design_state + compliance)
        self.bar_encoder = nn.Sequential(
            nn.Linear(2, bar_embed_dim),  # [design_state, compliance]
            nn.LayerNorm(bar_embed_dim),
            nn.ReLU(),
            nn.Linear(bar_embed_dim, bar_embed_dim)
        )
        
        # Force direction encoder (dynamic feature)
        self.force_encoder = nn.Sequential(
            nn.Linear(2, force_embed_dim),  # [cos(theta), sin(theta)]
            nn.LayerNorm(force_embed_dim),
            nn.ReLU(),
            nn.Linear(force_embed_dim, force_embed_dim)
        )
        
        # Pre-compute static features (no encoding needed)
        self.node_positions = torch.tensor(truss.nodes[:, :2], dtype=floatType, device=device)  # [n_nodes, 2]
        self.node_fixed = torch.tensor([1.0 if i in self.fixed_nodes else 0.0 for i in range(self.n_nodes)], 
                                      dtype=floatType, device=device)  # [n_nodes]
        
        # Connectivity matrix (static)
        self.connectivity_matrix = torch.zeros(len(truss.all_edges), truss.n_nodes, dtype=floatType, device=device)
        for i, (node1, node2) in enumerate(truss.all_edges):
            self.connectivity_matrix[i, node1] = 1.0
            self.connectivity_matrix[i, node2] = 1.0
        
        # Global encoder to combine encoded dynamic features + raw static features
        total_bar_dim = bar_embed_dim * self.n_bars
        total_force_dim = force_embed_dim
        total_static_dim = self.n_nodes * 2 + self.n_nodes + self.n_bars * self.n_nodes  # positions + fixed + connectivity
        
        self.global_encoder = nn.Sequential(
            nn.Linear(total_bar_dim + total_force_dim + total_static_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim)
        )
        
        self.to(device)
    
    def encode_force(self, force_direction):
        """Encode force direction information"""
        batch_size = force_direction.shape[0]
        
        # Convert direction index to angle
        directions = torch.linspace(0, 2 * torch.pi, 8, endpoint=False, device=device)
        angles = directions[force_direction]  # [B]
        
        # Convert to [cos(theta), sin(theta)]
        force_vectors = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=-1)  # [B, 2]
        
        # Encode force direction
        force_encoded = self.force_encoder(force_vectors)  # [B, force_embed_dim]
        
        return force_encoded
    
    def encode_bars(self, design_state, compliance):
        """Encode bar information including design state and compliance"""
        batch_size = design_state.shape[0]
        
        # Normalize compliance values to prevent very small numbers
        compliance_sum = torch.sum(compliance, dim=1, keepdim=True) + 1e-8
        normalized_compliance = compliance / compliance_sum
        
        # Combine design state and normalized compliance for each bar
        bar_features = torch.stack([design_state, normalized_compliance], dim=-1)  # [B, n_bars, 2]
        
        # Encode each bar
        bar_encoded = self.bar_encoder(bar_features)  # [B, n_bars, bar_embed_dim]
        bar_encoded = bar_encoded.flatten(1)  # [B, n_bars * bar_embed_dim]
        
        return bar_encoded
    
    def encode_observation(self, design_state, compliance, force_node, force_dir):
        """Encode complete truss observation with dynamic encoding + static raw features"""
        batch_size = design_state.shape[0]
        
        # Ensure inputs are on correct device
        device = next(self.parameters()).device
        design_state = design_state.to(device)
        compliance = compliance.to(device)
        force_node = force_node.to(device)
        force_dir = force_dir.to(device)
        
        # Encode dynamic features
        bar_encoded = self.encode_bars(design_state, compliance)
        force_encoded = self.encode_force(force_dir)
        
        # Static features (raw, no encoding needed)
        node_positions = self.node_positions.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_nodes, 2]
        node_positions = node_positions.flatten(1)  # [B, n_nodes * 2]
        
        node_fixed = self.node_fixed.unsqueeze(0).expand(batch_size, -1)  # [B, n_nodes]
        
        connectivity = self.connectivity_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_bars, n_nodes]
        connectivity = connectivity.flatten(1)  # [B, n_bars * n_nodes]
        
        # Combine encoded dynamic features + raw static features
        full_obs = torch.cat([
            bar_encoded,      # [B, n_bars * bar_embed_dim]
            force_encoded,    # [B, force_embed_dim]
            node_positions,   # [B, n_nodes * 2]
            node_fixed,       # [B, n_nodes]
            connectivity      # [B, n_bars * n_nodes]
        ], dim=-1)
        
        # Global encoding
        encoded_obs = self.global_encoder(full_obs)
        
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
            act_prob = mask * act_prob + mask * self.mask_prob

        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return logprob, value, entropy 