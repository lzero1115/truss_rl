import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32


class MLPActorCritic(nn.Module):
    def __init__(self, n_bars, obs_dim=256,
                 hidden_sizes=(512, 512, 256)):
        super().__init__()

        # === Simplified Observation Encoder ===
        # Only use design state (no node/direction encoding)
        action_dim = n_bars

        force_dim = 8
        node_dim = 9
        encode_dim_1 = 4
        encode_dim_2 = 4

        self.force_node_embedding = nn.Embedding(node_dim, encode_dim_1, device=device)
        self.force_direction_embedding = nn.Embedding(force_dim, encode_dim_2, device=device)


        self.raw_obs_dim = 2 * n_bars + encode_dim_1 + encode_dim_2
        # else:
        #     self.raw_obs_dim = n_bars  # only design_state

        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.raw_obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.Sigmoid(),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.Sigmoid(),
            nn.Linear(obs_dim, obs_dim)
        )

        # === Policy Network ===
        policy_layers = []
        prev_dim = obs_dim
        for hidden_size in hidden_sizes:
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Sigmoid()
            ])
            prev_dim = hidden_size
        policy_layers.append(nn.Linear(prev_dim, action_dim))
        policy_layers.append(nn.LayerNorm(action_dim))
        policy_layers.append(nn.Sigmoid())

        self.policy_net = nn.Sequential(*policy_layers)

        # === Value Network ===
        value_layers = []
        prev_dim = obs_dim
        for hidden_size in hidden_sizes:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Sigmoid()
            ])
            prev_dim = hidden_size
        value_layers.append(nn.Linear(prev_dim, 1))
        value_layers.append(nn.Tanh())

        self.value_net = nn.Sequential(*value_layers)

        self.mask_prob = 1E-9
        self.to(device)

    def encode_obs(self, design_state, compliance, force_node, force_dir):
        raw_obs = torch.cat([design_state, compliance], dim=-1)  # [B, 2 * n_bars]

        # 🛡️ Force all inputs to model's device
        device = next(self.parameters()).device
        raw_obs = raw_obs.to(device)
        force_node = force_node.to(device)
        force_dir = force_dir.to(device)

        node_embed = self.force_node_embedding(force_node)
        direction_embed = self.force_direction_embedding(force_dir)

        full_obs = torch.cat([raw_obs, node_embed, direction_embed], dim=-1).to(device)

        encoded_obs = self.encoder_mlp(full_obs)

        return encoded_obs

    def act(self, design_state, compliance, force_node, force_dir, mask=None, deterministic=False):

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

        obs = self.encode_obs(design_state, compliance, force_node, force_dir)

        act_prob = self.policy_net(obs)

        if mask is not None:
            act_prob = mask * act_prob + mask * self.mask_prob

        dist = Categorical(act_prob)
        value = self.value_net(obs).squeeze(-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return logprob, value, entropy
