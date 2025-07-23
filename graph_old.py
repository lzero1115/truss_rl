import torch
from torch_geometric.data import HeteroData
import numpy as np
from src.Truss import TrussStructure
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

class TrussGraphConstructor:
    def __init__(self, truss: TrussStructure):
        """
        truss: TrussStructure object
        """
        self.truss = copy.deepcopy(truss)
        self.n_joints = len(truss.nodes)
        self.n_bars = len(truss.all_edges)

    def compute_graph(self, design_state, compliance, force_node, force_dir):
        """
        Build a HeteroData graph for a single observation.
        Args:
            design_state: [n_bars] 0/1 array (1=present, 0=absent)
            compliance: [n_bars] array, per-bar compliance values
            force_node: int (index of loaded joint)
            force_dir: int (index or one-hot, as desired)
        Returns:
            HeteroData object encoding the current truss state
        """
        data = HeteroData()

        # --- Joint node features ---
        coords = np.array(self.truss.nodes)  # [n_joints, 3]
        fixed_nodes = self.truss.fixed_nodes  # list of indices
        is_fixed = np.zeros(self.n_joints, dtype=np.float32)
        is_fixed[fixed_nodes] = 1.0
        is_loaded = np.zeros(self.n_joints, dtype=np.float32)
        is_loaded[force_node] = 1.0
        # Encode force_dir as one-hot (optional: can use int index)
        force_dir_feat = np.zeros((self.n_joints, 8), dtype=np.float32)  # assuming 8 directions
        force_dir_feat[force_node, force_dir] = 1.0

        joint_features = np.concatenate([
            coords,
            is_fixed[:, None],
            is_loaded[:, None],
            force_dir_feat
        ], axis=1)  # [n_joints, 3+1+1+8]
        data['joint'].x = torch.tensor(joint_features, dtype=floatType, device=device)

        # --- Bar node features ---
        design_state = np.array(design_state).astype(np.float32)  # [n_bars]
        compliance = np.array(compliance).astype(np.float32)    # [n_bars]
        bar_endpoints = np.array(self.truss.all_edges, dtype=np.int64)  # [n_bars, 2]
        bar_features = np.concatenate([
            design_state[:, None],
            compliance[:, None],
            bar_endpoints
        ], axis=1)  # [n_bars, 1+1+2]
        data['bar'].x = torch.tensor(bar_features, dtype=floatType, device=device)

        # --- Edges: bar <-> joint ---
        # For each bar, connect to its two endpoint joints
        edge_index_bar_to_joint = []
        edge_index_joint_to_bar = []
        for bar_idx, (u, v) in enumerate(bar_endpoints):
            edge_index_bar_to_joint.append([bar_idx, u])
            edge_index_bar_to_joint.append([bar_idx, v])
            edge_index_joint_to_bar.append([u, bar_idx])
            edge_index_joint_to_bar.append([v, bar_idx])
        edge_index_bar_to_joint = torch.tensor(edge_index_bar_to_joint, dtype=intType, device=device).t().contiguous()
        edge_index_joint_to_bar = torch.tensor(edge_index_joint_to_bar, dtype=intType, device=device).t().contiguous()
        data['bar', 'connects', 'joint'].edge_index = edge_index_bar_to_joint
        data['joint', 'rev_connects', 'bar'].edge_index = edge_index_joint_to_bar

        return data

    def graphs(self, design_states, compliances, force_nodes, force_dirs):
        """
        Build a list of HeteroData graphs for a batch of observations.
        Args:
            design_states: [batch_size, n_bars] array-like
            compliances: [batch_size, n_bars] array-like
            force_nodes: [batch_size] list/array of ints
            force_dirs: [batch_size] list/array of ints
        Returns:
            List of HeteroData objects (one per observation)
        """
        batch_size = len(design_states)
        datalist = []
        for i in range(batch_size):
            datalist.append(self.compute_graph(
                design_states[i],
                compliances[i],
                force_nodes[i],
                force_dirs[i]
            ))
        return datalist