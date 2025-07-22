import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, deque
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import LinAlgError

# Import the existing bar classes
from src.BarLinearElastic import BarLinearElastic
from src.Bar import Bar
from src.BarMaterial import BarMaterial
from src.BarCrossSection import BarCrossSectionRound
from src.CoordinateSystem import CoordinateSystem

class TrussGPU:
    """
    GPU-accelerated truss structure solver using PyTorch.
    Mirrors the API of TrussStructure but uses GPU for linear algebra.
    Uses the existing BarLinearElastic class for proper bar elements.
    """
    
    def __init__(self, nodes, all_edges, design_variables, fixed_nodes, consider_weight=False):
        """
        Initialize GPU truss structure.
        
        Args:
            nodes: (N, 3) array of node coordinates
            all_edges: list of (i, j) node index pairs
            design_variables: array of bar densities (0-1)
            fixed_nodes: list of fixed node indices
            consider_weight: whether to consider self-weight
        """
        if len(all_edges) != len(design_variables):
            raise ValueError("Number of design variables must match number of edges.")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert inputs to tensors
        self.nodes = torch.tensor(nodes, dtype=torch.float32, device=self.device)
        self.all_edges = [(min(u, v), max(u, v)) for (u, v) in all_edges]
        self.temp_design = torch.tensor(design_variables, dtype=torch.float32, device=self.device)
        self.fixed_nodes = fixed_nodes
        self.consider_selfweight = consider_weight
        
        # Structure properties
        self.n_nodes = len(nodes)
        self.dof_per_node = 6  # 3D with rotations
        self.total_dofs = self.n_nodes * self.dof_per_node
        
        # Validation
        if not self.graph_check():
            raise ValueError("Truss design infeasible!")
        
        # Compliance tracking (same as original)
        self.bar_compliances = [0.0] * len(self.all_edges)
        self.total_compliance = 0.0
        
        # Material properties (same as original)
        self.E = 5e6  # Pa
        self.G = 46071428.57142857  # Pa
        self.rho = 58.0  # kg/m^3
        self.mu = self.E / (2 * self.G) - 1
        
        # Cross-sectional properties
        self.min_radius = 1e-6
        self.design_area = 7.853981633974483e-05  # m^2
        self.design_radius = np.sqrt(self.design_area / np.pi)
        self.design_section = BarCrossSectionRound(radius=self.design_radius)
        self.min_section = BarCrossSectionRound(radius=self.min_radius)
        
        # Initialize lists (same as original)
        self.bars = []
        self.bars_elastic = []
        self.volumes = []
        self.vertex_dof_indices = []
        self.edge_dof_indices = []
        self.K_bar = []
        self.K_virtual_bar = []
        
        # Initialize DOF mapping
        self._initialize_dof_mapping()
        
        # Initialize structure
        self._initialize_structure()
        
        # Global stiffness matrix and prefactorization (will be computed on demand)
        self.K = None
        self.K_factorized = None
        self._last_design_hash = None
    
    def graph_check(self):
        """
        Returns True if the graph formed by self.all_edges:
        - Contains only valid edges (no out-of-bounds or self-loop)
        - Is connected (all involved nodes reachable)
        Returns False otherwise.
        """
        adjacency = defaultdict(list)
        involved_nodes = set()

        # Check edge indices are valid and build adjacency list
        for u, v in self.all_edges:
            if u >= self.n_nodes or v >= self.n_nodes or u == v:
                return False  # invalid index or self-loop
            adjacency[u].append(v)
            adjacency[v].append(u)
            involved_nodes.update([u, v])

        if not involved_nodes:
            return False  # No edges = disconnected

        # DFS to check connectivity
        visited = set()
        stack = [next(iter(involved_nodes))]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return visited == involved_nodes
    
    def _initialize_dof_mapping(self):
        """Initialize DOF mapping for fixed nodes."""
        self.map_dof_entire2subset = {}
        self.map_dof_subset2entire = {}
        
        new_dof = 0
        for dof in range(self.total_dofs):
            node_id = dof // self.dof_per_node
            if node_id not in self.fixed_nodes:
                self.map_dof_entire2subset[dof] = new_dof
                self.map_dof_subset2entire[new_dof] = dof
                new_dof += 1
        
        self.proj_dofs = new_dof
        
    def _initialize_structure(self):
        """Initialize bars and compute stiffness matrices."""
        self.clear_compliance()
        self.compute_bars()
        self.compute_bars_linear_elasticity()
        
    def compute_bars(self):
        """Compute bar properties based on design variables."""
        self.volumes.clear()
        self.bars.clear()
        
        for idx, edge in enumerate(self.all_edges):
            start_node = self.nodes[edge[0]].cpu().numpy()
            end_node = self.nodes[edge[1]].cpu().numpy()
            
            # Create coordinate system
            coord = CoordinateSystem(origin=start_node, zaxis=end_node - start_node)
            
            # Compute bar length (same as original)
            length = float(np.linalg.norm(end_node - start_node))
            
            # Create bar with appropriate section based on design variable
            coef = self.temp_design[idx].item()
            if coef > 1e-6:
                radius = self.design_radius * np.sqrt(coef)
                section = BarCrossSectionRound(radius=radius)
                volume = coef * self.design_section.Ax_ * length
            else:
                section = self.min_section
                volume = 0.0
            
            # Create bar
            bar = Bar(coord, length, section)
            self.bars.append(bar)
            self.volumes.append(volume)
    
    def compute_bars_linear_elasticity(self):
        """Create BarLinearElastic objects for all bars."""
        self.bars_elastic.clear()
        self.vertex_dof_indices.clear()
        self.edge_dof_indices.clear()
        
        # Setup vertex DOF indices (same as original)
        for i in range(len(self.nodes)):
            dofs = [i * 6 + j for j in range(6)]
            self.vertex_dof_indices.append(dofs)
        
        # Setup edge DOF indices and create elastic bars
        for i, bar in enumerate(self.bars):
            material = BarMaterial(E=self.E, mu=self.mu, rho=self.rho, section=bar.section_)
            bar_elastic = BarLinearElastic(bar, material)
            self.bars_elastic.append(bar_elastic)
            
            # Setup edge DOF indices (same as original)
            edge_dof = []
            for vertex_id in self.all_edges[i]:
                edge_dof.extend(self.vertex_dof_indices[vertex_id])
            self.edge_dof_indices.append(edge_dof)
    
    def _compute_stiffness_matrix(self):
        """Assemble global stiffness matrix using existing bar elements."""
        # Clear and rebuild stiffness matrices (same as original)
        self.K_virtual_bar.clear()
        self.K_bar.clear()
        
        triplet_list = []
        
        for i, elastic_bar in enumerate(self.bars_elastic):
            k_G = elastic_bar.create_global_stiffness_matrix()
            K_a, K_v = self.get_bar_stiffness_matrix(i)
            self.K_bar.append(K_a)
            self.K_virtual_bar.append(K_v)
            self._assemble_stiff_matrix(triplet_list, k_G, i)
        
        # Create sparse matrix from triplets
        if triplet_list:
            rows, cols, data = zip(*triplet_list)
            K = sparse.csr_matrix((data, (rows, cols)), shape=(self.proj_dofs, self.proj_dofs))
        else:
            K = sparse.csr_matrix((self.proj_dofs, self.proj_dofs))
        
        # Add small regularization for numerical stability
        K += 1e-8 * sparse.eye(self.proj_dofs)
        
        return K
    
    def _get_design_hash(self):
        """Get a hash of the current design variables for caching."""
        return hash(tuple(self.temp_design.cpu().numpy().flatten()))
    
    def _ensure_stiffness_matrix_cached(self):
        """Ensure stiffness matrix is computed and cached if design has changed."""
        current_hash = self._get_design_hash()
        
        if self._last_design_hash != current_hash:
            # Design has changed, recompute stiffness matrix
            self.K = self._compute_stiffness_matrix()
            self.K_factorized = None  # Clear old factorization
            self._last_design_hash = current_hash
    
    def create_external_force_vector(self, force_indices, force_vectors):
        """Create external force vector (same API as original)."""
        if len(force_indices) != len(force_vectors):
            raise ValueError("Number of force indices must match number of force vectors")
        
        # Initialize force vector
        external_forces = torch.zeros(self.total_dofs, device=self.device)
        
        # Add forces at specified nodes
        for i, node_idx in enumerate(force_indices):
            if node_idx < 0 or node_idx >= self.n_nodes:
                raise ValueError(f"Node index {node_idx} is out of bounds")
            
            force = torch.tensor(force_vectors[i], dtype=torch.float32, device=self.device)
            dofs = len(force_vectors[i])
            base_idx = node_idx * self.dof_per_node
            external_forces[base_idx:base_idx + dofs] = force
        
        return external_forces
    
    def _compute_loads(self, external_forces=None):
        """Compute global load vector including self-weight."""
        F = torch.zeros(self.proj_dofs, device=self.device)
        
        # Add self-weight if enabled
        if self.consider_selfweight:
            for bar_idx, bar_elastic in enumerate(self.bars_elastic):
                coef = self.temp_design[bar_idx].item()
                if coef > 1e-6:  # Only consider designed edges for self-weight
                    load = bar_elastic.create_global_self_weight()
                    self._assembly_force(F, load, bar_idx)
        
        # Add external forces if provided
        if external_forces is not None:
            for i in range(external_forces.shape[0]):
                if i in self.map_dof_entire2subset:
                    subset_dof = self.map_dof_entire2subset[i]
                    if subset_dof < self.proj_dofs:  # Safety check
                        F[subset_dof] += external_forces[i]
        
        return F
    
    def _assembly_force(self, F, g, edge_id):
        """Assemble element force vector into global force vector."""
        new_dofs = []
        for j in range(len(self.edge_dof_indices[edge_id])):
            old_dof = self.edge_dof_indices[edge_id][j]
            new_dofs.append(self.map_dof_entire2subset.get(old_dof, -1))
        
        for j, new_dof in enumerate(new_dofs):
            if new_dof != -1:
                F[new_dof] += g[j]
    
    def solve_elasticity(self, external_forces=None):
        """
        Solve Kx = f using optimized solver strategy.
        
        Returns:
            displacement: Full displacement vector (including fixed DOFs as zeros)
            success: Boolean indicating if solve was successful
            message: Status message
        """
        try:
            # Ensure stiffness matrix is computed and cached
            self._ensure_stiffness_matrix_cached()
            
            # Compute load vector
            F = self._compute_loads(external_forces)
            
            # Convert to numpy for scipy solver
            F_np = F.cpu().numpy()
            
            # Choose solver strategy based on problem size
            if self.proj_dofs < 50:  # Small problems: use direct solver
                D = linalg.spsolve(self.K, F_np)
                solver_type = "Direct"
            else:  # Large problems: use prefactorization
                if self.K_factorized is None:
                    self.K_factorized = linalg.splu(self.K)
                    solver_type = "LU"
                else:
                    solver_type = "LU (cached)"
                
                D = self.K_factorized.solve(F_np)
            
            # Reconstruct full displacement vector
            displacement = np.zeros(self.total_dofs)
            for i in range(D.shape[0]):
                old_dof = self.map_dof_subset2entire[i]
                displacement[old_dof] = D[i]
            
            # Compute compliance
            self.compute_bar_compliance(D)
            self.total_compliance = sum(self.bar_compliances)
            
            return displacement, True, f"Success ({solver_type})"
            
        except Exception as e:
            # Return zero displacement on failure
            displacement = np.zeros(self.total_dofs)
            return displacement, False, f"Solve failed: {str(e)}"
    
    def compute_bar_compliance(self, displacement):
        """Compute compliance for each bar (same logic as original)."""
        expected_dim = self.proj_dofs
        if displacement.shape[0] != expected_dim:
            raise ValueError(
                f"Displacement vector has incorrect size: {displacement.shape[0]}, expected {expected_dim}"
            )

        for i in range(len(self.all_edges)):
            k = self.K_bar[i]
            # Convert sparse matrix to dense for computation
            if hasattr(k, 'toarray'):
                k_dense = k.toarray()
            else:
                k_dense = k.cpu().numpy() if hasattr(k, 'cpu') else k
            c_e = float(displacement @ k_dense @ displacement)
            self.bar_compliances[i] = c_e
    
    def clear_compliance(self):
        """Clear compliance data."""
        self.total_compliance = 0.0
        self.bar_compliances = [0.0] * len(self.all_edges)
    
    def interpolate_polynomial(self, D_local: np.ndarray, L: float) -> List[np.ndarray]:
        """Interpolate displacement polynomial."""
        # rotation angle around z axis already negative
        # negative sign refers to the sign in bending stiffness matrix
        d_y = np.array([D_local[1], D_local[7], D_local[5], D_local[11]])
        d_z = np.array([D_local[2], D_local[8], -D_local[4], -D_local[10]])

        A = np.zeros((4, 4))
        u0, u6 = D_local[0], D_local[6] + L

        A[0] = [1.0, u0, u0 * u0, u0 * u0 * u0]
        A[1] = [1.0, u6, u6 * u6, u6 * u6 * u6]
        A[2] = [0.0, 1.0, 2 * u0, 3 * u0 * u0]
        A[3] = [0.0, 1.0, 2 * u6, 3 * u6 * u6]

        return [np.linalg.solve(A, d_y), np.linalg.solve(A, d_z)]
    
    def get_truss_volume(self):
        """Get total truss volume."""
        if self.volumes:
            return sum(self.volumes)
        else:
            return 0
    
    def update_bars_with_weight(self, k):
        """Update design variables and recompute structure."""
        if len(k) != len(self.temp_design):
            raise ValueError("The length of k does not match the number of edges.")
        
        self.temp_design = torch.tensor(k, dtype=torch.float32, device=self.device)
        self._initialize_structure()
    
    def update_bars_with_weight_dict(self, weight_dict: Dict[Tuple[int, int], float], ct_new=False):
        """Update design variables using a dictionary mapping edges to weights."""
        # Convert dictionary to array format
        new_design = self.temp_design.clone()
        
        for (node1, node2), weight in weight_dict.items():
            edge_idx = self.get_edge_index(node1, node2)
            if edge_idx != -1:
                new_design[edge_idx] = weight
        
        self.temp_design = new_design
        self._initialize_structure()
    
    def get_edge_index(self, node1, node2):
        """Get edge index for given nodes."""
        norm_id = (min(node1, node2), max(node1, node2))
        try:
            return self.all_edges.index(norm_id)
        except ValueError:
            return -1
    
    def get_bar_stiffness_matrix(self, idx: int):
        """Get bar stiffness matrix (properly handling virtual bars like original)."""
        node1 = self.all_edges[idx][0]
        node2 = self.all_edges[idx][1]
        
        if node1 in self.fixed_nodes and node2 in self.fixed_nodes:
            reg_value = 1e-9
            k_reg = torch.eye(self.proj_dofs, device=self.device) * reg_value
            return k_reg, k_reg

        # Get actual bar
        elastic_bar = self.bars_elastic[idx]
        
        # Create virtual bar with minimum section
        start_node = self.nodes[node1].cpu().numpy()
        end_node = self.nodes[node2].cpu().numpy()
        coord = CoordinateSystem(origin=start_node, zaxis=end_node - start_node)
        length = float(np.linalg.norm(end_node - start_node))
        virtual_bar = Bar(coord, length, self.min_section)
        virtual_material = BarMaterial(E=self.E, mu=self.mu, rho=self.rho, section=self.min_section)
        virtual_elastic_bar = BarLinearElastic(virtual_bar, virtual_material)

        # Get stiffness matrices
        k_G = elastic_bar.create_global_stiffness_matrix()
        k_G_virtual = virtual_elastic_bar.create_global_stiffness_matrix()

        # Map DOFs for this bar
        new_dofs = []
        for j in range(len(self.edge_dof_indices[idx])):
            old_dof = self.edge_dof_indices[idx][j]
            new_dofs.append(self.map_dof_entire2subset.get(old_dof, -1))

        # Create actual and virtual stiffness matrices
        k_actual = torch.zeros(self.proj_dofs, self.proj_dofs, device=self.device)
        k_virtual = torch.zeros(self.proj_dofs, self.proj_dofs, device=self.device)
        
        for j in range(12):
            for k in range(12):
                new_dof_j = new_dofs[j]
                new_dof_k = new_dofs[k]
                if new_dof_j != -1 and new_dof_k != -1:
                    k_actual[new_dof_j, new_dof_k] += float(k_G[j, k])
                    k_virtual[new_dof_j, new_dof_k] += float(k_G_virtual[j, k])

        # Add regularization
        k_actual += 1e-8 * torch.eye(self.proj_dofs, device=self.device)
        k_virtual += 1e-8 * torch.eye(self.proj_dofs, device=self.device)

        return k_actual, k_virtual
    
    def _assemble_stiff_matrix(self, K_tri, k_G, edge_id):
        """Assemble element stiffness matrix into global stiffness matrix."""
        new_dofs = []
        for j in range(len(self.edge_dof_indices[edge_id])):
            old_dof = self.edge_dof_indices[edge_id][j]
            new_dofs.append(self.map_dof_entire2subset.get(old_dof, -1))
        
        for j in range(12):
            for k in range(12):
                if abs(float(k_G[j, k])) < 1e-12:
                    continue
                new_dof_j = new_dofs[j]
                new_dof_k = new_dofs[k]
                if new_dof_j != -1 and new_dof_k != -1:
                    K_tri.append((new_dof_j, new_dof_k, float(k_G[j, k])))
    
    @staticmethod
    def from_truss(truss):
        """Create TrussGPU from existing TrussStructure."""
        return TrussGPU(
            nodes=truss.nodes,
            all_edges=truss.all_edges,
            design_variables=truss.temp_design,
            fixed_nodes=truss.fixed_nodes,
            consider_weight=truss.consider_selfweight
        ) 