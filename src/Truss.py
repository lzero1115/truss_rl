import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, deque
from scipy import sparse
from scipy.sparse import linalg

from .BarLinearElastic import BarLinearElastic
from .Bar import Bar
from .BarMaterial import BarMaterial
from .BarCrossSection import BarCrossSection, BarCrossSectionRound, CrossSectionType
from .CoordinateSystem import CoordinateSystem
from .jetmap import jetmap

class TrussStructure: # guarantee a nice index mapping
    def __init__(self,
                 nodes, # Vx3 vertex list
                 all_edges, # Ex2 design feasible region
                 design_variables, # Ex1 design varibales
                 fixed_nodes, # fixed node index
                 consider_weight = False # default false
                 ):

        if len(all_edges)!=len(design_variables):
            raise ValueError("Number of design variables must match number of edges.")
        self.nodes = nodes
        self.n_nodes = len(self.nodes)
        self.fixed_nodes = fixed_nodes

        # Normalize edges so (u, v) always has u < v
        self.all_edges = [(min(u, v), max(u, v)) for (u, v) in all_edges]

        self.temp_design = design_variables
        if not self.graph_check():
            raise ValueError("Truss design infeasible!")

        self.bar_discretization = 10 # use for visualizer
        self.bar_compliances = [0.0] * len(self.all_edges) # record per bar compliance including the virtual bars
        self.total_compliance = 0.0

        self.consider_selfweight = consider_weight

        # Material properties which can be manually defined
        self.E = 5e6  # Pa
        self.G = 46071428.57142857  # Pa
        self.rho = 58.0  # kg/m^3
        self.mu = self.E / (2 * self.G) - 1

        # Cross-sectional properties
        self.min_radius = 1e-6  # TODO: numerical stability virtual bar, manually adjust
        self.min_section = BarCrossSectionRound(radius=self.min_radius)
        self.design_area = 7.853981633974483e-05  # m^2
        self.design_radius = np.sqrt(self.design_area / np.pi)
        self.design_section = BarCrossSectionRound(radius=self.design_radius)

        # include virtual bars with the same order of edges
        self.bars: List[Bar] = []
        self.bars_elastic: List[BarLinearElastic] = []
        self.vertex_dof_indices: List[List[int]] = [] # no degraded index, 6 per vertex
        self.edge_dof_indices: List[List[int]] = [] # each bar
        self.volumes = [] # per bar volume

        self._initialize_structure()

        # DoF mapping variables
        self.map_dof_entire2subset: Dict[int, int] = {} # original dof to projected dof
        self.map_dof_subset2entire: Dict[int, int] = {} # projected dof to original dof
        self.proj_dofs: int = 0 # number of projected dofs exclude fixed nodes

        self._initialize_dof_mapping()

        self.K_bar = []  # per bar contribution to self.K, len(self.all_edges)
        self.K_virtual_bar = [] # virtual stiffness matrix par bar
        self.K = self._compute_stiff_matrix() # global stiffness matrix which used for Kd = f
        self.K = self.K + 1e-8 * sparse.eye(self.proj_dofs) # avoid singularity keep numerical stability


    def get_edge_index(self,node1, node2):

        norm_id = (min(node1,node2),max(node1,node2))
        try:
            return self.all_edges.index(norm_id)
        except ValueError:
            return -1

    def get_truss_volume(self):

        if self.volumes:
            return sum(self.volumes)
        else:
            return 0

    def graph_check(self):
        """
        Returns True if the graph formed by self.all_edges:
        - Contains only valid edges (no out-of-bounds or self-loop)
        - Is connected (all involved nodes reachable)
        Returns False otherwise.
        """
        from collections import defaultdict

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

    def _initialize_structure(self):
        """Initialize bars and setup for simulation."""
        self.clear_compliance()
        self.compute_bars()
        self.compute_bars_linear_elasticity()

    def compute_bars(self):

        self.volumes.clear()
        self.bars.clear()
        for idx, edge in enumerate(self.all_edges):
            start_node = self.nodes[edge[0]]
            end_node = self.nodes[edge[1]]
            coord = CoordinateSystem(origin=start_node, zaxis=end_node - start_node)

            coef = self.temp_design[idx]
            if coef > 1e-6:
                radius = self.design_radius * np.sqrt(coef) # linear interpolation
                section = BarCrossSectionRound(radius=radius)
            else:
                section = self.min_section

            length = float(np.linalg.norm(end_node - start_node))
            volume = coef * self.design_section.Ax_ * length if coef > 1e-6 else 0.0
            bar = Bar(coord, length, section)
            self.bars.append(bar)
            self.volumes.append(volume)


    def compute_bars_linear_elasticity(self):
        self.bars_elastic.clear()
        self.vertex_dof_indices.clear()
        self.edge_dof_indices.clear()

        for i in range(len(self.nodes)):
            dofs = [i * 6 + j for j in range(6)]
            self.vertex_dof_indices.append(dofs)

        # each bar is assigned with two nodes
        for i, bar in enumerate(self.bars):
            material = BarMaterial(E=self.E, mu=self.mu, rho=self.rho, section=bar.section_)
            bar_elastic = BarLinearElastic(bar, material)
            self.bars_elastic.append(bar_elastic)

            edge_dof = []
            for vertex_id in self.all_edges[i]:
                edge_dof.extend(self.vertex_dof_indices[vertex_id])
            self.edge_dof_indices.append(edge_dof)

    def _initialize_dof_mapping(self):

        self.map_dof_entire2subset.clear()
        self.map_dof_subset2entire.clear()

        new_dof = 0
        for dof in range(6 * len(self.nodes)):
            node_id = dof // 6
            if node_id not in self.fixed_nodes:
                self.map_dof_entire2subset[dof] = new_dof
                self.map_dof_subset2entire[new_dof] = dof
                new_dof += 1

        self.proj_dofs = new_dof

    def get_bar_stiffness_matrix(self, idx: int) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:

        node1 = self.all_edges[idx][0]
        node2 = self.all_edges[idx][1]
        if node1 in self.fixed_nodes and node2 in self.fixed_nodes:
            reg_value = 1e-9 # TODO: can be adjusted manually, actually this should be 0 since fixed bar contributes 0
            k_reg = sparse.eye(self.proj_dofs) * reg_value
            return k_reg, k_reg

        # coeff = self.temp_design[idx]
        elastic_bar = self.bars_elastic[idx]

        start_node = self.nodes[node1]
        end_node = self.nodes[node2]
        coord = CoordinateSystem(origin=start_node, zaxis=end_node - start_node)
        length = float(np.linalg.norm(end_node - start_node))
        virtual_bar = Bar(coord, length, self.min_section)
        # Create elastic bar with minimum section
        virtual_material = BarMaterial(E=self.E, mu=self.mu, rho=self.rho, section=self.min_section)
        virtual_elastic_bar = BarLinearElastic(virtual_bar, virtual_material)

        k_G = elastic_bar.create_global_stiffness_matrix()
        # Get virtual bar's global stiffness matrix
        k_G_virtual = virtual_elastic_bar.create_global_stiffness_matrix()

        triplet_list_actual = []
        triplet_list_virtual = []

        # Map the DoFs for this bar
        new_dofs = []
        for j in range(len(self.edge_dof_indices[idx])):
            old_dof = self.edge_dof_indices[idx][j]
            new_dofs.append(self.map_dof_entire2subset.get(old_dof, -1))

        for j in range(12):
            for k in range(12):
                new_dof_j = new_dofs[j]
                new_dof_k = new_dofs[k]
                if new_dof_j != -1 and new_dof_k != -1:
                    triplet_list_actual.append((new_dof_j, new_dof_k, float(k_G[j, k])))
                    triplet_list_virtual.append((new_dof_j, new_dof_k, float(k_G_virtual[j, k])))

        # Create sparse matrix for actual bar and virtual bar
        k_actual = sparse.csr_matrix((self.proj_dofs, self.proj_dofs))
        if triplet_list_actual:
            rows, cols, data = zip(*triplet_list_actual)
            k_actual = sparse.csr_matrix((data, (rows, cols)), shape=(self.proj_dofs, self.proj_dofs))
        k_virtual = sparse.csr_matrix((self.proj_dofs, self.proj_dofs))
        if triplet_list_virtual:
            rows, cols, data = zip(*triplet_list_virtual)
            k_virtual = sparse.csr_matrix((data, (rows, cols)), shape=(self.proj_dofs, self.proj_dofs))

        k_actual = k_actual + 1e-8 * sparse.eye(self.proj_dofs)
        k_virtual = k_virtual + 1e-8 * sparse.eye(self.proj_dofs) # numerical stability

        return k_actual, k_virtual


    def _compute_stiff_matrix(self) -> sparse.csr_matrix:
        """Compute global stiffness matrix."""
        self.K_virtual_bar.clear()
        self.K_bar.clear()

        triplet_list = []

        for i, elastic_bar in enumerate(self.bars_elastic):
            k_G = elastic_bar.create_global_stiffness_matrix() # still treat the virtual bar a bar :(
            K_a, K_v = self.get_bar_stiffness_matrix(i)
            self.K_bar.append(K_a)
            self.K_virtual_bar.append(K_v)
            self._assemble_stiff_matrix(triplet_list, k_G, i)

        if triplet_list:
            rows, cols, data = zip(*triplet_list)
            return sparse.csr_matrix((data, (rows, cols)), shape=(self.proj_dofs, self.proj_dofs))

        return sparse.csr_matrix((self.proj_dofs, self.proj_dofs))

    def _assemble_stiff_matrix(self, K_tri: List[Tuple[int, int, float]],
                               k_G: np.ndarray,
                               edge_id: int):
        """Assemble element stiffness matrix into global stiffness matrix."""
        new_dofs = [] # degenerate the fixed nodes degrees
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

    def create_external_force_vector(self, force_indices, force_vectors): # unprojected force!
        """
        Create an external force vector for the truss structure based on specified forces.

        Args:
            force_indices: List/array of node indices where forces are applied
            force_vectors: List/array of 3D force vectors (same length as force_indices)
                          Each force vector should be [Fx, Fy, Fz]

        Returns:
            external_forces: Numpy array of size 6*num_nodes with forces at specified positions
        """
        # Validate inputs
        if len(force_indices) != len(force_vectors):
            raise ValueError("Number of force indices must match number of force vectors")

        # Initialize external force vector with zeros (6 DOFs per node)
        external_forces = np.zeros(len(self.nodes) * 6)

        # Populate the force vector at specified positions
        for i, node_idx in enumerate(force_indices):
            if node_idx < 0 or node_idx >= len(self.nodes):
                raise ValueError(f"Node index {node_idx} is out of bounds")
            if len(force_vectors[i]) > 6 or len(force_vectors[i]) == 0:
                raise ValueError(f"Bad force")

            # Convert force vector to numpy array if it's not already
            force = np.array(force_vectors[i], dtype=float)
            dofs = force.size


            base_idx = node_idx * 6
            external_forces[base_idx:base_idx + dofs] = force

        return external_forces


    def _compute_loads(self, external_forces: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute global load vector including self-weight for designed edges and external forces."""
        # external_forces should be (6 x self.n_nodes) degree

        F = np.zeros(self.proj_dofs) # projected lower dimension space

        # Get the set of designed edges for efficient lookup
        if self.consider_selfweight:
            # Add self-weight for designed edges only
            for beam_id in range(len(self.bars_elastic)):
                design_val = self.temp_design[beam_id]
                if design_val > 1e-6:  # Only consider designed edges for self-weight
                    load = self.bars_elastic[beam_id].create_global_self_weight()
                    self._assembly_force(F, load, beam_id) # add to F

        # Add external forces if provided
        if external_forces is not None:
            for i in range(external_forces.shape[0]):
                if i in self.map_dof_entire2subset:
                    F[self.map_dof_entire2subset[i]] += external_forces[i]

        return F # projected global force vec

    def _assembly_force(self, F: np.ndarray, g: np.ndarray, edge_id: int):
        """Assemble element force vector into global force vector."""
        new_dofs = []
        for j in range(len(self.edge_dof_indices[edge_id])):
            old_dof = self.edge_dof_indices[edge_id][j]
            new_dofs.append(self.map_dof_entire2subset.get(old_dof, -1))

        for j, new_dof in enumerate(new_dofs):
            if new_dof != -1:
                F[new_dof] += g[j]

    def solve_elasticity(self, external_forces: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, str]:

        try:
            F = self._compute_loads(external_forces)
            D = sparse.linalg.spsolve(self.K, F)  # Solve KD = F

            # Reconstruct full global displacement (with fixed DoFs = 0)
            displacement = np.zeros(self.n_nodes * 6)
            for i in range(D.shape[0]):
                old_dof = self.map_dof_subset2entire[i]
                displacement[old_dof] = D[i]

            self.compute_bar_compliance(D)
            self.total_compliance = sum(self.bar_compliances)

            return displacement, True, "Success"

        except Exception as e:
            # Return zero displacement and a failure message
            displacement = np.zeros(self.n_nodes * 6)
            return displacement, False, f"Solve failed: {str(e)}"

    def compute_bar_compliance(self, displacement):
        expected_dim = self.proj_dofs # we compute in projected dimension
        if displacement.shape[0] != expected_dim:
            raise ValueError(
                f"Displacement vector has incorrect size: {displacement.shape[0]}, expected {expected_dim}"
            )

        for i in range(len(self.all_edges)):
            # if self.temp_design[i] > 1e-6:
            #     dofs = self.edge_dof_indices[i]
            #
            #     k = self.K_bar[i]
            #     c_e = float(displacement @ k @ displacement)
            #     self.bar_compliances[i] = c_e
            # else:
            #     self.bar_compliances[i] = 0.0
            k = self.K_bar[i]
            c_e = float(displacement @ k @ displacement)
            self.bar_compliances[i] = c_e

    def clear_compliance(self):

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

    def visualize_displacement(self, bar_id: int, displacement: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Visualize displacement for a single bar."""
        # displacement is 6 x len(self.n_nodes)

        def poly_val(x: float, c: np.ndarray) -> float:
            return sum(c[i] * x ** i for i in range(len(c)))

        node_u, node_v = self.all_edges[bar_id]
        end_u = self.nodes[node_u]
        end_v = self.nodes[node_v]

        R3 = self.bars_elastic[bar_id].create_global_transformation_matrix()
        R = self.bars_elastic[bar_id].turn_diagblock(R3)

        D_global = np.zeros(12)
        D_global[:6] = displacement[node_u * 6:(node_u + 1) * 6]
        D_global[6:] = displacement[node_v * 6:(node_v + 1) * 6]
        D_local = R @ D_global # local

        L = self.bars_elastic[bar_id].length_
        u_poly = self.interpolate_polynomial(D_local, L)

        polylines = []
        distance = []

        for i in range(self.bar_discretization + 1):
            t = i / self.bar_discretization
            s = D_local[0] + t * (D_local[6] + L - D_local[0])
            v = poly_val(s, u_poly[0])
            w = poly_val(s, u_poly[1])
            d = np.array([s, v, w]) # local displacement

            inter_pt = end_u + R3.T @ d
            ori_pt = end_u + t * (end_v - end_u)

            polylines.append(inter_pt) # global deformed section position
            distance.append(float(np.linalg.norm(inter_pt - ori_pt)))

        return polylines, distance

    def visualize_displacement_all(self, displacement: np.ndarray) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
        """Visualize displacement for design edges only."""
        # displacement is global variable
        segments_list = []
        deviation_list = []

        for i, edge in enumerate(self.all_edges):

            if self.temp_design[i] > 1e-6:
                segments, deviations = self.visualize_displacement(i, displacement)
                segments_list.append(segments)
                deviation_list.append(deviations)

        return segments_list, deviation_list

    def get_bar_geometry(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the static (undeformed) 3D mesh of each designed bar."""
        Vs, Fs = [], []
        for i, bar in enumerate(self.bars):

            if self.temp_design[i] > 1e-6:
                V, F = bar.get_mesh()
                Vs.append(V)
                Fs.append(F)
        return Vs, Fs

    def get_deformed_bar_geometry(self, displacement: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the deformed 3D mesh of each designed bar."""
        Vs, Fs = [], []

        for idx in range(len(self.all_edges)):
            if self.temp_design[idx] > 1e-6:
                segments, _ = self.visualize_displacement(idx, displacement)
                for i in range(self.bar_discretization):
                    end_u, end_v = segments[i], segments[i + 1]
                    cross_section = self.bars[idx].section_
                    bar = Bar.from_points(end_u, end_v, cross_section)
                    V, F = bar.get_mesh()
                    Vs.append(V)
                    Fs.append(F)

        return Vs, Fs

    def get_deformed_bar_displacement_colors(self, displacement: np.ndarray, max_disp: float) -> List[np.ndarray]:
        """Get colors based on deformation, one color per bar segment."""
        colors = []

        # Then process only those bars, creating colors for each segment
        for idx in range(len(self.all_edges)):
            if self.temp_design[idx] > 1e-6:
                _, deviations = self.visualize_displacement(idx, displacement)

                # Create a color for each segment of this bar
                for j in range(self.bar_discretization):
                    dev = (deviations[j] + deviations[j + 1]) / 2.0
                    color = jetmap(dev, 0, max_disp)
                    colors.append(color)

        return colors

    def update_bars_with_weight(self, k) -> None:
        """
        Args:
            k: new design variables (bar densities). Absolute value!
        """
        # Ensure rho is the same length as the number of bars
        if len(k) != len(self.temp_design):
            raise ValueError("The length of rho does not match the number of edges (bars) in the structure.")

        self.temp_design = k.copy()

        self._initialize_structure()
        self.K = self._compute_stiff_matrix()
        self.K = self.K + 1e-8 * sparse.eye(self.proj_dofs)


    def update_bars_with_weight_dict(self, weight_dict: Dict[Tuple[int, int], float], ct_new=False) -> None:
        # update some design variables for the truss
        if ct_new:
            k = np.zeros(len(self.temp_design))
        else:
            k = self.temp_design.copy()
        for edge, weight in weight_dict.items():
            node1 = edge[0]
            node2 = edge[1]
            idx = self.get_edge_index(node1, node2)
            if idx == -1:
                raise ValueError(f"Edge ({node1}, {node2}) not found in all_edges.")

            k[idx] = weight

        self.update_bars_with_weight(k)

