import copy
import numpy as np
from src.Truss import TrussStructure
from typing import List, Tuple, Optional, Dict
from scipy import sparse
from scipy.sparse import linalg


class TrussEnv:
    def __init__(self, truss: TrussStructure, volume_ratio: float = 0.5):
        self.temp_truss = copy.deepcopy(truss)
        self.volume_ratio = volume_ratio
        self.bar_vols = self.temp_truss.volumes
        self.ub = 1.2
        self.lb = 0.7
        self.target_volume_ub = self.volume_ratio * self.ub * self.temp_truss.get_truss_volume()
        self.target_volume_lb = self.volume_ratio * self.lb * self.temp_truss.get_truss_volume()
        self.n_bars = len(self.temp_truss.all_edges)
        self.compliance_threshold = 1.0  # manually set
        self.cratio = 0.2
        self.directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        self.compliance_cache = {}
        self.stability_cache = {}
        self.stiffness_cache = {}  # key: tuple(design_state), value: K
        self.curriculum = None
        self.optimal_compliance = {}
        # Store mappings and bar info from base truss
        self.map_dof_entire2subset = self.temp_truss.map_dof_entire2subset
        self.map_dof_subset2entire = self.temp_truss.map_dof_subset2entire
        self.K_bar = self.temp_truss.K_bar
        self.edge_dof_indices = [list(e) for e in self.temp_truss.edge_dof_indices]
        self.fixed_nodes = set(self.temp_truss.fixed_nodes)
        self.n_nodes = self.temp_truss.n_nodes

    @property
    def action_dim(self):
        return self.n_bars  # just remove

    def next_states(self, design_states, actions):  # batch size operation
        new_design_states = design_states.copy()
        inds = np.arange(len(design_states))

        # Check that all selected entries are not already 0
        if np.any(new_design_states[inds, actions] == 0):
            raise ValueError("Some selected bars are already removed (value is 0).")

        new_design_states[inds, actions] = 0  # just remove the bar
        return new_design_states

    def set_curriculum(self, curriculum: List[Dict]):
        self.curriculum = copy.deepcopy(curriculum)
        self.curriculum_design = [item['curriculum_design_variables'] for item in curriculum]

        for item in curriculum:
            node_idx = item['force_node_indices'][0]
            optimal_compliance = item['optimal_compliance']
            force_dir = item['direction_index']
            key = tuple([node_idx, force_dir])
            if key not in self.optimal_compliance:
                self.optimal_compliance[key] = optimal_compliance

        self.curriculum_force_inds = [item['force_node_indices'][0] for item in curriculum]
        self.curriculum_force = [item['force_list'] for item in curriculum]
        self.curriculum_force_dir = [item['direction_index'] for item in curriculum]
        self.force_amplitude = 0.5  # TODO: read it from json
        self.n_curriculum = len(curriculum)
        # Initialize caches for curriculum items (assume all are stable and feasible)
        self.compliance_cache = {}
        self.stability_cache = {}
        self.stiffness_cache = {}
        for i, design in enumerate(self.curriculum_design):
            force_node = self.curriculum_force_inds[i]
            force_dir = self.curriculum_force_dir[i]
            key = (tuple(design), int(force_node), int(force_dir))
            if key not in self.compliance_cache:
                compliance = self.compute_compliance_only(design, force_node, force_dir)
                self.compliance_cache[key] = compliance
                self.stability_cache[key] = 1
            # Precompute and cache K for this design state
            design_key = tuple(design)
            if design_key not in self.stiffness_cache:
                truss = TrussStructure(
                    nodes=self.temp_truss.nodes,
                    all_edges=self.temp_truss.all_edges,
                    design_variables=design,
                    fixed_nodes=self.temp_truss.fixed_nodes
                )
                self.stiffness_cache[design_key] = truss.K.copy()


    def action_masks(self, design_states):
        masks = (design_states > 1e-6).astype(np.int32)
        return masks

    def solve_elasticity_with_cache(self, design_state, force_node, force_dir):
        design_state = np.asarray(design_state)
        key = tuple(design_state)
        # Use cache if available
        if key in self.stiffness_cache:
            K = self.stiffness_cache[key]
            #map_dof_entire2subset = self.map_dof_entire2subset
            map_dof_subset2entire = self.map_dof_subset2entire
            K_bar = self.K_bar
            #edge_dof_indices = self.edge_dof_indices
            #fixed_nodes = self.fixed_nodes
            n_nodes = self.n_nodes
            proj_dofs = K.shape[0]
            # Build force vector
            directions = self.directions
            theta = directions[force_dir]
            fx, fy = np.cos(theta), np.sin(theta)
            force_vector = [fx * self.force_amplitude, fy * self.force_amplitude, 0.0]
            F_ext = self.temp_truss.create_external_force_vector([force_node], [force_vector])
            F_sub = self.temp_truss._compute_loads(F_ext)
            # Solve
            try:
                #L = torch.linalg.cholesky(K)
                # D_sub = torch.cholesky_solve(F_sub.unsqueeze(-1), L).squeeze(-1)
                # disp = torch.zeros(n_nodes * 6, device=K.device)
                D_sub = sparse.linalg.spsolve(K, F_sub)
                disp = np.zeros(n_nodes * 6)

                for i in range(proj_dofs):
                    disp[map_dof_subset2entire[i]] = D_sub[i]
                # Compute compliance for each bar
                bar_compliances = [0.0] * len(self.temp_truss.all_edges)
                for i, K_act in enumerate(K_bar):

                    if design_state[i] > 1e-6:
                        c_e = float(D_sub @ K_act @ D_sub)
                    else:
                        c_e = 0.0

                    bar_compliances[i] = c_e

                bar_compliances = np.array(bar_compliances)
                return disp, True, "", bar_compliances

            except Exception as e:

                raise ValueError("simulation failed!")
        # Not in cache: build new truss, cache K only
        truss = TrussStructure(
            nodes=self.temp_truss.nodes,
            all_edges=self.temp_truss.all_edges,
            design_variables=design_state,
            fixed_nodes=self.temp_truss.fixed_nodes
        )
        self.stiffness_cache[key] = truss.K.copy()
        # Now solve as usual
        return self.solve_elasticity_with_cache(design_state, force_node, force_dir)

    def check_stability(self, design_state, force_node, force_dir):
        key = (tuple(design_state), int(force_node), int(force_dir))
        if key in self.stability_cache:
            return self.stability_cache[key]

        try:
            disp, success, message, bar_compliances = self.solve_elasticity_with_cache(design_state, force_node,
                                                                                       force_dir)
            if not success:
                raise ValueError("simulation failed!")

            temp_compliance = np.array(bar_compliances)
            temp_volume = np.sum(np.array(self.bar_vols) * design_state)

            # Filter compliance: set removed bars (design_state == 0) to zero compliance
            filtered_compliance = temp_compliance.copy()
            filtered_compliance[design_state < 1e-6] = 0.0

            self.compliance_cache[key] = filtered_compliance

            # Check for unstable conditions
            if any(val >= self.compliance_threshold for val in temp_compliance):
                self.stability_cache[key] = -1
                return -1

            if temp_volume <= self.target_volume_lb:
                self.stability_cache[key] = -1
                return -1

            # Check connectivity to force nodes
            bars_connected = False
            for i, edge in enumerate(self.temp_truss.all_edges):
                if design_state[i] > 1e-6 and (force_node in edge):
                    bars_connected = True
                    break
            if not bars_connected:
                self.stability_cache[key] = -1
                return -1

            # If we get here, the state is stable
            self.stability_cache[key] = 1
            return 1

        except Exception as e:
            self.stability_cache[key] = -1
            return -1

    def check_terminate(self, design_states, force_nodes, force_dirs):
        """Check termination condition for batch of states"""
        terminate_flag = np.zeros(len(design_states), dtype=np.int32)

        for idx in range(len(design_states)):
            try:
                design_state = np.asarray(design_states[idx])
                force_node = force_nodes[idx]
                force_dir = force_dirs[idx]
                key = tuple([force_node, force_dir])
                compliance_key = (tuple(design_state), int(force_nodes[idx]), int(force_dirs[idx]))
                disp, success, message, bar_compliances = self.solve_elasticity_with_cache(design_state, force_node,
                                                                                           force_dir)
                if not success:
                    raise ValueError("simulation failed!")

                total_compliance = float(np.sum(bar_compliances))
                temp_volume = np.sum(np.array(self.bar_vols) * design_state)
                filtered_compliance = np.array(bar_compliances)
                filtered_compliance[design_state < 1e-6] = 0.0
                self.compliance_cache[compliance_key] = filtered_compliance

                # Check for unstable states
                if any(val >= self.compliance_threshold for val in bar_compliances):
                    self.stability_cache[compliance_key] = -1
                    continue

                if temp_volume < self.target_volume_lb:
                    self.stability_cache[compliance_key] = -1
                    continue

                # Check connectivity to force node
                bars_connected = False
                for i, edge in enumerate(self.temp_truss.all_edges):
                    if design_state[i] > 1e-6 and (force_node in edge):
                        bars_connected = True
                        break
                if not bars_connected:
                    self.stability_cache[compliance_key] = -1
                    continue

                # Only terminate if compliance is close to optimal and volume is within bounds
                # Use stored optimal compliance (same for all curriculum items)
                if (
                        abs(total_compliance - self.optimal_compliance[key]) <= self.cratio * self.optimal_compliance[
                    key]
                        and self.target_volume_lb <= temp_volume <= self.target_volume_ub
                ):
                    terminate_flag[idx] = 1
                    self.stability_cache[compliance_key] = 1

            except Exception as e:
                raise ValueError("simulation failed!")

        return terminate_flag

    def check_stability_batch(self, design_states, force_nodes, force_dirs):
        """Check stability for batch of states"""
        return np.array([
            self.check_stability(design_states[i], force_nodes[i], force_dirs[i])
            for i in range(len(design_states))
        ], dtype=np.int32)

    def compute_compliance_only(self, design_state, force_node, force_dir):
        design_state = np.asarray(design_state)
        key = (tuple(design_state), int(force_node), int(force_dir))
        if key in self.compliance_cache:
            return self.compliance_cache[key]
        disp, success, message, bar_compliances = self.solve_elasticity_with_cache(design_state, force_node, force_dir)
        if not success:
            raise ValueError("simulation failed!")
        filtered_compliance = np.array(bar_compliances)
        filtered_compliance[design_state < 1e-6] = 0.0
        self.compliance_cache[key] = filtered_compliance
        # print(
        #     f"[DEBUG] Compliance cache size: {len(self.compliance_cache)}, Stiffness cache size: {len(self.stiffness_cache)}")

        return self.compliance_cache[key]

    def get_compliance_batch(self, design_states, force_nodes, force_dirs):
        compliances = []
        for i in range(len(design_states)):
            compliance = self.compute_compliance_only(design_states[i], force_nodes[i], force_dirs[i])
            #compliance = torch.tensor(compliance, device=self.temp_truss.device).unsqueeze(0)
            compliances.append(compliance)
        return compliances

    def step(self, design_states, actions, force_nodes, force_dirs):
        """Environment step function with force parameters"""
        n_batch = len(design_states)
        inds = np.arange(n_batch)

        new_design_states = self.next_states(design_states, actions)
        masks = self.action_masks(design_states)
        valid_action_flag = masks[inds, actions]
        rewards = np.where(valid_action_flag, 0, -1).astype(np.int32)

        if not np.all(valid_action_flag):
            print("Illegal action taken")

        success_flag = np.logical_and(
            self.check_terminate(new_design_states, force_nodes, force_dirs),
            valid_action_flag
        )
        rewards[success_flag] = 1

        stability = self.check_stability_batch(new_design_states, force_nodes, force_dirs)
        rewards = np.where(stability == -1, -1, rewards)

        #print(f"[DEBUG] Rewards: {rewards}, Success flag: {success_flag}, Stability: {stability}")

        return new_design_states, rewards, stability