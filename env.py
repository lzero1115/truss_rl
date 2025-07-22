import copy
import numpy as np
from src.Truss import TrussStructure
from typing import List, Tuple, Optional, Dict


class TrussEnv:
    def __init__(self, truss: TrussStructure, volume_ratio: float = 0.5):
        self.temp_truss = copy.deepcopy(truss)
        self.volume_ratio = volume_ratio
        self.ub = 1.2
        self.lb = 0.7
        self.target_volume_ub = self.volume_ratio * self.ub * self.temp_truss.get_truss_volume()
        self.target_volume_lb = self.volume_ratio * self.lb * self.temp_truss.get_truss_volume()
        self.n_bars = len(self.temp_truss.all_edges)
        self.compliance_threshold = 1.0  # manually set
        self.cratio = 0.2
        self.directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        
        # Stability cache: key = (tuple(design_state)), value = stability
        # self.stability_cache = {}
        # Compliance cache: key = (tuple(design_state)), value = per-bar compliance array
        # self.compliance_cache = {}
        #self.num_rollouts = 64
        
        # Curriculum management - will be set when curriculum is loaded
        self.curriculum = None
        self.optimal_compliance = {}

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

    # def encode_state(self, state):
    #     # Round to integers to avoid numerical errors in cache keys
    #     # Since we have single load case, only use design state as cache key
    #     state_tuple = tuple(int(round(x)) for x in state)
    #     return state_tuple

    # def add_stability_history(self, design_states, stable_flags):
    #     for ind in range(len(design_states)):
    #         state_encode = self.encode_state(design_states[ind])
    #         if state_encode not in self.stability_cache:
    #             self.stability_cache[state_encode] = stable_flags[ind]
    #
    # def add_compliance_history(self, design_states, compliance_arrays):
    #     """Store compliance data in cache"""
    #     for ind in range(len(design_states)):
    #         state_encode = self.encode_state(design_states[ind])
    #         if state_encode not in self.compliance_cache:
    #             self.compliance_cache[state_encode] = np.array(compliance_arrays[ind])

    # def get_compliance_history(self, design_state):
    #     """Get cached compliance data for a design state"""
    #     key = self.encode_state(design_state)
    #     if key in self.compliance_cache:
    #         return self.compliance_cache[key]
    #     return None  # unknown

    # def cache_curriculum_compliance(self, design_state, curriculum_idx):
    #     """
    #     Compute and cache compliance for curriculum states (known to be stable)
    #     More efficient than check_stability since we skip stability validation
    #     """
    #     try:
    #         # Convert to numpy array if needed
    #         if not isinstance(design_state, np.ndarray):
    #             design_state = np.array(design_state, dtype=np.float32)
    #
    #         # Check if already cached - avoid unnecessary computation
    #         cached_compliance = self.get_compliance_history(design_state)
    #         if cached_compliance is not None:
    #             self.stability_cache[self.encode_state(design_state)] = 1  # Mark as stable
    #             return True
    #
    #         temp_truss = copy.deepcopy(self.temp_truss)
    #
    #         # Get force info from curriculum
    #         force_node = self.curriculum_force_inds[curriculum_idx][0]
    #         force = self.curriculum_force[curriculum_idx]
    #
    #         ext_force = temp_truss.create_external_force_vector(force_indices=[force_node], force_vectors=force)
    #         temp_truss.update_bars_with_weight(design_state)
    #         displacement, success, message = temp_truss.solve_elasticity(ext_force)
    #
    #         if not success:
    #             self.stability_cache[self.encode_state(design_state)] = -1
    #             self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
    #             return False
    #
    #         temp_compliance = temp_truss.bar_compliances # per bar compliance
    #
    #         # Filter compliance: set removed bars (design_state == 0) to zero compliance
    #         filtered_compliance = np.array(temp_compliance)
    #         filtered_compliance[design_state < 1e-6] = 0.0
    #
    #         # Cache the compliance data and mark as stable
    #         self.add_compliance_history([design_state], [filtered_compliance])
    #         self.stability_cache[self.encode_state(design_state)] = 1  # Known stable
    #         return True
    #
    #     except Exception as e:
    #         self.stability_cache[self.encode_state(design_state)] = -1
    #         self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
    #         return False

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


    def action_masks(self, design_states):
        masks = (design_states > 1e-6).astype(np.int32)
        return masks

    def check_stability(self, design_state, force_node, force_dir):

        try:
            # # Check both stability and compliance cache first
            # stability_cached = self.get_history(design_state)
            # compliance_cached = self.get_compliance_history(design_state)
            #
            # # If both are cached, return early
            # if stability_cached != 0 and compliance_cached is not None:
            #     return stability_cached

            temp_truss = copy.deepcopy(self.temp_truss)
            theta = self.directions[force_dir]
            fx, fy = np.cos(theta), np.sin(theta)
            force_vector = [fx * self.force_amplitude, fy * self.force_amplitude, 0.0]
            
            ext_force = temp_truss.create_external_force_vector(force_indices=[force_node], force_vectors=[force_vector])
            temp_truss.update_bars_with_weight(design_state)
            displacement, success, message = temp_truss.solve_elasticity(ext_force)

            if not success:
                # self.stability_cache[self.encode_state(design_state)] = -1
                # # Cache zero compliance for failed states (no valid physics)
                # self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
                raise ValueError("simulation failed!")

            temp_compliance = temp_truss.bar_compliances # per bar compliance
            temp_volume = temp_truss.get_truss_volume()

            # Filter compliance: set removed bars (design_state == 0) to zero compliance
            filtered_compliance = np.array(temp_compliance)
            filtered_compliance[design_state < 1e-6] = 0.0

            # # Cache the filtered compliance data regardless of stability outcome
            # self.add_compliance_history([design_state], [filtered_compliance])

            # Check for unstable conditions
            if any(val >= self.compliance_threshold for val in temp_compliance):
                #self.stability_cache[self.encode_state(design_state)] = -1
                return -1

            if temp_volume <= self.target_volume_lb:
                #self.stability_cache[self.encode_state(design_state)] = -1
                return -1

            # Check connectivity to force nodes
            bars_connected = False
            for i, edge in enumerate(temp_truss.all_edges):
                if design_state[i] > 1e-6 and (force_node in edge):
                    bars_connected = True
                    break
            if not bars_connected:
                #self.stability_cache[self.encode_state(design_state)] = -1
                return -1

            # If we get here, the state is stable
            # self.stability_cache[self.encode_state(design_state)] = 1
            return 1

        except Exception as e:
            # self.stability_cache[self.encode_state(design_state)] = -1
            # Cache zero compliance for error states (no valid physics)
            # self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
            return -1

    def check_terminate(self, design_states, force_nodes, force_dirs):
        """Check termination condition for batch of states"""
        terminate_flag = np.zeros(len(design_states), dtype=np.int32)
        
        for idx in range(len(design_states)):
            try:
                temp_truss = copy.deepcopy(self.temp_truss)
                
                force_node = force_nodes[idx]
                force_dir = force_dirs[idx]
                #directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                theta = self.directions[force_dir]
                fx, fy = np.cos(theta), np.sin(theta)
                key = tuple([force_node, force_dir])
                force_vector = [fx * self.force_amplitude, fy * self.force_amplitude, 0.0]


                ext_force = temp_truss.create_external_force_vector(force_indices=[force_node], force_vectors=[force_vector])
                temp_truss.update_bars_with_weight(design_states[idx])
                displacement, success, message = temp_truss.solve_elasticity(ext_force)

                if not success:

                    raise ValueError("simulation failed!")

                total_compliance = sum(temp_truss.bar_compliances)
                temp_volume = temp_truss.get_truss_volume()

                # Check for unstable states
                if any(val >= self.compliance_threshold for val in temp_truss.bar_compliances):
                    # self.add_stability_history([design_states[idx]], [-1])
                    continue
                    
                if temp_volume < self.target_volume_lb:
                    # self.add_stability_history([design_states[idx]], [-1])
                    continue
                    
                # Check connectivity to force node
                bars_connected = False
                for i, edge in enumerate(temp_truss.all_edges):
                    if design_states[idx][i] > 1e-6 and (force_node in edge):
                        bars_connected = True
                        break
                if not bars_connected:
                    # self.add_stability_history([design_states[idx]], [-1])
                    continue

                # Only terminate if compliance is close to optimal and volume is within bounds
                # Use stored optimal compliance (same for all curriculum items)
                if (
                        abs(total_compliance - self.optimal_compliance[key]) <= self.cratio * self.optimal_compliance[key]
                        and self.target_volume_lb <= temp_volume <= self.target_volume_ub
                ):
                    terminate_flag[idx] = 1
                    # Cache this good state
                    # self.add_stability_history([design_states[idx]], [1])

            except Exception as e:
                raise ValueError("simulation failed!")

        return terminate_flag

    # def get_history(self, design_state):
    #     key = self.encode_state(design_state)
    #     if key in self.stability_cache:
    #         return self.stability_cache[key]
    #     return 0  # unknown

    def check_stability_batch(self, design_states, force_nodes, force_dirs):
        """Check stability for batch of states"""
        return np.array([
            self.check_stability(design_states[i], force_nodes[i], force_dirs[i])
            for i in range(len(design_states))
        ], dtype=np.int32)

    def compute_compliance_only(self, design_state, force_node, force_dir):

        try:
            temp_truss = copy.deepcopy(self.temp_truss)

            #directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            theta = self.directions[force_dir]
            fx, fy = np.cos(theta), np.sin(theta)
            force_vector = [fx * self.force_amplitude, fy * self.force_amplitude, 0.0]
            
            ext_force = temp_truss.create_external_force_vector(force_indices=[force_node], force_vectors=[force_vector])
            temp_truss.update_bars_with_weight(design_state)
            displacement, success, message = temp_truss.solve_elasticity(ext_force)

            if not success:
                # Cache zero compliance for failed states
                # self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
                raise ValueError("simulation failed!")

            temp_compliance = temp_truss.bar_compliances
            
            # Filter compliance: set removed bars (design_state == 0) to zero compliance
            filtered_compliance = np.array(temp_compliance)
            filtered_compliance[design_state < 1e-6] = 0.0

            # Cache the filtered compliance data
            # self.add_compliance_history([design_state], [filtered_compliance])
            return filtered_compliance

        except Exception as e:
            # Cache zero compliance for error states
            # self.add_compliance_history([design_state], [np.zeros(self.n_bars)])
            raise ValueError("simulation failed!")

    def get_compliance_batch(self, design_states, force_nodes, force_dirs):
        """Get compliance data for a batch of states"""
        compliances = []
        for i in range(len(design_states)):
            compliance = self.compute_compliance_only(design_states[i], force_nodes[i], force_dirs[i])
            compliance = np.asarray(compliance).reshape(1, -1)  # make sure it's 2D
            compliances.append(compliance)

        return np.vstack(compliances)

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

        return new_design_states, rewards, stability