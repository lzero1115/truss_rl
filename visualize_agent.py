import torch
import numpy as np
import json
import polyscope as ps
import polyscope.imgui as psim
from typing import Dict, List, Optional
from src.Truss import TrussStructure
from env import TrussEnv
from policy_global import MLPActorCritic
import pickle
from pathlib import Path


class TrussVisualizer:
    def __init__(self, checkpoint_path: str, curriculum_json_path: str):
        """
        Initialize the truss visualizer with a trained policy and curriculum

        Args:
            checkpoint_path: Path to the trained policy checkpoint
            curriculum_json_path: Path to the curriculum JSON file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.floatType = torch.float32
        self.intType = torch.long

        # Initialize truss objects as None first
        self.truss = None
        self.temp_truss = None
        self.optimal_truss = None
        self.policy = None

        # Load curriculum first to create truss objects
        self.load_curriculum(curriculum_json_path)

        # Load checkpoint and initialize policy with the truss
        self.load_checkpoint(checkpoint_path)

        # Initialize environment
        if self.truss is not None and self.temp_truss is not None and self.optimal_truss is not None and self.policy is not None:
            # Create environment with complete truss (for physical simulation and termination logic)
            self.env = TrussEnv(self.truss)

            # Set single curriculum item
            print(f"Setting curriculum with {len([self.curriculum_item])} items")
            print(f"Curriculum item keys: {list(self.curriculum_item.keys())}")
            self.env.set_curriculum([self.curriculum_item])
            print("Environment curriculum initialized")
        else:
            raise ValueError(
                "Truss, temp_truss, optimal_truss and policy must be initialized before creating environment")

        # Color scheme
        self.temp_color = [0.2, 0.2, 0.8]  # Blue
        self.optimal_color = [0.2, 0.8, 0.2]  # Green

        # Initialize visualization
        self.init_visualization()

        # State tracking
        self.current_step = 0
        self.deterministic_mode = False
        self.episode_reward = 0
        self.episode_actions = []
        self.episode_compliances = []
        self.terminated = False

        # Design state tracking
        self.design = self.curriculum_design.copy()

    def load_checkpoint(self, checkpoint_path: str):
        """Load the trained policy from checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Extract settings and policy weights
        self.settings = checkpoint.get('settings', {})
        self.policy_state_dict = checkpoint.get('policy_state_dict')

        # Initialize policy with the truss that was created from curriculum
        if self.truss is not None:
            ppo_config = self.settings.get("ppo", {})
            obs_dim = ppo_config.get("obs_dim", 256)
            hidden_sizes = ppo_config.get("hidden_sizes", (512, 512, 256))

            self.policy = MLPActorCritic(
                truss=self.truss,
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes
            ).to(self.device)

            if self.policy_state_dict is not None:
                self.policy.load_state_dict(self.policy_state_dict)
            else:
                raise ValueError("Checkpoint must contain 'policy_state_dict'")

            self.policy.eval()
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"   - Policy weights loaded")
        else:
            raise ValueError("Truss must be initialized before loading checkpoint")

    def load_curriculum(self, curriculum_json_path: str):
        """Load curriculum data from JSON file"""
        with open(curriculum_json_path, 'r') as f:
            curriculum_data = json.load(f)

        # Get the curriculum item (handle different formats)
        if isinstance(curriculum_data, list):
            curriculum_item = curriculum_data[0]
        else:
            curriculum_item = curriculum_data

        # Extract essential information
        nodes = np.array(curriculum_item['node_coordinates'])
        edges = [tuple(edge) for edge in curriculum_item['design_edges']]
        fixed_nodes = curriculum_item.get('fixed_nodes', [])

        # Get curriculum design variables
        curriculum_design = np.array(curriculum_item['curriculum_design_variables'], dtype=np.float32)

        # Get optimized binary design for comparison
        optimal_design = np.array(curriculum_item['optimized_binary_design'], dtype=np.float32)

        # Create complete truss for policy (all bars present)
        self.truss = TrussStructure(
            nodes=nodes,
            all_edges=edges,
            design_variables=np.ones(len(edges), dtype=np.float32),  # All bars present
            fixed_nodes=fixed_nodes
        )

        # Create temporary truss with curriculum design variables
        self.temp_truss = TrussStructure(
            nodes=nodes,
            all_edges=edges,
            design_variables=curriculum_design,  # Curriculum-specific design
            fixed_nodes=fixed_nodes
        )

        # Create optimal truss with optimized binary design
        self.optimal_truss = TrussStructure(
            nodes=nodes,
            all_edges=edges,
            design_variables=optimal_design,  # Optimized binary design
            fixed_nodes=fixed_nodes
        )

        # Store curriculum data
        self.curriculum_item = curriculum_item
        self.curriculum_design = curriculum_design
        self.optimal_design = optimal_design
        self.force_node_idx = curriculum_item['force_node_indices'][0]
        self.force_dir = curriculum_item['direction_index']

        print(f"Loaded curriculum from {curriculum_json_path}")
        print(f"   - Nodes: {len(nodes)}, Edges: {len(edges)}")
        print(f"   - Curriculum design: {np.sum(curriculum_design > 0.5)}/{len(curriculum_design)} active bars")
        print(f"   - Optimal design: {np.sum(optimal_design > 0.5)}/{len(optimal_design)} active bars")
        print(f"   - Force node: {self.force_node_idx}")
        print(f"   - Optimal compliance: {curriculum_item.get('optimal_compliance', 'N/A')}")

    def init_visualization(self):
        """Initialize Polyscope visualization"""
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        # Register the truss meshes
        self.register_truss_mesh()

        # Set callback for UI
        ps.set_user_callback(self.ui_callback)

        print("Initialized Polyscope visualization")

    def register_truss_mesh(self):
        """Register the temp and optimal truss meshes"""
        # Get static mesh from temp truss
        Vs_temp, Fs_temp = self.temp_truss.get_bar_geometry()

        # Get static mesh from optimal truss
        Vs_optimal, Fs_optimal = self.optimal_truss.get_bar_geometry()

        if not Vs_temp or not Fs_temp:
            raise ValueError("No bars present in temp truss - cannot create visualization")

        # Create temp truss mesh
        all_vertices_temp = []
        all_faces_temp = []
        vertex_offset = 0

        for V, F in zip(Vs_temp, Fs_temp):
            all_vertices_temp.append(V)
            F_adjusted = F + vertex_offset
            all_faces_temp.append(F_adjusted)
            vertex_offset += len(V)

        if all_vertices_temp:
            combined_vertices_temp = np.vstack(all_vertices_temp)
            combined_faces_temp = np.vstack(all_faces_temp)
            
            # Register temp truss mesh
            self.temp_truss_mesh = ps.register_surface_mesh(
                "Temp Truss", 
                combined_vertices_temp, 
                combined_faces_temp
            )
            
            # Store individual bar meshes for temp truss
            self.temp_bar_meshes = list(zip(Vs_temp, Fs_temp))
            
            # Set initial colors for temp truss
            face_colors_temp = []
            for i, (V, F) in enumerate(self.temp_bar_meshes):
                # All bars in temp truss mesh are active (blue)
                bar_color = self.temp_color
                
                # Apply color to all faces of this bar
                for _ in range(len(F)):
                    face_colors_temp.append(bar_color)
            
            # Set temp truss colors
            if face_colors_temp:
                face_colors_temp = np.array(face_colors_temp)
                self.temp_truss_mesh.add_color_quantity("Bar States", face_colors_temp, defined_on='faces',
                                                        enabled=True)
            
            # Add force vector visualization
            self.register_force_vector()

        # Create optimal truss mesh with z offset
        all_vertices_optimal = []
        all_faces_optimal = []
        vertex_offset = 0

        for V, F in zip(Vs_optimal, Fs_optimal):
            # Apply z offset of -1.5
            V_offset = V.copy()
            V_offset[:, 2] -= 1.5
            all_vertices_optimal.append(V_offset)
            F_adjusted = F + vertex_offset
            all_faces_optimal.append(F_adjusted)
            vertex_offset += len(V)

        if all_vertices_optimal:
            combined_vertices_optimal = np.vstack(all_vertices_optimal)
            combined_faces_optimal = np.vstack(all_faces_optimal)

            # Register optimal truss mesh
            self.optimal_truss_mesh = ps.register_surface_mesh(
                "Optimal Truss",
                combined_vertices_optimal,
                combined_faces_optimal
            )

            # Store individual bar meshes for optimal truss
            self.optimal_bar_meshes = list(zip(Vs_optimal, Fs_optimal))

            # Set colors for optimal truss (always green)
            face_colors_optimal = []
            for i, (V, F) in enumerate(self.optimal_bar_meshes):
                # Optimal design is always shown in green
                bar_color = self.optimal_color

                # Apply color to all faces of this bar
                for _ in range(len(F)):
                    face_colors_optimal.append(bar_color)

            # Set optimal truss colors
            if face_colors_optimal:
                face_colors_optimal = np.array(face_colors_optimal)
                self.optimal_truss_mesh.add_color_quantity("Optimal Design", face_colors_optimal, defined_on='faces',
                                                           enabled=True)

    def register_force_vector(self):
        """Register force vector visualization on the force node"""
        # Get force node position
        force_node_pos = self.temp_truss.nodes[self.force_node_idx]
        
        # Create single point cloud for force node and vector
        self.force_cloud = ps.register_point_cloud(
            "Force Node",
            np.array([force_node_pos])
        )
        
        # Set point cloud radius for better visibility
        self.force_cloud.set_radius(0.005)
        
        # Set force node color (red for force application point)
        self.force_cloud.add_color_quantity(
            "Force Node", 
            np.array([[1.0, 0.0, 0.0]]),  # Red color
            enabled=True
        )
        
        # Calculate force vector direction
        directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        theta = directions[self.force_dir]
        fx, fy = np.cos(theta), np.sin(theta)
        force_vector = np.array([fx, fy, 0.0]) * 0.5  # Scale for visualization
        
        # Add vector quantity to the same point cloud
        self.force_cloud.add_vector_quantity(
            "Applied Force",
            np.array([force_vector]),
            length=0.3,  # Vector length
            radius=0.002,  # Vector radius
            color=(1.0, 0.0, 0.0),  # Red color
            enabled=True
        )
        
        print(f"Force vector registered at node {self.force_node_idx} with direction {self.force_dir}")

    def update_truss_visualization(self):
        """Update the temp truss mesh based on current design state"""
        if not hasattr(self, 'temp_truss_mesh') or self.temp_truss_mesh is None:
            raise ValueError("Temp truss mesh not initialized - cannot update visualization")

        # Update temp truss design variables
        self.temp_truss.update_bars_with_weight(self.design)

        # Get updated mesh geometry
        Vs_temp, Fs_temp = self.temp_truss.get_bar_geometry()

        # Recreate temp truss mesh with new geometry
        if Vs_temp and Fs_temp:
            # Combine all bar meshes into single mesh
            all_vertices_temp = []
            all_faces_temp = []
            vertex_offset = 0

            for V, F in zip(Vs_temp, Fs_temp):
                all_vertices_temp.append(V)
                F_adjusted = F + vertex_offset
                all_faces_temp.append(F_adjusted)
                vertex_offset += len(V)

            if all_vertices_temp:
                combined_vertices_temp = np.vstack(all_vertices_temp)
                combined_faces_temp = np.vstack(all_faces_temp)

                # Recreate temp truss mesh with new geometry
                ps.remove_surface_mesh("Temp Truss")
                self.temp_truss_mesh = ps.register_surface_mesh(
                    "Temp Truss",
                    combined_vertices_temp,
                    combined_faces_temp
                )

                # Update colors
                face_colors_temp = []
                for i, (V, F) in enumerate(zip(Vs_temp, Fs_temp)):
                    bar_color = self.temp_color
                    for _ in range(len(F)):
                        face_colors_temp.append(bar_color)

                if face_colors_temp:
                    face_colors_temp = np.array(face_colors_temp)
                    self.temp_truss_mesh.add_color_quantity("Bar States", face_colors_temp, defined_on='faces',
                                                            enabled=True)

    def step_agent(self):
        """Execute one step of the agent"""
        # Check if episode is terminated
        terminate_flag = self.env.check_terminate([self.design], [self.force_node_idx], [self.force_dir])
        if terminate_flag[0] == 1:
            print("Episode already terminated! Cannot step further.")
            return

        # Get current design state and force info from curriculum
        design_state = torch.tensor(self.design, dtype=self.floatType, device=self.device).unsqueeze(0)
        force_node = torch.tensor([self.force_node_idx], dtype=self.intType, device=self.device)
        force_dir = torch.tensor([self.force_dir], dtype=self.intType, device=self.device)

        # Get compliance from physical simulation
        compliance = self.env.compute_compliance_only(self.design, self.force_node_idx, self.force_dir)
        compliance_tensor = torch.tensor(compliance, dtype=self.floatType, device=self.device).unsqueeze(0)

        # Get action mask from environment
        action_mask = self.env.action_masks([self.design])

        # Get action from policy
        with torch.no_grad():
            action, action_logprob, state_val = self.policy.act(
                design_state, compliance_tensor, force_node, force_dir,
                mask=action_mask, deterministic=self.deterministic_mode
            )

        # Execute action
        action_np = action.cpu().numpy()[0]

        # Update design state based on action (remove bar)
        if 0 <= action_np < len(self.design):
            if self.design[action_np] == 0:
                raise ValueError(f"Invalid action: bar {action_np} is already removed")
            self.design[action_np] = 0.0  # Remove bar

        # Check termination after action
        terminate_flag = self.env.check_terminate([self.design], [self.force_node_idx], [self.force_dir])
        self.terminated = terminate_flag[0] == 1

        # Get reward (simplified - could be enhanced)
        reward = 1.0 if self.terminated else 0.0

        # Update tracking
        self.current_step += 1
        self.episode_reward += reward
        self.episode_actions.append(action_np)
        self.episode_compliances.append(compliance)

        # Update visualization
        self.update_truss_visualization()

        # Print step info
        print(f"Step {self.current_step}: Action={action_np}, Reward={reward:.4f}, "
              f"Compliance={np.sum(compliance):.6f}, Terminated={self.terminated}")

        if self.terminated:
            print(f"   Episode finished! Total reward: {self.episode_reward:.4f}")
            print(f"   Final compliance: {np.sum(compliance):.6f}")
            print(f"   Optimal compliance: {self.curriculum_item['optimal_compliance']:.6f}")

            # Check if optimization succeeded
            final_compliance = np.sum(compliance)
            optimal_compliance = self.curriculum_item['optimal_compliance']

            # Success criteria: final compliance close to optimal (within 10%)
            if final_compliance <= optimal_compliance * 1.1:
                print("Optimization SUCCEEDED! Agent found good solution.")
            else:
                print("Optimization FAILED! Agent did not find good solution.")

            # Note: Design accuracy removed - focus on compliance comparison

    def reset_episode(self):
        """Reset the episode"""
        self.current_step = 0
        self.episode_reward = 0
        self.episode_actions = []
        self.episode_compliances = []
        self.terminated = False

        # Reset design state to curriculum design
        self.design = self.curriculum_design.copy()

        # Update visualization
        self.update_truss_visualization()

        print("Episode reset!")

    def ui_callback(self):
        """UI callback function using psim"""
        # Deterministic mode toggle
        changed, self.deterministic_mode = psim.Checkbox(
            "Deterministic Mode",
            self.deterministic_mode
        )
        if changed:
            print(f"Switched to {'deterministic' if self.deterministic_mode else 'non-deterministic'} mode")

        psim.Separator()

        # Step agent button (disabled if terminated)
        if self.terminated:
            psim.TextUnformatted("Episode terminated - cannot step further")
            if psim.Button("Step Agent (Disabled)"):
                pass  # Button disabled when terminated
        else:
            if psim.Button("Step Agent"):
                self.step_agent()

        # Reset episode button
        if psim.Button("Reset Episode"):
            self.reset_episode()

        psim.Separator()

        # Display current state information
        psim.TextUnformatted(f"Step: {self.current_step}")
        psim.TextUnformatted(f"Episode Reward: {self.episode_reward:.4f}")
        psim.TextUnformatted(f"Status: {'Terminated' if self.terminated else 'Active'}")

        # Display curriculum information
        if hasattr(self, 'curriculum_item'):
            optimal_compliance = self.curriculum_item.get('optimal_compliance', 'N/A')
            psim.TextUnformatted(f"Optimal Compliance: {optimal_compliance}")
            psim.TextUnformatted(f"Force Node: {self.force_node_idx}")


def main():
    """Main function to run the visualization"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize truss optimization agent')
    parser.add_argument('--checkpoint', type=str, default='models_simple/truss_progressive_policy.pol',
                        help='Path to the trained policy checkpoint (default: checkpoint.pol)')
    parser.add_argument('--curriculum', type=str, default='easy_test/node2_dir6_v0.50.json',
                        help='Path to the curriculum JSON file (default: curriculum.json)')

    args = parser.parse_args()

    # Create visualizer
    visualizer = TrussVisualizer(args.checkpoint, args.curriculum)

    # Show the visualization
    print("Starting truss agent visualization...")
    print("Controls:")
    print("  - Use the UI panel to control the agent")
    print("  - Toggle deterministic mode for consistent behavior")
    print("  - Step through actions manually")
    print("  - Reset episode to start fresh")

    # Polyscope will handle the main loop automatically
    ps.show()


if __name__ == "__main__":
    main()