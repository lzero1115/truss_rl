
import os
import json
import copy
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from time import perf_counter
from types import SimpleNamespace
import random
from src.Truss import TrussStructure
from env import TrussEnv
from ppo import TrussPPO
from buffer import TrussRolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.int32

def load_curriculum_from_folder(curriculum_folder: str) -> List[Dict]:
    """
    Load all curriculum files from the progressive_curriculum folder

    Args:
        curriculum_folder: Path to the folder containing curriculum JSON files

    Returns:
        List of curriculum data dictionaries
    """
    curriculum_folder = Path(curriculum_folder)
    if not curriculum_folder.exists():
        raise ValueError(f"Curriculum folder {curriculum_folder} does not exist")

    curriculum_data = []

    # Check if this is a progressive curriculum structure (level_i folders)
    level_dirs = [d for d in curriculum_folder.iterdir()
                  if d.is_dir() and d.name.startswith("level_")]

    if level_dirs:
        # Progressive curriculum structure - load from level directories
        print(f"Detected progressive curriculum structure with {len(level_dirs)} level directories")

        if not level_dirs:
            raise ValueError(f"No level directories found in {curriculum_folder}")

        print(f"Loading curriculum from {len(level_dirs)} level directories")

        # Load curriculum data from each level
        for level_dir in level_dirs:
            level_id = int(level_dir.name.split("_")[1])
            json_files = list(level_dir.glob("*.json"))

            print(f"Loading level {level_id}: {len(json_files)} curriculum designs")

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        curriculum_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")
                    continue
    else:
        # Legacy curriculum structure - recursively find all JSON files
        print("Detected legacy curriculum structure")
        json_files = list(curriculum_folder.rglob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON curriculum files found in {curriculum_folder}")

        print(f"Found {len(json_files)} curriculum files")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    curriculum_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

    print(f"Successfully loaded {len(curriculum_data)} curriculum items")
    return curriculum_data


def create_truss_from_curriculum(curriculum_item: Dict) -> TrussStructure:
    """
    Create a TrussStructure from curriculum data

    Args:
        curriculum_item: Single curriculum data dictionary

    Returns:
        TrussStructure object
    """
    # Extract truss structure information from curriculum
    nodes = np.array(curriculum_item['node_coordinates'])
    edges = [tuple(edge) for edge in curriculum_item['design_edges']]

    # Initialize with ALL bars available (all 1s) - this is the full design space
    # NOT the optimized_binary_design which is the sparse optimal solution
    design_variables = np.ones(len(edges), dtype=np.float32)

    fixed_nodes = curriculum_item['fixed_nodes']

    # Create truss structure
    truss = TrussStructure(
        nodes=nodes,
        all_edges=edges,
        design_variables=design_variables,
        fixed_nodes=fixed_nodes
    )

    return truss


def compute_accuracy(env, state_dict, settings, deterministic=True):

    ppo_agent = TrussPPO(env.temp_truss, settings)
    
    # # Handle both old format (just policy state_dict) and new format (policy and encoder state_dicts)
    # if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
    #     # New format with separate policy and encoder state_dicts
    #     ppo_agent.policy_old.load_state_dict(state_dict['policy_state_dict'])
    #     ppo_agent.policy.load_state_dict(state_dict['policy_state_dict'])
    #     if 'encoder_state_dict' in state_dict:
    #         ppo_agent.encoder.load_state_dict(state_dict['encoder_state_dict'])
    # else:
    # Old format with just policy state_dict
    ppo_agent.policy_old.load_state_dict(state_dict)
    ppo_agent.policy.load_state_dict(state_dict)
    
    ppo_agent.deterministic = deterministic
    
    ppo_agent.buffer.reset_curriculum(len(env.curriculum))
    # Use accuracy_sample_size parameter instead of training env.num_rollouts
    training_config = settings.get("training", {})
    accuracy_sample_size = training_config.get("accuracy_sample_size", 100)
    num_of_sample = min(accuracy_sample_size, len(env.curriculum))
    ppo_agent.buffer.curriculum_inds = np.random.choice(len(env.curriculum), size=num_of_sample, replace=False)
    ppo_agent.buffer.rewards = training_rollout(ppo_agent,
                                                env,
                                                ppo_agent.buffer.curriculum_inds,
                                                SimpleNamespace(**settings["training"]),
                                                )
    return torch.sum(ppo_agent.buffer.rewards > 0).item() / ppo_agent.buffer.curriculum_inds.shape[0]


def training_rollout(ppo_agent: TrussPPO,
                     env: TrussEnv,
                     curriculum_inds: np.ndarray,
                     training_settings: SimpleNamespace) -> torch.Tensor:
    """
    Run training rollout with optimized GPU utilization
    """

    # Extract curriculum data
    current_design_states = []
    force_nodes = []
    force_dirs = []
    #force_vectors = []
    
    for i, curr_idx in enumerate(curriculum_inds):
        #curr_data = env.curriculum[curr_idx]
        current_design_states.append(env.curriculum_design[curr_idx])
        force_nodes.append(env.curriculum_force_inds[curr_idx])  # Single force node

        force_dirs.append(env.curriculum_force_dir[curr_idx])

    current_design_states = np.array(current_design_states)
    force_nodes = np.array(force_nodes)
    force_dirs = np.array(force_dirs)
    #force_vectors = np.array(force_vectors)

    # Clear buffer for new rollout
    n_env = current_design_states.shape[0]
    ppo_agent.buffer.clear_replay_buffer(n_env)

    # Initialize: all environments are active
    env_inds = np.arange(n_env)
    n_step = 0

    while True:

        # Get current data for active environments
        current_design_states_active = current_design_states[env_inds, :]
        force_nodes_active = force_nodes[env_inds]
        force_dirs_active = force_dirs[env_inds]

        # Get compliance data for active environments
        compliances_active = env.get_compliance_batch(current_design_states_active, force_nodes_active, force_dirs_active)

        # Get action masks for active environments
        masks = env.action_masks(current_design_states_active)

        # Select actions using PPO agent with optimized GPU usage
        current_actions = ppo_agent.select_action(
            current_design_states_active, compliances_active, force_nodes_active, force_dirs_active,
            masks, env_inds
        )

        # Take environment step with force parameters
        next_design_states, rewards, next_stability = env.step(
            current_design_states_active, current_actions, force_nodes_active, force_dirs_active
        )

        # Update design states for active environments
        current_design_states[env_inds, :] = next_design_states

        # Add experience to buffer
        # ppo_agent.buffer.add("next_design_states", next_design_states.tolist(), env_inds)
        ppo_agent.buffer.add("rewards_per_step", rewards, env_inds)
        # ppo_agent.buffer.add("next_stability", next_stability, env_inds)

        # Get valid environment indices
        env_inds, _ = ppo_agent.buffer.get_valid_env_inds()
        n_step += 1

        if len(env_inds) == 0:
            break

    # Get final rewards
    _, rewards = ppo_agent.buffer.get_valid_env_inds()

    return torch.tensor(rewards, dtype=floatType)


def train_truss_optimization(curriculum_folder: str,
                             settings: Dict) -> None:
    """
    Main training function for truss optimization

    Args:
        curriculum_folder: Path to folder containing curriculum JSON files
        settings: Training configuration dictionary
    """

    # Load curriculum from folder
    print("Loading curriculum from folder...")
    curriculum_data = load_curriculum_from_folder(curriculum_folder)
    random.shuffle(curriculum_data)  # Shuffle the curriculum order for robustness

    # Create base truss from first curriculum item (they should all have the same structure)
    base_curriculum = curriculum_data[0]
    base_truss = create_truss_from_curriculum(base_curriculum)

    print(f"Created base truss with {len(base_truss.all_edges)} edges and {len(base_truss.nodes)} nodes")

    # Initialize environment
    env = TrussEnv(base_truss)
    env.set_curriculum(curriculum_data)

    # Initialize PPO agent
    ppo_agent = TrussPPO(base_truss, settings)
    ppo_agent.buffer.reset_curriculum(len(env.curriculum))


    # Training configuration
    training_config = settings.get("training", {})
    max_train_episodes = training_config.get("max_train_episodes", 5000)
    save_delta_accuracy = training_config.get("save_delta_accuracy", 0.01)
    policy_update_batch_size = training_config.get("policy_update_batch_size", 256)
    K_epochs = training_config.get("K_epochs", 5)
    num_rollouts = training_config.get("num_rollouts", 128)
    policy_name = training_config.get("policy_name", "truss_policy")
    terminate_nondeterministic_accuracy = training_config.get("terminate_nondeterministic_accuracy", 0.9)
    terminate_deterministic_accuracy = training_config.get("terminate_deterministic_accuracy", 0.98)
    save_epochs = training_config.get("save_epochs", 10)
    # accuracy_sample_size = training_config.get("accuracy_sample_size", 100)
    


    # Create save directory
    folder_path = f"./models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    saved_model_path = f"{folder_path}/{policy_name}.pol"

    print(f"Starting training with {len(curriculum_data)} curriculum items")
    print(f"Training configuration:")
    print(f"  - Max episodes: {max_train_episodes}")
    print(f"  - Rollouts per episode: {num_rollouts}")
    print(f"  - Policy update batch size: {policy_update_batch_size}")

    while ppo_agent.episode <= max_train_episodes:
        start_time = perf_counter()

        # Sample curriculum and run rollout
        curriculum_inds, sample_new_curriculum = ppo_agent.buffer.sample_curriculum(num_rollouts)
        ppo_agent.buffer.curriculum_inds = curriculum_inds
        rewards = training_rollout(ppo_agent, env, curriculum_inds, SimpleNamespace(**training_config))
        ppo_agent.buffer.rewards = rewards

        # Update entropy weights
        inds = curriculum_inds.to(device)
        success_inds, failed_inds = inds[(rewards > 0)], inds[(rewards < 0)]
        ppo_agent.buffer.num_success[success_inds] += 1
        ppo_agent.buffer.num_failed[success_inds] = 0
        ppo_agent.buffer.num_failed[failed_inds] = ppo_agent.buffer.num_failed[failed_inds] + 1
        ppo_agent.buffer.modify_entropy()

        # Calculate accuracy (randomly sampled, controlled by env.num_rollout)
        accuracy_of_sample_curriculum = (torch.sum(rewards > 0) / len(rewards)).item()
        accuracy_of_entire_curriculum = None
        accuracy_of_entire_curriculum_deterministic = None

        # Save policy if improved (following disassembly training logic)
        if ((ppo_agent.episode + 1) % save_epochs == 0 and not sample_new_curriculum):

            # Evaluate both non-deterministic and deterministic accuracy on sampled curriculum
            accuracy_of_entire_curriculum = compute_accuracy(env, ppo_agent.policy_old.state_dict(), settings, deterministic=False)
            accuracy_of_entire_curriculum_deterministic = compute_accuracy(env, ppo_agent.policy_old.state_dict(), settings, deterministic=True)
            
            # Save if non-deterministic accuracy improved (following disassembly logic)
            if accuracy_of_entire_curriculum > ppo_agent.accuracy_of_entire_curriculum:
                ppo_agent.accuracy_of_entire_curriculum = accuracy_of_entire_curriculum
                ppo_agent.save(saved_model_path, settings)

        # Policy update
        loss, sur_loss, val_loss, entropy = ppo_agent.update_policy(
            batch_size=policy_update_batch_size,
            update_iter=K_epochs)

        # Print status
        elapsed = perf_counter() - start_time
        print("")
        print("Episode : {} \t\t Accuracy : {:.2f}\t\t Time : {:.2f}".format(
            ppo_agent.episode, accuracy_of_sample_curriculum, elapsed))
        print("Loss : {:.2e} \t\t Sur: {:.2e} \t\t Val: {:.2e} \t\t Entropy : {:.2e}".format(
            loss, sur_loss, val_loss, entropy))
        

        if accuracy_of_entire_curriculum is not None:
            print("Non-deter. Acc:\t {:.2f}, \t\t Deter. Acc:\t {:.2f}".format(accuracy_of_entire_curriculum, accuracy_of_entire_curriculum_deterministic))

            
            # Exit if both accuracies meet termination thresholds (like disassembly training)
            if (accuracy_of_entire_curriculum > terminate_nondeterministic_accuracy
                and accuracy_of_entire_curriculum_deterministic > terminate_deterministic_accuracy):
                break

        ppo_agent.episode = ppo_agent.episode + 1


