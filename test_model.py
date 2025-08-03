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
import pickle
import argparse
from src.Truss import TrussStructure
from env import TrussEnv
from policy_global import MLPActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float32
intType = torch.long

def load_curriculum_from_folder(curriculum_folder: str) -> List[Dict]:
    """
    Load all curriculum files from the curriculum folder

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
    fixed_nodes = curriculum_item.get('fixed_nodes', [])

    # Create truss with all bars present (for policy)
    design_variables = np.ones(len(edges), dtype=np.float32)

    truss = TrussStructure(
        nodes=nodes,
        all_edges=edges,
        design_variables=design_variables,
        fixed_nodes=fixed_nodes
    )

    return truss


def load_trained_model(checkpoint_path: str, truss: TrussStructure) -> MLPActorCritic:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to the trained model checkpoint
        truss: TrussStructure object for the model

    Returns:
        Loaded MLPActorCritic model
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Extract settings and policy weights
    settings = checkpoint.get('settings', {})
    policy_state_dict = checkpoint.get('policy_state_dict')

    if policy_state_dict is None:
        raise ValueError("Checkpoint must contain 'policy_state_dict'")

    # Initialize policy with the truss
    ppo_config = settings.get("ppo", {})
    obs_dim = ppo_config.get("obs_dim", 256)
    hidden_sizes = ppo_config.get("hidden_sizes", (512, 512, 256))

    policy = MLPActorCritic(
        truss=truss,
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes
    ).to(device)

    policy.load_state_dict(policy_state_dict)
    policy.eval()

    print(f"Loaded model from {checkpoint_path}")
    return policy


def test_single_curriculum_item(env: TrussEnv, policy: MLPActorCritic, 
                               curriculum_idx: int, deterministic: bool = True) -> bool:
    """
    Test a single curriculum item and return success/failure

    Args:
        env: TrussEnv environment
        policy: Trained policy
        curriculum_idx: Index of curriculum item to test
        deterministic: Whether to use deterministic actions

    Returns:
        True if successful, False otherwise
    """
    # Get curriculum data
    initial_design = env.curriculum_design[curriculum_idx]
    force_node = env.curriculum_force_inds[curriculum_idx]
    force_dir = env.curriculum_force_dir[curriculum_idx]

    # Initialize episode
    current_design = initial_design.copy()
    step_count = 0
    max_steps = len(current_design)  # Maximum steps is number of bars

    while step_count < max_steps:
        # Check termination using environment's logic
        terminate_flag = env.check_terminate([current_design], [force_node], [force_dir])
        if terminate_flag[0] == 1:
            return True  # Successfully terminated
        elif terminate_flag[0] == -1:
            return False  # Failed (unstable, invalid, etc.)

        # Prepare inputs for policy
        design_state = torch.tensor(current_design, dtype=floatType, device=device).unsqueeze(0)
        force_node_tensor = torch.tensor([force_node], dtype=intType, device=device)
        force_dir_tensor = torch.tensor([force_dir], dtype=intType, device=device)

        # Get compliance
        compliance = env.compute_compliance_only(current_design, force_node, force_dir)
        compliance_tensor = torch.tensor(compliance, dtype=floatType, device=device).unsqueeze(0)

        # Get action mask
        action_mask = env.action_masks([current_design])

        # Get action from policy
        with torch.no_grad():
            action, action_logprob, state_val = policy.act(
                design_state, compliance_tensor, force_node_tensor, force_dir_tensor,
                mask=action_mask, deterministic=deterministic
            )

        # Execute action using environment step
        action_np = action.cpu().numpy()[0]
        
        # Use environment step to get proper reward
        next_design_states, rewards, stability = env.step(
            np.array([current_design]), np.array([action_np]), np.array([force_node]), np.array([force_dir])
        )
        
        reward = rewards[0]
        
        # Check if we got a success reward (+1)
        if reward == 1:
            return True  # Successfully terminated
        
        # Check if we got a failure reward (-1)
        if reward == -1:
            return False  # Failed (invalid action, unstable, etc.)
        
        # Update design state for next iteration
        current_design = next_design_states[0]
        step_count += 1

    # If we reach here, the episode didn't terminate successfully
    return False


def test_model_on_curriculum(checkpoint_path: str, curriculum_folder: str, 
                           deterministic: bool = True) -> float:
    """
    Test trained model on all curriculum items and return accuracy

    Args:
        checkpoint_path: Path to trained model checkpoint
        curriculum_folder: Path to curriculum folder
        deterministic: Whether to use deterministic actions

    Returns:
        Accuracy (success rate) as float
    """
    print(f"Loading curriculum from {curriculum_folder}")
    curriculum_data = load_curriculum_from_folder(curriculum_folder)
    print(f"Found {len(curriculum_data)} curriculum items to test")

    # Create base truss from first curriculum item
    base_curriculum = curriculum_data[0]
    base_truss = create_truss_from_curriculum(base_curriculum)
    print(f"Created base truss with {len(base_truss.all_edges)} edges and {len(base_truss.nodes)} nodes")

    # Initialize environment
    env = TrussEnv(base_truss)
    env.set_curriculum(curriculum_data)

    # Load trained model
    policy = load_trained_model(checkpoint_path, base_truss)

    # Test all curriculum items
    print(f"Testing model on {len(curriculum_data)} curriculum items...")
    successful_tests = 0
    
    for i, curriculum_item in enumerate(curriculum_data):
        if i % 50 == 0:
            print(f"Testing curriculum item {i+1}/{len(curriculum_data)}")
        
        success = test_single_curriculum_item(env, policy, i, deterministic)
        if success:
            successful_tests += 1

    accuracy = successful_tests / len(curriculum_data) if curriculum_data else 0.0
    return accuracy


def print_accuracy_results(deterministic_accuracy: float, non_deterministic_accuracy: float):
    """Print formatted accuracy results"""
    print("\n" + "="*50)
    print("MODEL ACCURACY RESULTS")
    print("="*50)
    print(f"Deterministic Accuracy: {deterministic_accuracy:.4f} ({deterministic_accuracy*100:.2f}%)")
    print(f"Non-deterministic Accuracy: {non_deterministic_accuracy:.4f} ({non_deterministic_accuracy*100:.2f}%)")
    print("="*50)


def main():
    """Main function to run model testing"""
    parser = argparse.ArgumentParser(description='Test trained truss optimization model accuracy')
    parser.add_argument('--checkpoint', type=str, default='models_simple/truss_progressive_policy.pol',
                        help='Path to the trained model checkpoint (default: models_simple/truss_progressive_policy.pol)')
    parser.add_argument('--curriculum', type=str, default='progressive_curriculum',
                        help='Path to the curriculum folder (default: progressive_curriculum)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Testing model: {args.checkpoint}")
    print(f"Curriculum folder: {args.curriculum}")

    # Test deterministic accuracy
    print("\nTesting deterministic accuracy...")
    start_time = perf_counter()
    deterministic_accuracy = test_model_on_curriculum(
        checkpoint_path=args.checkpoint,
        curriculum_folder=args.curriculum,
        deterministic=True
    )
    det_time = perf_counter() - start_time

    # Test non-deterministic accuracy
    print("\nTesting non-deterministic accuracy...")
    start_time = perf_counter()
    non_deterministic_accuracy = test_model_on_curriculum(
        checkpoint_path=args.checkpoint,
        curriculum_folder=args.curriculum,
        deterministic=False
    )
    nondet_time = perf_counter() - start_time

    # Print results
    print_accuracy_results(deterministic_accuracy, non_deterministic_accuracy)
    print(f"Deterministic testing completed in {det_time:.2f} seconds")
    print(f"Non-deterministic testing completed in {nondet_time:.2f} seconds")


if __name__ == "__main__":
    main() 