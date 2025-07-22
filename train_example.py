import os
import json
import numpy as np
import torch
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from training import train_truss_optimization, load_curriculum_from_folder


def load_progressive_curriculum(curriculum_folder: str) -> list:
    """
    Load progressive curriculum from level_i folders

    Args:
        curriculum_folder: Path to the progressive_curriculum folder

    Returns:
        List of curriculum data dictionaries
    """
    curriculum_folder = Path(curriculum_folder)
    if not curriculum_folder.exists():
        raise ValueError(f"Curriculum folder {curriculum_folder} does not exist")

    curriculum_data = []

    # Find all level directories
    level_dirs = [d for d in curriculum_folder.iterdir()
                  if d.is_dir() and d.name.startswith("level_")]

    if not level_dirs:
        raise ValueError(f"No level directories found in {curriculum_folder}")

    print(f"Found {len(level_dirs)} level directories")

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

    print(f"Successfully loaded {len(curriculum_data)} curriculum items")
    return curriculum_data


def main():


    # Training settings (MODIFIED for better confidence)
    settings = {
        "ppo": {
            "eps_clip": 0.2,
            "lr_actor": 4e-3,  # INCREASED: 2e-3 → 4e-3 for faster convergence to sharp distributions
            "betas_actor": [0.95, 0.999],
            "lr_milestones": [100, 300],
            "gamma": 0.95,
            # 🚀 FIXED ENTROPY SETTINGS - Reduced to improve policy confidence
            "base_entropy_weight": 0.001,        # Keep current value - focus on learning rate instead
            "entropy_weight_increase": 0.001,   # CHANGED: 0.001 → 0.0001 (10x slower increase)
            "max_entropy_weight": 0.01,         # CHANGED: 0.01 → 0.002 (5x reduction)
            "per_alpha": 0.8,
            "per_beta": 0.1,
            "per_num_anneal": 500,
            "hidden_sizes": (512, 512, 256),  # Deeper architecture for better performance
        },
        "training": {
            "max_train_episodes": 5000,
            "save_delta_accuracy": 0.01,
            "policy_update_batch_size": 1024,  # Increased for better GPU utilization
            "K_epochs": 5,
            "num_rollouts": 512,  # Increased for better GPU utilization
            "policy_name": "truss_progressive_policy",
            "terminate_nondeterministic_accuracy": 0.88,
            "terminate_deterministic_accuracy": 0.95,
            "save_epochs": 10,  # Evaluate accuracy every N episodes (like disassembly training)
            "accuracy_sample_size": 200  # Number of curriculum items to sample for accuracy evaluation
        }
    }

    # Path to progressive curriculum folder
    curriculum_folder = "progressive_curriculum"

    # Check if curriculum folder exists
    if not Path(curriculum_folder).exists():
        print(f"Progressive curriculum folder {curriculum_folder} not found!")
        print("Please run curriculum_generator_progressive.py first to generate the curriculum.")
        return

    # Run training
    try:
        print("Starting truss optimization training with progressive curriculum...")
        print(f"Curriculum folder: {curriculum_folder}")
        print(f"Settings: {json.dumps(settings, indent=2)}")

        # Load curriculum from all levels
        curriculum_data = load_progressive_curriculum(curriculum_folder)
        print(f"Loaded {len(curriculum_data)} curriculum items")

        # Start training
        start_time = perf_counter()
        train_truss_optimization(curriculum_folder, settings)
        total_time = perf_counter() - start_time

        print(f"\nTraining completed in {total_time:.2f} seconds")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
