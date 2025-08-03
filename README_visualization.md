# Truss Agent Visualization

This visualization tool allows you to observe how the trained RL agent works on truss optimization tasks using Polyscope.

## Features

- **Interactive 3D Visualization**: View the truss structure in real-time
- **Agent Control**: Step through the agent's decision-making process
- **Deterministic/Non-deterministic Mode**: Switch between deterministic and stochastic policies
- **Real-time Feedback**: See compliance values, rewards, and termination conditions
- **Auto-play Mode**: Watch the agent work automatically

## Installation

1. **Install Polyscope**:
   ```bash
   pip install polyscope
   ```

2. **Install other dependencies** (if not already installed):
   ```bash
   pip install torch numpy
   ```

## Usage

### 1. Create a Sample Curriculum

First, create a sample curriculum for testing:

```bash
python create_sample_curriculum.py
```

This will create `sample_curriculum.json` with a simple 2D truss structure.

### 2. Run the Visualization

```bash
python visualize_agent.py --checkpoint path/to/checkpoint.pkl --curriculum sample_curriculum.json
```

### 3. Controls

Once the visualization window opens, you can:

- **Toggle "Deterministic Mode"**: Switch between deterministic and non-deterministic policy execution
- **Click "Step Agent"**: Execute one action manually
- **Click "Reset Episode"**: Start a new episode
- **Toggle "Auto Play"**: Enable continuous stepping with delay

## Visualization Features

### Color Coding

- **Green Bars**: Present/active bars in the truss
- **Red Bars**: Absent/removed bars
- **Blue Nodes**: Fixed nodes (boundary conditions)
- **Yellow Nodes**: Force application nodes
- **Gray Nodes**: Regular nodes

### Information Display

The console will show:
- Step-by-step action information
- Current compliance values
- Rewards received
- Episode termination status
- Comparison with optimal compliance

## Curriculum JSON Format

The curriculum JSON file should have the following structure:

```json
{
  "curriculum_key": {
    "optimal_compliance": 0.001234,
    "initial_design": [1, 1, 1, 1, 1, 1, 1],
    "target_design": [1, 1, 0, 1, 1, 0, 1],
    "force_node": 1,
    "force_direction": 0,
    "volume_constraint": 0.5,
    "description": "Description of the curriculum"
  }
}
```

## Troubleshooting

### Common Issues

1. **Polyscope not found**: Install with `pip install polyscope`
2. **Checkpoint format error**: Ensure the checkpoint contains the required fields
3. **Curriculum format error**: Check that the JSON follows the expected format
4. **Truss structure mismatch**: Ensure the truss in the checkpoint matches the curriculum

### Debug Mode

Add debug prints by modifying the visualization script:

```python
# In step_agent method, add:
print(f"Current design state: {obs['design_state']}")
print(f"Current compliance: {obs['compliance']}")
print(f"Policy action probabilities: {action_probs}")
```

## Customization

### Adding New Controls

To add new UI controls, modify the `setup_ui_controls` method:

```python
def custom_callback():
    # Your custom action here
    pass

ps.register_scalar_quantity("Controls", "Custom Action", [0.0], callback=custom_callback)
```

### Modifying Visualization

To change the visualization appearance, modify the `update_truss_visualization` method:

```python
# Change colors
bar_colors[i] = [R, G, B]  # RGB values 0-1

# Change node sizes
self.truss_network.set_radius(0.1)  # Adjust node radius
```

## Integration with Training

The visualization works with checkpoints saved by the training script. The checkpoint should contain:

- `policy_state_dict`: Trained policy weights
- `settings`: Training configuration
- `truss`: Truss structure (optional)

## Example Workflow

1. **Train the agent**:
   ```bash
   python train_example.py
   ```

2. **Create curriculum**:
   ```bash
   python create_sample_curriculum.py
   ```

3. **Visualize results**:
   ```bash
   python visualize_agent.py --checkpoint checkpoint.pkl --curriculum sample_curriculum.json
   ```

4. **Observe and analyze**:
   - Watch how the agent removes bars
   - Compare final compliance with optimal
   - Switch between deterministic/stochastic modes
   - Reset and try different scenarios 