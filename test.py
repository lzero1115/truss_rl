import numpy as np
import polyscope as ps
import torch
import json

from src.Truss import TrussStructure
from env import TrussEnv
from policy import TrussGNNActorCritic
from graph import TrussGraphConstructor

# ---- CONFIG ----
CURRICULUM_JSON = "path/to/your/curriculum.json"
MODEL_PATH = "path/to/your/trained_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------

# 1. Load curriculum JSON (single design)
with open(CURRICULUM_JSON, "r") as f:
    item = json.load(f)

nodes = np.array(item["node_coordinates"])
edges = np.array(item["design_edges"])
fixed_nodes = np.array(item["fixed_nodes"])
design_state = np.array(item["curriculum_design_variables"])
force_node = item["force_node_indices"][0]
force_dir = item["direction_index"]

# 2. Initialize TrussStructure and Environment
truss = TrussStructure(nodes, edges, design_state, fixed_nodes)
env = TrussEnv(truss)

# 3. Load trained policy
policy = TrussGNNActorCritic(num_actions=len(edges))  # Add other args as needed
policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy.to(DEVICE)
policy.eval()

# 4. Graph constructor
graph_constructor = TrussGraphConstructor()

# 5. Polyscope setup
ps.init()
ps.remove_all_structures()
ps_cloud = ps.register_point_cloud("nodes", nodes)
ps_net = ps.register_curve_network("truss", nodes, edges)

# --- State for UI ---
current_state = design_state.copy()
force_nodes = [force_node]
force_dirs = [force_dir]
step_count = [0]  # Use list for mutability in closure
deterministic = [True]  # Use list for mutability in closure

def update_visualization(design_state):
    present_edges = [e for i, e in enumerate(edges) if design_state[i] > 1e-6]
    ps_net.update_edge_list(np.array(present_edges))
    ps_net.set_enabled(True)
    ps_cloud.set_enabled(True)

def step_callback():
    compliance = env.compute_compliance_only(current_state, force_node, force_dir)
    graph = graph_constructor.compute_graph(
        current_state, compliance, force_node, force_dir
    )
    graph = graph.to(DEVICE)
    with torch.no_grad():
        action, _, _ = policy.act(graph, mask=None, deterministic=deterministic[0])
    action = action.cpu().numpy()[0] if hasattr(action, "cpu") else int(action)
    next_state, reward, _ = env.step([current_state], [action], force_nodes, force_dirs)
    current_state[:] = next_state[0]
    update_visualization(current_state)
    step_count[0] += 1
    print(f"Step {step_count[0]}: Action={action}, Reward={reward[0]}, Deterministic={deterministic[0]}")

def switch_callback():
    deterministic[0] = not deterministic[0]
    print(f"Deterministic mode: {deterministic[0]}")

def polyscope_callback():
    ps.add_button("Step", step_callback)
    ps.add_button("Switch Determinism", switch_callback)
    ps.add_text(f"Step: {step_count[0]}")
    ps.add_text(f"Deterministic: {deterministic[0]}")

update_visualization(current_state)
ps.set_user_callback(polyscope_callback)
ps.show()