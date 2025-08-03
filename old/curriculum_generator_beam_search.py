import json
import numpy as np
from solvers.knitro import KnitroTrussOptimizer
from src.Truss import TrussStructure
from pathlib import Path
from random import shuffle
import random
import copy
from collections import defaultdict, deque
import time


# below code is for 2D cases, although the computing framework is 3D
# only one force on one node simple example code
# by lex 06/10/2025
# easy 2d truss example
def generate_grid_truss(nx, ny, nz):
    # n is the number of nodes per side

    nodes = []
    index_map = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = len(nodes)
                index_map[(i, j, k)] = idx
                nodes.append(np.array([i, j, k], dtype=float))
    nodes = np.array(nodes)

    edges = []

    # 1. X-direction edges
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                a = index_map[(i, j, k)]
                b = index_map[(i + 1, j, k)]
                edges.append((a, b))

    # 2. Y-direction edges
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                a = index_map[(i, j, k)]
                b = index_map[(i, j + 1, k)]
                edges.append((a, b))

    # 3. Z-direction edges
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                a = index_map[(i, j, k)]
                b = index_map[(i, j, k + 1)]
                edges.append((a, b))

    # 4. Diagonals in XY-plane
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = index_map[(i, j, k)]
                b = index_map[(i + 1, j + 1, k)]
                c = index_map[(i + 1, j, k)]
                d = index_map[(i, j + 1, k)]
                edges.append((a, b))
                edges.append((c, d))

    design_variables = np.ones(len(edges))
    fixed_nodes = [index_map[(0, j, 0)] for j in range(ny)]  # left side

    truss = TrussStructure(nodes, edges, design_variables, fixed_nodes)
    return truss


# CHANGE 1: Add filter function for connected design variables
def filter_connected_design_variables(truss: TrussStructure, design_variables: list) -> list:
    """Filter design variables to keep only the largest connected component"""

    # Get active edges
    active_edges = []
    for idx, val in enumerate(design_variables):
        if abs(val - 1) < 1e-6:
            active_edges.append(truss.all_edges[idx])

    if not active_edges:
        return design_variables.copy()

    # Find largest connected component that includes fixed nodes
    connected_edges = get_largest_connected_component_with_fixed_nodes(active_edges, truss.fixed_nodes)

    # Create filtered design
    filtered_design = [0.0] * len(design_variables)
    for edge in connected_edges:
        try:
            idx = truss.get_edge_index(edge[0],edge[1])
            filtered_design[idx] = design_variables[idx]  # Keep original value
        except ValueError:
            continue

    return filtered_design


def get_largest_connected_component_with_fixed_nodes(edges, fixed_nodes):
    """Find the largest connected component that includes at least one fixed node"""
    if not edges:
        return []

    # Build adjacency list
    adj = defaultdict(list)
    nodes = set()

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        nodes.update([u, v])

    # Find all connected components
    visited = set()
    components = []

    for node in nodes:
        if node not in visited:
            # BFS to find this component
            component_nodes = set()
            queue = deque([node])

            while queue:
                curr = queue.popleft()
                if curr not in visited:
                    visited.add(curr)
                    component_nodes.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            queue.append(neighbor)

            # Get edges in this component
            component_edges = []
            for u, v in edges:
                if u in component_nodes and v in component_nodes:
                    component_edges.append((u, v))

            # Check if this component includes any fixed nodes
            has_fixed_node = any(node in fixed_nodes for node in component_nodes)

            components.append({
                'edges': component_edges,
                'nodes': component_nodes,
                'has_fixed': has_fixed_node,
                'size': len(component_edges)
            })

    # Filter components that include fixed nodes
    valid_components = [comp for comp in components if comp['has_fixed']]

    if not valid_components:
        # If no component includes fixed nodes, return the largest component
        return max(components, key=lambda x: x['size'])['edges'] if components else []

    # Return the largest component that includes fixed nodes
    largest_valid = max(valid_components, key=lambda x: x['size'])
    return largest_valid['edges']


# CHANGE 2: Generate multiple designs for each level with global duplicate filtering
def generate_beam_search_curriculum(optimal_edge_list, all_edges, n_beam=10):
    """Generate multiple feasible designs for each level: optimal + k bars (k >= 1) with duplicate filtering for the same external load case"""

    m = len(optimal_edge_list)  # Number of edges in optimal design
    n = len(all_edges)  # Total number of edges
    max_levels = n - m  # Maximum levels to generate

    curriculum_by_level = {}
    seen_designs_this_case = set()  # Track all designs across all levels for this external load case
    seen_designs_this_case.add(tuple(sorted(optimal_edge_list)))  # Add optimal to seen set

    all_edges_set = set(all_edges)

    # Generate designs for level 1, 2, ..., max_levels
    for level in range(1, max_levels + 1):
        target_edge_count = m + level  # optimal + level bars
        level_designs = []

        # Start from previous level designs
        if level == 1:
            prev_designs = [optimal_edge_list]  # Start from optimal for level 1
        else:
            prev_designs = curriculum_by_level.get(level - 1, [])

        # For each design from previous level, try adding one more edge
        for prev_design in prev_designs:
            prev_state = set(prev_design)
            available_edges = list(all_edges_set - prev_state)

            # Try adding each available edge
            for edge in available_edges:
                new_state = prev_state | {edge}

                # Check if new state is connected and has correct edge count
                if len(new_state) == target_edge_count and is_edge_set_connected(new_state):
                    new_design = list(new_state)
                    design_tuple = tuple(sorted(new_design))

                    # Only add if not seen globally across all levels in this external load case
                    if design_tuple not in seen_designs_this_case:
                        seen_designs_this_case.add(design_tuple)
                        level_designs.append(new_design)

        if not level_designs:
            print(f"Warning: No feasible design found for level {level} ({target_edge_count} edges)")
            break

        # Keep multiple designs for diversity (up to n_beam)
        if len(level_designs) > n_beam:
            indices = np.random.choice(len(level_designs), n_beam, replace=False)
            level_designs = [level_designs[i] for i in indices]

        curriculum_by_level[level] = level_designs

        if len(level_designs) == 0:
            print(f"Warning: No unique designs for level {level} after duplicate filtering")

    return curriculum_by_level


def is_edge_set_connected(edge_set):
    """Simple connectivity check for edge set"""
    if not edge_set:
        return False

    edges = list(edge_set)

    # Build adjacency list
    adj = defaultdict(list)
    nodes = set()

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        nodes.update([u, v])

    if not nodes:
        return False

    # BFS from any node
    start_node = next(iter(nodes))
    visited = set()
    queue = deque([start_node])

    while queue:
        curr = queue.popleft()
        if curr not in visited:
            visited.add(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return len(visited) == len(nodes)


def generate_beam_curriculum(
        truss: TrussStructure,
        force_indices: list[int],
        force_vectors: list[list[float]],
        volume_ratio: float,
        output_dir: str,
        direction_idx: int,
        beam_size: int,
        force_amplitude: float
) -> None:
    temp_truss = copy.deepcopy(truss)
    temp_design = temp_truss.temp_design
    all_edges = temp_truss.all_edges
    temp_edge_list = [all_edges[idx] for idx, val in enumerate(temp_design) if abs(val - 1) < 1e-6]

    optimizer = KnitroTrussOptimizer(temp_truss, force_indices, force_vectors, volume_ratio)
    rho = optimizer.solve_binary()
    rho_list = rho.tolist()
    optimal_design = [x * y for x, y in zip(temp_design, rho_list)]

    filtered_optimal_design = filter_connected_design_variables(temp_truss, optimal_design)
    temp_truss.update_bars_with_weight(filtered_optimal_design)
    ext_force = temp_truss.create_external_force_vector(force_indices, force_vectors)

    node_coordinates = temp_truss.nodes.tolist()
    disp, success, _ = temp_truss.solve_elasticity(ext_force)
    optimal_compliance = temp_truss.total_compliance if success else None
    compliance_status = "valid" if success else "failed"

    optimal_edge_list = [all_edges[idx] for idx, val in enumerate(filtered_optimal_design) if abs(val - 1) < 1e-6]

    output_dir = Path(output_dir)
    curriculum_by_level = generate_beam_search_curriculum(optimal_edge_list, all_edges, n_beam=beam_size)
    timestamp = int(time.time() * 1000)

    # Build relative force index mapping
    relative_index_map = {node: i for i, node in enumerate(force_indices)}
    relative_indices = [relative_index_map[n] for n in force_indices]

    total_designs = 0
    for level, designs in curriculum_by_level.items():
        level_dir = output_dir / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)

        for design_idx, curriculum_edges in enumerate(designs):
            curriculum_design_variables = [0.0] * len(all_edges)
            for edge in curriculum_edges:
                try:
                    edge_idx = all_edges.index(edge)
                    curriculum_design_variables[edge_idx] = 1.0
                except ValueError:
                    continue

            data = {
                "node_coordinates": node_coordinates,
                "design_edges": [list(e) for e in all_edges],
                "fixed_nodes": temp_truss.fixed_nodes,

                "force_node_indices": force_indices,
                "force_node_relative_indices": relative_indices,
                "force_list": force_vectors,
                "volume_fraction": volume_ratio,
                "optimal_design_edge_dict": {str(e): int(v) for e, v in zip(all_edges, filtered_optimal_design)},
                "optimized_binary_design": filtered_optimal_design,
                "optimal_compliance": optimal_compliance,
                "compliance_status": compliance_status,
                "force_amplitude": force_amplitude,
                "curriculum_level": level,
                "design_index": design_idx,
                "curriculum_edges": [list(e) for e in curriculum_edges],
                "curriculum_design_variables": curriculum_design_variables,
                "n_active_edges": len(curriculum_edges),
                "direction_index": direction_idx,
                "timestamp": timestamp,
                "is_optimal": level == 0,
                "is_complete": len(curriculum_edges) == len(all_edges)
            }

            fname = (
                f"node{force_indices[0]}"
                f"_dir{direction_idx}"
                f"_v{volume_ratio:.2f}"
                f"_t{timestamp}"
                f"_level{level}"
                f"_design{design_idx}.json"
            )
            with open(level_dir / fname, 'w') as f:
                json.dump(data, f, indent=2)

            total_designs += 1

    if curriculum_by_level:
        min_level = min(curriculum_by_level.keys())
        max_level = max(curriculum_by_level.keys())
        optimal_edges = len(optimal_edge_list)
        total_edges = len(all_edges)
        print(
            f"Generated curriculum: {total_designs} designs across levels {min_level}-{max_level} (optimal {optimal_edges} → complete {total_edges} edges) for node {force_indices[0]}, direction {direction_idx}")
    else:
        print(f"No curriculum generated for node {force_indices[0]}, direction {direction_idx}")


# Keep original random method as fallback (not used in main flow anymore)
def add_connected_edges(A: list[tuple[int, int]], B: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    A_set = set(A)
    candidates = [e for e in B if e not in A_set]
    nodes = {n for u, v in A for n in (u, v)}  # current connected nodes
    added = []
    for _ in range(k):
        connected = [e for e in candidates if e[0] in nodes or e[1] in nodes]
        if not connected:  # no more edges to add
            break
        choice = connected[np.random.randint(len(connected))]
        added.append(choice)
        nodes.update(choice)
        candidates.remove(choice)

    return sorted(A + added)


if __name__ == "__main__":

    nx, ny, nz = 3, 3, 1
    force_nodes = [1, 2, 4, 5, 7, 8]  # exclude fixed nodes
    truss = generate_grid_truss(nx, ny, nz)
    directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    volume_fraction = 0.5

    output_dir = "beam_curriculum"  # Changed from "chained_curriculum"
    Path(output_dir).mkdir(exist_ok=True)

    for node in force_nodes:
        for direction_idx, theta in enumerate(directions):
            fx, fy = np.cos(theta), np.sin(theta)
            generate_beam_curriculum(
                truss,
                [node],
                [[fx * 0.5, fy * 0.5, 0.0]],
                volume_fraction,
                output_dir=output_dir,
                direction_idx=direction_idx,
                beam_size=16,  # beam search width, not number of paths
                force_amplitude = 0.5
            )