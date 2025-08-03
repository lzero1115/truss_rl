import json
import numpy as np
from solvers.knitro import KnitroTrussOptimizer
from src.Truss import TrussStructure
from pathlib import Path
from random import shuffle, seed
import random
import copy
from collections import defaultdict, deque
import time
import os


# Generate 2D grid truss structure
def generate_grid_truss(nx, ny, nz):
    """Generate a 3D grid truss structure"""
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


def get_connected_candidates(current_edges, all_edges):
    """
    Get all feasible bars that connect to current structure
    Updates dynamically as structure grows
    """
    current_nodes = {node for edge in current_edges for node in edge}
    candidates = []
    
    for edge in all_edges:
        if edge not in current_edges:
            # Check if this bar connects to current structure
            if edge[0] in current_nodes or edge[1] in current_nodes:
                candidates.append(edge)
    
    return candidates


def generate_single_path(optimal_edges, all_edges, path_id):
    """
    Generate one curriculum path with controlled randomness
    """
    # random.seed(path_id)  # Ensure reproducible but different paths
    
    current_edges = set(optimal_edges)
    path = [list(current_edges)]
    
    while len(current_edges) < len(all_edges):
        candidates = get_connected_candidates(current_edges, all_edges)
        
        if not candidates:
            break
        
        # Randomly select from candidates
        selected_bar = random.choice(candidates)
        current_edges.add(selected_bar)
        path.append(list(current_edges))
    
    return path


def generate_multiple_paths(optimal_edges, all_edges, max_paths=3):
    """
    Generate multiple curriculum paths with different random selections
    """
    paths = []
    seen_paths = set()
    
    for path_id in range(max_paths):
        path = generate_single_path(optimal_edges, all_edges, path_id)
        
        # Check if this path is unique
        path_signature = tuple(tuple(sorted(edges)) for edges in path)
        if path_signature not in seen_paths:
            seen_paths.add(path_signature)
            paths.append(path)
        else:
            # Try again with different random seed
            path_id -= 1  # Retry this path
    
    return paths


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
            idx = truss.get_edge_index(edge[0], edge[1])
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


def generate_progressive_curriculum(
        truss: TrussStructure,
        force_indices: list[int],
        force_vectors: list[list[float]],
        volume_ratio: float,
        output_dir: str,
        direction_idx: int,
        max_paths: int,
        force_amplitude: float
):
    """
    Generate progressive curriculum using dynamic candidates list
    Saves to level_i folders and filters duplicates across all paths for same load case
    Returns: number of designs generated for this load case
    """
    # 1. Solve optimal design
    temp_truss = copy.deepcopy(truss)
    optimizer = KnitroTrussOptimizer(temp_truss, force_indices, force_vectors, volume_ratio)
    optimal_design = optimizer.solve_binary()
    
    if optimal_design is None:
        print(f"Failed to solve optimal design for node {force_indices[0]}, direction {direction_idx}")
        return 0
    
    # 2. Filter connected components
    filtered_optimal_design = filter_connected_design_variables(temp_truss, optimal_design)
    temp_truss.update_bars_with_weight(filtered_optimal_design)
    
    # 3. Get optimal edges
    optimal_edges = [temp_truss.all_edges[i] for i, val in enumerate(filtered_optimal_design) if abs(val - 1) < 1e-6]
    
    # 4. Generate multiple curriculum paths
    curriculum_paths = generate_multiple_paths(optimal_edges, temp_truss.all_edges, max_paths)
    
    # Debug: Print path information
    print(f"Node {force_indices[0]}, Direction {direction_idx}: Generated {len(curriculum_paths)} paths")
    for i, path in enumerate(curriculum_paths):
        print(f"  Path {i}: {len(path)} levels (optimal {len(path[0])} → complete {len(path[-1])} edges)")
    
    # 5. Evaluate optimal design performance
    ext_force = temp_truss.create_external_force_vector(force_indices, force_vectors)
    node_coordinates = temp_truss.nodes.tolist()
    disp, success, _ = temp_truss.solve_elasticity(ext_force)
    optimal_compliance = temp_truss.total_compliance if success else None
    compliance_status = "valid" if success else "failed"
    
    # 6. Organize designs by level and filter duplicates using hash table
    output_dir = Path(output_dir)
    timestamp = int(time.time() * 1000)
    
    # Hash table to filter duplicates: key = (force_node, force_direction, design_state)
    design_cache = {}
    designs_by_level = defaultdict(list)
    
    total_before_filtering = 0
    total_after_filtering = 0
    
    for path_id, path in enumerate(curriculum_paths):
        # Skip level_0 (optimal design) and start from level_1
        for level_id, curriculum_edges in enumerate(path[1:], start=1):
            total_before_filtering += 1
            
            # Create hash key: (force_node, force_direction, sorted_design_edges)
            force_node = force_indices[0]
            force_direction = direction_idx
            design_state = tuple(sorted(curriculum_edges))
            cache_key = (force_node, force_direction, design_state)
            
            # Only add if not seen before for this load case
            if cache_key not in design_cache:
                design_cache[cache_key] = True
                total_after_filtering += 1
                
                # Create design variables for this level
                curriculum_design_variables = [0.0] * len(temp_truss.all_edges)
                for edge in curriculum_edges:
                    try:
                        edge_idx = temp_truss.all_edges.index(edge)
                        curriculum_design_variables[edge_idx] = 1.0
                    except ValueError:
                        continue
                
                design_data = {
                    "node_coordinates": np.array(node_coordinates).tolist(),
                    "design_edges": [list(e) for e in temp_truss.all_edges],
                    "fixed_nodes": [int(n) for n in temp_truss.fixed_nodes],
                    "force_node_indices": [int(n) for n in force_indices],
                    "force_list": [list(map(float, f)) for f in force_vectors],
                    "volume_fraction": float(volume_ratio),
                    "optimal_design_edge_dict": {str(e): int(v) for e, v in zip(temp_truss.all_edges, filtered_optimal_design)},
                    "optimized_binary_design": [int(x) for x in filtered_optimal_design],
                    "optimal_compliance": float(optimal_compliance) if optimal_compliance is not None else None,
                    "compliance_status": str(compliance_status),
                    "force_amplitude": float(force_amplitude),
                    "curriculum_level": int(level_id),
                    "path_id": int(path_id),
                    "curriculum_edges": [list(e) for e in curriculum_edges],
                    "curriculum_design_variables": [int(x) for x in curriculum_design_variables],
                    "n_active_edges": int(len(curriculum_edges)),
                    "direction_index": int(direction_idx),
                    "timestamp": int(timestamp),
                    "is_complete": bool(len(curriculum_edges) == len(temp_truss.all_edges))
                }
                
                designs_by_level[level_id].append(design_data)
    
    # 7. Save designs to level_i folders
    #total_designs = 0
    for level_id, designs in designs_by_level.items():
        level_dir = output_dir / f"level_{level_id}"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        for design_idx, design_data in enumerate(designs):
            fname = (
                f"node{force_indices[0]}"
                f"_dir{direction_idx}"
                f"_v{volume_ratio:.2f}"
                f"_t{timestamp}"
                f"_level{level_id}"
                f"_design{design_idx}.json"
            )
            final_path = level_dir / fname
            tmp_path = str(final_path) + '.tmp'
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(design_data, f, indent=2)
                os.replace(tmp_path, final_path)  # atomic move
            except Exception as e:
                print(f"Error writing {final_path}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            #total_designs += 1
    
    if designs_by_level:
        optimal_edges_count = len(optimal_edges)
        total_edges_count = len(temp_truss.all_edges)
        max_level = max(designs_by_level.keys()) if designs_by_level else 0
        total_unique_designs = sum(len(designs) for designs in designs_by_level.values())
        print(
            f"Generated progressive curriculum: {total_unique_designs} unique designs across levels 0-{max_level} "
            f"(optimal {optimal_edges_count} → complete {total_edges_count} edges) "
            f"for node {force_indices[0]}, direction {direction_idx}")
        print(f"  Before filtering: {total_before_filtering}, After filtering: {total_after_filtering}")
        print(f"  Cache key format: (force_node={force_indices[0]}, direction={direction_idx}, design_state)")
    else:
        print(f"No curriculum generated for node {force_indices[0]}, direction {direction_idx}")
    
    #return total_designs


if __name__ == "__main__":
    # Configuration
    nx, ny, nz = 3, 3, 1  # 3x3x1 grid truss
    force_nodes = [1,2,4,5,7,8]
    truss = generate_grid_truss(nx, ny, nz)
    directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 force directions
    amplitude = 0.5
    #direction_idx = 6  # Only use direction 6
    volume_fraction = 0.5
    max_paths = 16  # Number of curriculum paths to generate
    
    output_dir = "progressive_curriculum"
    Path(output_dir).mkdir(exist_ok=True)
    
    # print(f"Generating progressive curriculum for node {force_node} with direction {direction_idx}")
    # print(f"Will generate up to {max_paths} curriculum paths")
    # print(f"Designs will be saved to level_i folders with hash table duplicate filtering")
    
    # Generate curriculum for single load case: node 2, direction 6
    #theta = directions[direction_idx]
    for idx in range(len(directions)):
        theta = directions[idx]
        fx, fy = np.cos(theta), np.sin(theta)
        for node in force_nodes:

            # Count designs generated for this load case
            generate_progressive_curriculum(
                truss,
                [node],
                [[fx * amplitude, fy * amplitude, 0.0]],
                volume_fraction,
                output_dir=output_dir,
                direction_idx=idx,
                max_paths=max_paths,
                force_amplitude=0.5
            )
    
    # Count total designs in output directory
    total_curriculum_designs = 0
    output_path = Path(output_dir)
    if output_path.exists():
        for level_dir in output_path.iterdir():
            if level_dir.is_dir() and level_dir.name.startswith("level_"):
                level_designs = len(list(level_dir.glob("*.json")))
                total_curriculum_designs += level_designs
                print(f"Level {level_dir.name}: {level_designs} designs")
    
    print(f"\n=== CURRICULUM GENERATION SUMMARY ===")
    #print(f"Load case: Node {force_node}, Direction {direction_idx}")
    print(f"Total curriculum designs generated: {total_curriculum_designs}")
    #print(f"Curriculum generation complete! Check '{output_dir}' directory for results.")