import json
import numpy as np
import os
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import networkx as nx
from src.Truss import TrussStructure


def check_connectivity(design_state: np.ndarray, truss: TrussStructure, force_node_relative: int) -> bool:
    """
    Check if the design state is connected to the force node.

    Args:
        design_state: Binary array indicating which bars are present (1) or removed (0)
        truss: TrussStructure object
        force_node_relative: Relative index of the force node

    Returns:
        bool: True if connected, False otherwise
    """
    # Convert relative node index to absolute
    unfixed_nodes = [i for i in range(len(truss.nodes)) if i not in truss.fixed_nodes]
    force_node_absolute = unfixed_nodes[force_node_relative]

    # Create graph from remaining bars
    G = nx.Graph()

    # Add nodes
    for i in range(len(truss.nodes)):
        G.add_node(i)

    # Add edges (bars) that are present in design_state
    for i, (node1, node2) in enumerate(truss.all_edges):
        if design_state[i] > 0.5:  # Bar is present
            G.add_edge(node1, node2)

    # Check if force node is connected to any fixed node
    fixed_nodes = set(truss.fixed_nodes)
    if not fixed_nodes:
        # If no fixed nodes, check if force node has any connections
        return G.degree(force_node_absolute) > 0

    # Check connectivity to fixed nodes
    for fixed_node in fixed_nodes:
        if nx.has_path(G, force_node_absolute, fixed_node):
            return True

    return False


def check_volume_constraints(design_state: np.ndarray, truss: TrussStructure, ratio=0.5, atol=1e-8):
    """
    Check if the design satisfies volume constraints using TrussStructure's built-in volume calculation.
    Allows a small tolerance for floating-point errors.
    """
    import numpy as np
    truss_copy = copy.deepcopy(truss)
    truss_copy.update_bars_with_weight(design_state)
    current_volume = truss_copy.get_truss_volume()
    full_truss = copy.deepcopy(truss)
    full_truss.update_bars_with_weight(np.ones(len(truss.all_edges)))
    full_volume = full_truss.get_truss_volume()
    target_volume = ratio * full_volume
    satisfies = (current_volume < target_volume) or np.isclose(current_volume, target_volume, atol=atol)
    return satisfies, current_volume


def check_compliance_validity(optimal_compliance: float, threshold=1.0) -> bool:
    """
    Check if the optimal compliance is valid (should be < 1.0).

    Args:
        optimal_compliance: Optimal compliance value

    Returns:
        bool: True if valid, False otherwise
    """
    return optimal_compliance < threshold and optimal_compliance > 0.0


def verify_curriculum_file(file_path: str, truss: TrussStructure) -> Dict:
    """
    Verify a single curriculum JSON file.

    Args:
        file_path: Path to the curriculum JSON file
        truss: TrussStructure object

    Returns:
        Dict: Verification results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract data
        curriculum_design = np.array(data['curriculum_design_variables'])
        optimal_design = np.array(data['optimized_binary_design'])
        optimal_compliance = data['optimal_compliance']
        force_node_relative = data['force_node_relative_indices'][0]  # First relative index
        force_dir = data['direction_index']

        # Check 1: Curriculum design connectivity
        curriculum_connected = bool(check_connectivity(curriculum_design, truss, force_node_relative))

        # Check 2: Optimal design connectivity
        optimal_connected = bool(check_connectivity(optimal_design, truss, force_node_relative))

        # Check 3: Volume constraint for optimal design
        optimal_volume_ok, opt_vol = check_volume_constraints(optimal_design, truss)
        optimal_volume_ok = bool(optimal_volume_ok)
        opt_vol = float(opt_vol)

        # Check 4: Compliance constraint for optimal design
        compliance_valid = bool(check_compliance_validity(optimal_compliance))
        optimal_compliance = float(optimal_compliance)

        # Check 5: Design state validity (should be binary)
        curriculum_binary = bool(np.all(np.isin(curriculum_design, [0, 1])))
        optimal_binary = bool(np.all(np.isin(optimal_design, [0, 1])))

        # Check 6: Optimal design should be subset of curriculum design
        optimal_subset = bool(np.all(optimal_design <= curriculum_design))

        return {
            'file_path': file_path,
            'curriculum_connected': curriculum_connected,
            'optimal_connected': optimal_connected,
            'optimal_volume_ok': optimal_volume_ok,
            'compliance_valid': compliance_valid,
            'curriculum_binary': curriculum_binary,
            'optimal_binary': optimal_binary,
            'optimal_subset': optimal_subset,
            'optimal_volume': opt_vol,
            'optimal_compliance': optimal_compliance,
            'force_node_relative': int(force_node_relative),
            'force_dir': int(force_dir),
            'all_valid': bool(curriculum_connected and optimal_connected and
                              optimal_volume_ok and compliance_valid and
                              curriculum_binary and optimal_binary and optimal_subset)
        }

    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'all_valid': False
        }


def verify_curriculum_folder(folder_path: str) -> Dict:
    """
    Verify all curriculum files in a folder.

    Args:
        folder_path: Path to the curriculum folder

    Returns:
        Dict: Summary of verification results
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        return {'error': f'Folder {folder_path} does not exist'}

    # Find all JSON files
    json_files = list(folder_path.rglob("*.json"))

    if not json_files:
        return {'error': f'No JSON files found in {folder_path}'}

    print(f"Found {len(json_files)} curriculum files to verify...")

    # Try to create truss from the first valid file
    truss = None
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                first_data = json.load(f)
            # Extract truss information
            nodes = np.array(first_data['node_coordinates'])
            edges = [tuple(edge) for edge in first_data['design_edges']]
            fixed_nodes = first_data['fixed_nodes']
            truss = TrussStructure(
                nodes=nodes,
                all_edges=edges,
                design_variables=np.ones(len(edges)),
                fixed_nodes=fixed_nodes
            )
            print(f"Created truss with {len(truss.nodes)} nodes and {len(truss.all_edges)} edges (from {json_file})")
            break
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    if truss is None:
        return {'error': 'No valid JSON files found to create truss.'}

    # Verify each file
    results = []
    valid_count = 0
    error_count = 0

    for i, json_file in enumerate(json_files):
        if i % 100 == 0:
            print(f"Verifying file {i + 1}/{len(json_files)}: {json_file.name}")

        result = verify_curriculum_file(str(json_file), truss)
        results.append(result)

        if result.get('all_valid', False):
            valid_count += 1
        elif 'error' in result:
            error_count += 1

    # Summary statistics
    total_files = len(results)
    invalid_count = total_files - valid_count - error_count

    summary = {
        'total_files': total_files,
        'valid_files': valid_count,
        'invalid_files': invalid_count,
        'error_files': error_count,
        'validity_rate': valid_count / total_files if total_files > 0 else 0,
        'detailed_results': results
    }

    # Print summary
    print(f"\n=== CURRICULUM VERIFICATION SUMMARY ===")
    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_count} ({summary['validity_rate']:.2%})")
    print(f"Invalid files: {invalid_count}")
    print(f"Error files: {error_count}")

    if invalid_count > 0 or error_count > 0:
        print(f"\n=== ISSUES FOUND ===")

        # Count different types of issues
        issue_counts = {
            'curriculum_not_connected': 0,
            'optimal_not_connected': 0,
            'optimal_volume_violation': 0,
            'invalid_compliance': 0,
            'non_binary_design': 0,
            'optimal_not_subset': 0
        }

        for result in results:
            if not result.get('all_valid', False) and 'error' not in result:
                if not result.get('curriculum_connected', True):
                    issue_counts['curriculum_not_connected'] += 1
                if not result.get('optimal_connected', True):
                    issue_counts['optimal_not_connected'] += 1
                if not result.get('optimal_volume_ok', True):
                    issue_counts['optimal_volume_violation'] += 1
                if not result.get('compliance_valid', True):
                    issue_counts['invalid_compliance'] += 1
                if not result.get('curriculum_binary', True) or not result.get('optimal_binary', True):
                    issue_counts['non_binary_design'] += 1
                if not result.get('optimal_subset', True):
                    issue_counts['optimal_not_subset'] += 1

        for issue, count in issue_counts.items():
            if count > 0:
                print(f"  {issue}: {count} files")

    return summary


def main():
    """Main verification function"""
    curriculum_folder = "progressive_curriculum"
    
    print("=== CURRICULUM VERIFICATION ===")
    print(f"Verifying curriculum folder: {curriculum_folder}")
    
    results = verify_curriculum_folder(curriculum_folder)
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    # Compute full truss volume for ratio calculation
    if results['detailed_results']:
        # Use the first file's truss for full volume
        with open(results['detailed_results'][0]['file_path'], 'r') as f:
            first_data = json.load(f)
        nodes = np.array(first_data['node_coordinates'])
        edges = [tuple(edge) for edge in first_data['design_edges']]
        fixed_nodes = first_data['fixed_nodes']
        from src.Truss import TrussStructure
        truss = TrussStructure(
            nodes=nodes,
            all_edges=edges,
            design_variables=np.ones(len(edges)),
            fixed_nodes=fixed_nodes
        )
        full_volume = truss.get_truss_volume()
    else:
        full_volume = 1.0

    # Print some sample volume statistics if there are volume violations
    if results.get('invalid_files', 0) > 0:
        print(f"\n=== OPTIMAL DESIGN VIOLATION DEBUGGING ===")
        violations = [r for r in results['detailed_results'] 
                      if not r.get('all_valid', False) and 'error' not in r]
        if violations:
            sample = violations[:5]  # First 5 violations
            for i, result in enumerate(sample):
                print(f"Sample {i+1}: {result['file_path']}")
                print(f"  Curriculum connected: {result['curriculum_connected']}")
                print(f"  Optimal connected: {result['optimal_connected']}")
                ratio = result['optimal_volume'] / full_volume if full_volume > 0 else float('nan')
                print(f"  Optimal volume ratio: {ratio:.6f}")
                print(f"  Optimal volume OK: {result['optimal_volume_ok']}")
                print(f"  Compliance: {result['optimal_compliance']:.6f}")
                print(f"  Compliance valid: {result['compliance_valid']}")
                print()
    print(f"\nVerification completed!")
    
    # Print info about error files
    error_files = [r for r in results['detailed_results'] if 'error' in r]
    if error_files:
        print("\n=== ERROR FILES ===")
        for err in error_files:
            print(f"File: {err['file_path']}")
            print(f"Error: {err['error']}")
    
    # Save detailed results to file
    output_file = "curriculum_verification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main() 