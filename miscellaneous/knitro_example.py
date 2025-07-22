import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.Truss import TrussStructure
from visualizer.knitro_visualizer import KnitroOptimizationVisualizer
import numpy as np

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
    fixed_nodes = [index_map[(0, j, 0)] for j in range(ny)]

    truss = TrussStructure(nodes, edges, design_variables, fixed_nodes)
    return truss

if __name__ == "__main__":

    truss = generate_grid_truss(3,3,1)
    visualizer = KnitroOptimizationVisualizer(truss)
    visualizer.show()