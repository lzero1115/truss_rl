import numpy as np
from mosek.fusion import *
from scipy import sparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Truss import TrussStructure
import copy

# PSDP reformulation for minimal compliance problem (relaxed version)
# min \tau
# respect to sum \rho_j V_j <= V_c
# SDP [[\tau, -f^T],[-f,K]]
# \rho>=0, \rho<=1
# K = sum (1-\rho_j)*K_v_j + \rho_j*K_a_j = sum K_v + sum (K_a-K_v)_j*\rho_j

# Be careful the solved rho is always relative to current truss, which means to apply rho*A_current, not a normalized cross section

class Relaxed_SDP_Optimizer:

    def __init__(self, truss: TrussStructure, force_indices, force_vectors, volume_ratio):
        self.truss = copy.deepcopy(truss) # it has
        # force size: np.zeros(len(self.nodes) * 6)
        self.force_vector = self.truss.create_external_force_vector(force_indices, force_vectors)
        self.temp_design = self.truss.temp_design

        self.volume_ratio = volume_ratio # volume constraint
        # TODO: design region switching
        self.n_var = len(self.temp_design)
        self.n_dofs_full = len(self.truss.nodes) * 6
        self.n_dofs_proj = self.truss.proj_dofs

        self.mapped_force = self.truss._compute_loads(self.force_vector) # projected F

        self.matrix_size = self.n_dofs_proj + 1 # \tau for extra degree
        self.total_volume = self.truss.get_truss_volume()
        self.volume_constraint = self.total_volume * self.volume_ratio
        print(f"Full DOFs: {self.n_dofs_full}")
        print(f"Reduced DOFs: {self.n_dofs_proj}")

        self.K_a = self.truss.K_bar.copy()
        self.K_v = self.truss.K_virtual_bar.copy()

        self.K_fixed = np.zeros((self.n_dofs_proj, self.n_dofs_proj)) # TODO: use sparse
        self.K_diff = []

        for idx in range(len(self.temp_design)):
            # kdiff = np.zeros((self.n_dofs_proj, self.n_dofs_proj))
            self.K_fixed += self.K_v[idx]
            kdiff = self.K_a[idx] - self.K_v[idx]
            kdiff += 1e-8 * sparse.eye(self.n_dofs_proj) # TODO: necessity analysis
            self.K_diff.append(kdiff)

        self.K_fixed = self.K_fixed + 1e-8 * sparse.eye(self.n_dofs_proj) # maybe unnecessary (

    def solve(self):

        epsilon = 1e-8
        with Model("Truss_SDP_Standard") as M:

            X = M.variable("X", Domain.inPSDCone(self.matrix_size))
            rho = M.variable("rho", self.n_var, Domain.inRange(epsilon, 1.0))
            vols = self.truss.volumes.copy()
            vol_terms = []
            for idx in range(len(self.temp_design)):
                vol_terms.append(Expr.mul(vols[idx], rho.index(idx)))

            M.constraint("volume", Expr.add(vol_terms), Domain.lessThan(self.volume_constraint))

            # force constraints
            for i in range(self.n_dofs_proj):
                M.constraint(Expr.sub(X.index(0, i + 1), -self.mapped_force[i]), Domain.equalsTo(0.0))
                M.constraint(Expr.sub(X.index(i + 1, 0), -self.mapped_force[i]), Domain.equalsTo(0.0))

            # stiffness matrix constraints with SDP symmetric
            for i in range(self.n_dofs_proj):
                for j in range(i, self.n_dofs_proj):

                    k_const = self.K_fixed[i,j]

                    # Add variable part from design edges
                    k_expr_terms = []
                    for idx in range(len(self.temp_design)):
                        k_diff = self.K_diff[idx][i, j]

                        k_expr_terms.append(Expr.mul(k_diff, rho.index(idx)))

                    if k_expr_terms:
                        k_expr = Expr.add(k_const, Expr.add(k_expr_terms))
                    else:
                        k_expr = Matrix.dense([[k_const]])

                    M.constraint(Expr.sub(X.index(i + 1, j + 1), k_expr), Domain.equalsTo(0.0))
                    if i != j: # symmetric
                        M.constraint(Expr.sub(X.index(j + 1, i + 1), k_expr), Domain.equalsTo(0.0))

            # Objective
            M.objective(ObjectiveSense.Minimize, X.index(0, 0))

            # Solver parameters
            M.setSolverParam("intpntCoTolPfeas", 1e-6)
            M.setSolverParam("intpntCoTolDfeas", 1e-6)
            M.setSolverParam("intpntCoTolRelGap", 1e-6)
            M.setSolverParam("intpntSolveForm", "dual")
            M.setSolverParam("numThreads", 8)
            M.setSolverParam("optimizer", "conic")

            try:
                print("Running optimization...")

                M.solve()
                status = M.getPrimalSolutionStatus()
                print(f"Solution status: {status}")

                if status in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
                    # Extract optimized design variables
                    design_rho = rho.level()
                    opt_tau = X.index(0, 0).level()

                    # Map back to full bar array
                    opt_rho = np.zeros(len(self.temp_design))
                    for idx in range(self.n_var):
                        opt_rho[idx] = design_rho[idx]

                    print(f"Optimal compliance (tau): {opt_tau}")
                    # print("Sample rho values:", design_rho[:5])
                    return opt_rho, opt_tau
                else:
                    print(f"Problem status: {M.getProblemStatus()}")
                    print(f"Primal status: {M.getPrimalSolutionStatus()}")
                    print(f"Dual status: {M.getDualSolutionStatus()}")
                    return None, None
            except Exception as e:
                print(f"Optimization error type: {type(e).__name__}")
                print(f"Optimization error: {str(e)}")
                print("Optimization failed with exception")
                return None, None