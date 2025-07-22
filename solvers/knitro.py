import numpy as np
from knitro import *
from scipy import sparse
from scipy.sparse import linalg, csr_matrix
from src.Truss import TrussStructure
import copy

# f^T K^-1 f, returned rho is not the optimal design, the optimal design is (rho * prev_design)!
# since rho is a relative variable respect to current truss
# below solver is not considering the self weight of each bar but only external force
class KnitroTrussOptimizer:
    def __init__(self, truss: TrussStructure, force_indices, force_vectors, volume_ratio):
        self.truss = copy.deepcopy(truss)  # we will not consider its self.weight
        # force size: np.zeros(len(self.nodes) * 6)
        self.force_vector = self.truss.create_external_force_vector(force_indices, force_vectors)

        self.mapped_force = self.truss._compute_loads(self.force_vector)  # projected F
        self.temp_design = self.truss.temp_design
        self.vols = self.truss.volumes.copy()
        self.volume_ratio = volume_ratio  # volume constraint
        self.total_volume = self.truss.get_truss_volume()
        self.volume_constraint = self.total_volume * self.volume_ratio

        self.n_var = len(self.temp_design)
        self.n_dofs_full = len(self.truss.nodes) * 6
        self.n_dofs_proj = self.truss.proj_dofs

        self.K_a = self.truss.K_bar.copy()
        self.K_v = self.truss.K_virtual_bar.copy()

        #self.K_fixed = np.zeros((self.n_dofs_proj, self.n_dofs_proj))
        self.K_fixed = csr_matrix((self.n_dofs_proj, self.n_dofs_proj))
        self.K_diff = []

        for idx in range(len(self.temp_design)):
            # kdiff = np.zeros((self.n_dofs_proj, self.n_dofs_proj))
            self.K_fixed += self.K_v[idx]
            kdiff = self.K_a[idx] - self.K_v[idx]
            #kdiff += 1e-8 * sparse.eye(self.n_dofs_proj)  # TODO: necessity analysis
            self.K_diff.append(kdiff)

        self.K_fixed = self.K_fixed + 1e-8 * sparse.eye(self.n_dofs_proj)  # maybe unnecessary (

    def _assemble_global_stiffness(self, design_rho):
        K_global = self.K_fixed.copy()  # changed from deepcopy to shallow copy for speed, still safe
        for idx in range(len(self.temp_design)):
            K_global += design_rho[idx] * self.K_diff[idx]
        return K_global.tocsr()

    def callbackEvalFC(self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            return -1
        rho = evalRequest.x
        try:
            K = self._assemble_global_stiffness(rho)
            u = linalg.spsolve(K, self.mapped_force)
            evalResult.obj = self.mapped_force.dot(u) # compliance
            return 0
        except Exception as e:
            print(f"[callbackEvalFC] failed: {e}")
            evalResult.obj = 1e10
            return 0

    # ∂(f^TK(rho)^-1f)/∂(rho_i) = f^T * ∂(K(rho)^-1)/∂(rho_i) * f
    # ∂(K(rho)^-1)/∂(rho_i) = -K^-1 * K_i * K^-1
    # f^T * ∂(K(rho)^-1)/∂(rho_i) * f = -u^T * K_i * u
    def callbackEvalGA(self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            return -1
        rho = evalRequest.x
        try:
            K = self._assemble_global_stiffness(rho)
            u = linalg.spsolve(K, self.mapped_force)
            for idx in range(len(self.temp_design)):
                dK = self.K_diff[idx]
                evalResult.objGrad[idx] = -u.dot(dK @ u)
            return 0
        except Exception as e:
            print(f"[callbackEvalGA] failed: {e}")
            evalResult.objGrad[:] = 0.0
            return 0

    # ∂^2(f^TK(rho)^-1f)/∂(rho_i)∂(rho_j) = -∂(u^T * K_i * u)/∂(rho_j)
    # u = K^-1 * f, ∂(u)/∂(rho_j) = -K^-1 * K_j * K^-1 * f = -K^-1 * K_j * u
    # ∂(u^T)/∂(rho_j) = -u^T * K_j * K^-1
    # -∂(u^T * K_i * u)/∂(rho_j)
    # = u^T * K_j * K^-1 * K_i * u + u^T * K_i * K^-1 * K_j * u
    def callbackEvalH(self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type not in [KN_RC_EVALH, KN_RC_EVALH_NO_F]:
            return -1
        rho = evalRequest.x
        sigma = evalRequest.sigma
        try:
            K = self._assemble_global_stiffness(rho)
            u = linalg.spsolve(K, self.mapped_force)
            hidx = 0
            cache = {}

            for i_idx in range(len(self.temp_design)):
                Ki = self.K_diff[i_idx]
                Ki_u = Ki @ u
                temp_i = linalg.spsolve(K, Ki_u)
                cache[i_idx] = temp_i

                evalResult.hess[hidx] = sigma * 2 * u.dot(Ki @ temp_i)
                hidx += 1
                for j_idx in range(i_idx + 1, len(self.temp_design)):
                    Kj = self.K_diff[j_idx]
                    temp1 = cache[i_idx]
                    temp2 = cache.get(j_idx)
                    if temp2 is None:
                        temp2 = linalg.spsolve(K, Kj @ u)
                        cache[j_idx] = temp2

                    hval = sigma * (u.dot(Kj @ temp1) + u.dot(Ki @ temp2))
                    evalResult.hess[hidx] = hval
                    hidx += 1
            return 0
        except Exception as e:
            print(f"[callbackEvalH] failed: {e}")
            evalResult.hess[:] = 0.0
            return 0

    def solve_relaxed(self):

        try:
            kc = KN_new()
        except:
            print("Knitro license error.")
            return None

        #(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ITER)
        KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_NONE)
        KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_BAR_DIRECT)
        KN_set_int_param(kc, KN_PARAM_MAXIT, 500)
        # Stricter tolerances for accuracy
        KN_set_double_param(kc, KN_PARAM_FEASTOL, 1e-8)
        KN_set_double_param(kc, KN_PARAM_OPTTOL, 1e-8)


        KN_add_vars(kc, self.n_var)
        KN_set_var_lobnds(kc, xLoBnds=[1e-6] * self.n_var)
        KN_set_var_upbnds(kc, xUpBnds=[1.0] * self.n_var)

        KN_add_cons(kc, 1)
        KN_set_con_upbnds(kc, [0], [self.volume_constraint])
        KN_set_con_lobnds(kc, [0], [0.0])
        for idx in range(self.n_var):
            volume = self.vols[idx]
            KN_add_con_linear_struct(kc, [0], [idx], [volume])

        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

        cb = KN_add_eval_callback(kc, evalObj=True, indexCons=[], funcCallback=self.callbackEvalFC)
        KN_set_cb_grad(kc, cb,
                       objGradIndexVars=list(range(self.n_var)),
                       jacIndexCons=[], jacIndexVars=[],
                       gradCallback=self.callbackEvalGA)

        h1, h2 = [], [] # upper triangle
        for i in range(self.n_var):
            h1.append(i)
            h2.append(i)
            for j in range(i + 1, self.n_var):
                h1.append(i)
                h2.append(j)
        KN_set_cb_hess(kc, cb, hessIndexVars1=h1, hessIndexVars2=h2, hessCallback=self.callbackEvalH)


        avg_density = self.volume_constraint / self.total_volume
        init_rho = np.clip([avg_density] * self.n_var, 1e-6, 1.0)
        KN_set_var_primal_init_values(kc, xInitVals=init_rho)

        try:
            KN_solve(kc)
            _, obj_val, design_rho, _ = KN_get_solution(kc)
            print(f"Optimal compliance: {obj_val:.6f}")

            KN_free(kc)
            return np.array(design_rho)
        except Exception as e:
            print(f"Solver error: {e}")
            KN_free(kc)
            return None

    def solve_binary(self, warm_start=None):

        try:
            kc = KN_new()
        except:
            print("Knitro license error.")
            return None

        KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_NONE)
        KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_BAR_DIRECT)
        KN_set_int_param(kc, KN_PARAM_MIP_METHOD, KN_MIP_METHOD_BB)
        KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
        # KN_set_double_param(kc, KN_PARAM_MIP_INTGAPREL, 0.05)
        # Tighter tolerances for numerical accuracy
        KN_set_double_param(kc, KN_PARAM_FEASTOL, 1e-8)
        KN_set_double_param(kc, KN_PARAM_OPTTOL, 1e-8)

        KN_set_double_param(kc, KN_PARAM_MIP_INTGAPREL, 0.001)

        KN_add_vars(kc, self.n_var)
        KN_set_var_types(kc, xTypes=[KN_VARTYPE_BINARY] * self.n_var)
        KN_set_var_lobnds(kc, xLoBnds=[0.0] * self.n_var)
        KN_set_var_upbnds(kc, xUpBnds=[1.0] * self.n_var)

        KN_add_cons(kc, 1)
        KN_set_con_upbnds(kc, [0], [self.volume_constraint])
        KN_set_con_lobnds(kc, [0], [0.0])

        for idx in range(self.n_var):
            volume = self.vols[idx]
            KN_add_con_linear_struct(kc, [0], [idx], [volume])

        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

        cb = KN_add_eval_callback(kc, evalObj=True, indexCons=[], funcCallback=self.callbackEvalFC)
        KN_set_cb_grad(kc, cb,
                       objGradIndexVars=list(range(self.n_var)),
                       jacIndexCons=[], jacIndexVars=[],
                       gradCallback=self.callbackEvalGA)

        h1, h2 = [], []  # upper triangle
        for i in range(self.n_var):
            h1.append(i)
            h2.append(i)
            for j in range(i + 1, self.n_var):
                h1.append(i)
                h2.append(j)
        KN_set_cb_hess(kc, cb, hessIndexVars1=h1, hessIndexVars2=h2, hessCallback=self.callbackEvalH)

        if warm_start is not None:
            init_vals = np.round(warm_start).clip(0, 1)
            KN_set_var_primal_init_values(kc, xInitVals=init_vals)

        try:
            KN_solve(kc)
            _, obj_val, design_rho, _ = KN_get_solution(kc)

            #print(f"Optimal compliance: {obj_val:.6f}")
            #print(f"rho list: {design_rho}")

            KN_free(kc)
            # Ensure strict binary output
            design_rho = np.round(design_rho).astype(int)
            return np.array(design_rho)
        except Exception as e:
            print(f"Solver error: {e}")
            KN_free(kc)
            return None
