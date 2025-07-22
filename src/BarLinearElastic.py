import numpy as np
#from scipy import sparse
#from scipy.sparse import linalg as splinalg
from typing import List, Tuple
from .Bar import Bar
from .BarMaterial import BarMaterial

class BarLinearElastic(Bar):
    def __init__(self, bar: Bar, material: BarMaterial):
        super().__init__(bar.coord_, bar.length_, bar.section_)
        self.material_ = material

    @classmethod
    def from_bar(cls, bar: 'BarLinearElastic') -> 'BarLinearElastic':
        # Use super().__new__ to create base Bar object first
        base_bar = Bar(bar.coord_, bar.length_, bar.section_)
        return cls(base_bar, bar.material_)

    # 2x2 stiffness matrix, refers to 2 DoF2 of two beam end axial deformation f = k * d
    # node1 |-----| node2 ---> (+)
    # u1>u2: compression, u1<u2: stretching (u1,u2: absolute displacement)
    # the force applied to end1: E * A / L * (u2-u1)
    def axial_stiffness_matrix(self, L: float, A: float, E: float) -> np.ndarray:

        d_inv = (E * A) / L
        K = np.array([[1, -1],
                      [-1, 1]], dtype=float) * d_inv
        return K

    # 2x2
    # theta = T*L/(G*J), T is the torque, thus T = (G * J) / L * theta = T
    def torsional_stiffness_matrix(self, L: float, J: float, G: float) -> np.ndarray:
        """
        :param L: length of beam
        :param J: torsional constant
        :param G: shear modulus
        """
        return self.axial_stiffness_matrix(L, J, G)

    # bending along local traversal planes 4x4
    # R * phi = L, \sigma = E * y / R, \integral E * y * y / R dA = E * I / R = M
    # E * I * k = M
    # d = [w₁, θz₁, w₂, θz₂], axis traversal displacement and plane rotation
    # w(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3 = w1 * N1(x) + θ1 * N2(x) + w2 * N3(x) + θ2 * N4(x)
    # N is the shape function
    # [F1, M1, F2, M2]^T = k * [w₁, θ1, w₂, θ2]^T
    # virtual work δW=δw^T * k * w
    # U = 0.5 * \integral (0->L) E*I*(w''(x))^2 dx
    # local +y --> clockwise around local z axis, local +z --> counterclockwise around local y axis
    def bending_stiffness_matrix(self, L: float, E: float, I: float, axis: int=2) -> np.ndarray:

        k = np.zeros((4, 4))
        sign = 1 if axis == 2 else -1
        LL = L * L
        k[0] = np.array([12.0/LL, sign*6/L, -12.0/LL, sign*6/L])
        k[1] = np.array([sign*6/L, 4, -sign*6/L, 2])
        k[2] = -k[0]
        k[3] = np.array([sign*6/L, 2, -sign*6/L, 4])

        k *= E*I/L
        return k

    # add elements from B to A
    def add_to_matrix(self, index_i: List[int], index_j: List[int], A: np.ndarray, B: np.ndarray):
        for id, II in enumerate(index_i):
            for jd, JJ in enumerate(index_j):
                A[II, JJ] += B[id, jd]

    # [x1, y1, z1, theta_x1, theta_y1, theta_z1, x2, y2, z2, theta_x2, theta_y2, theta_z2]
    def create_local_stiffness_matrix(self) -> np.ndarray:
        L = self.length_
        Ax, Jxx, Iyy, Izz = self.material_.Ax(), self.material_.Jxx(), self.material_.Iyy(), self.material_.Izz()
        E, mu = self.material_.E_, self.material_.mu_
        G = E/(2*(1+mu))

        k = np.zeros((12,12))
        # local x axis = beam axis
        axial_x_k = self.axial_stiffness_matrix(L,Ax,E)
        tor_x_k = self.torsional_stiffness_matrix(L,Jxx,G)
        bend_z_k = self.bending_stiffness_matrix(L,E,Izz,2)
        bend_y_k = self.bending_stiffness_matrix(L,E,Iyy,1)

        self.add_to_matrix([0, 6],[0, 6], k, axial_x_k) # x axis displacement
        self.add_to_matrix([3, 9], [3, 9], k, tor_x_k) # x axis torsion
        self.add_to_matrix([1, 5, 7, 11], [1, 5, 7, 11], k, bend_z_k)
        self.add_to_matrix([2, 4, 8, 10], [2, 4, 8, 10], k, bend_y_k)

        return k

    # expanding 3x3 rotation matrix to 12x12
    def turn_diagblock(self, R3: np.ndarray) -> np.ndarray:
        R_LG = np.zeros((12,12))
        for id in range(4):
            R_LG[id*3:(id+1)*3, id*3:(id+1)*3] = R3

        return R_LG

    # global: z axis is the beam axis
    # local: x axis is the beam axis
    def create_global_transformation_matrix(self) ->np.ndarray:
        end_vert_u = self.coord_.origin_
        end_vert_v = self.coord_.origin_ + self.coord_.zaxis_ * self.length_
        L = self.length_

        c_x = (end_vert_v[0] - end_vert_u[0]) / L
        c_y = (end_vert_v[1] - end_vert_u[1]) / L
        c_z = (end_vert_v[2] - end_vert_u[2]) / L
        R = np.zeros((3,3))

        if abs(abs(c_z) - 1.0) < 1e-8:
            R[0, 2] = c_z
            R[1, 1] = 1
            R[2, 0] = -c_z

        else:
            new_x = np.array([c_x, c_y, c_z]) # local x axis represented in global coordinate
            new_y = -np.cross(new_x, np.array([0, 0, 1.0]))
            new_y /= np.linalg.norm(new_y)
            new_z = np.cross(new_x, new_y)
            R[0] = new_x
            R[1] = new_y
            R[2] = new_z
        # v_l = R * v_g: from global to local
        return R

    # f_local = k_l * d_local
    # f_global = R^T * f_local = R^T * k_l * d_local = R^T * k_l * R * d_global
    # f_g = k_g * d_g ---> k_g = R^T * k_l * R
    def create_global_stiffness_matrix(self) -> np.ndarray:
        k = self.create_local_stiffness_matrix()
        R = self.create_global_transformation_matrix()
        R_LG = self.turn_diagblock(R)
        k_G = R_LG.T @ k @ R_LG
        return k_G

    # x <--- global
    def create_global_self_weight(self) -> np.ndarray:
        Ax = self.material_.Ax()
        rho = self.material_.rho_
        w_G = np.array([0, -Ax * rho, 0]) # global load direction
        L = self.length_
        R = self.create_global_transformation_matrix()

        # [F1x, F1y, F1z, M1x, M1y, M1z, F2x, F2y, F2z, M2x, M2y, M2z]
        loads = np.zeros(12) # in global coordinates

        loads[0: 3] = w_G * L / 2
        loads[6: 9] = w_G * L / 2
        # local loads
        w_L = R @ w_G
        LL = self.length_ * self.length_
        # end slope 0 assumption, M/(E*I) = 0 at both end points
        M_L0 = np.array([0, -w_L[2] * LL / 12., w_L[1] * LL / 12.])
        M_L1 = -M_L0
        RT = R.T
        loads[3: 6] = RT @ M_L0
        loads[9: 12] = RT @ M_L1

        return loads

    # local forces
    def compute_internal_force(self, u: np.ndarray) -> np.ndarray:
        # u is global d
        R3 = self.create_global_transformation_matrix()
        R = self.turn_diagblock(R3)
        Ru = R @ u # localization
        k = self.create_local_stiffness_matrix() # 12x12
        return k @ Ru