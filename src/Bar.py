import numpy as np
from typing import Tuple, Optional
from .CoordinateSystem import CoordinateSystem
from .BarCrossSection import BarCrossSection

class Bar:
    def __init__(self,
                 coord: Optional[CoordinateSystem] = None,
                 length: Optional[float] = None,
                 section: Optional[BarCrossSection] = None):

        self.coord_ = CoordinateSystem() if coord is None else coord
        self.length_ = 0.0 if length is None else length
        self.section_ = section
        self.color_ = np.array([0.0, 0.0, 0.0]) # for further rendering

    @classmethod
    def from_points(cls, sta_pt: np.ndarray, end_pt: np.ndarray, cross_section: BarCrossSection)->'Bar':
        coord = CoordinateSystem(origin=sta_pt, zaxis=end_pt-sta_pt)
        length = float(np.linalg.norm(sta_pt-end_pt))
        return cls(coord, length, cross_section)

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]: # V and F
        if self.section_ is None:
            raise ValueError("Bar section is not initialized!")

        points = self.section_.get_points()
        n = len(points)
        # not consider the two side disks
        V = np.zeros((2 * n, 3))
        F = np.zeros((2 * n, 3), dtype=int)

        for id in range(n):
            V[id] = (self.coord_.origin_ +
                     self.coord_.xaxis_ * points[id][0] +
                     self.coord_.yaxis_ * points[id][1])
            V[id+n] = V[id] + self.length_ * self.coord_.zaxis_

        for kd in range(n):
            next_kd = (kd+1) % n # circular connection
            F[2 * kd] = [kd, next_kd, next_kd + n]
            F[2 * kd + 1] = [kd, next_kd + n, kd + n]

        return V, F

    def center(self) -> np.ndarray:
        return self.coord_.origin_ + self.coord_.zaxis_ * self.length_ / 2

    def point(self, t: float):
        return self.coord_.origin_ + self.coord_.zaxis_ * t

    def compute_center_y(self):
        return float (self.center()[1])
