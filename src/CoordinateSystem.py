import numpy as np
from typing import Union, Tuple
# from dataclasses import dataclass

Vector3 = np.ndarray # alias for 3d vector
# global system (x,y,z), z is along beam axis

class CoordinateSystem:
    def __init__(self,
                 origin: Union[Vector3, Tuple[float,float,float]] = None,
                 xaxis: Union[Vector3, Tuple[float,float,float]] = None,
                 yaxis: Union[Vector3, Tuple[float,float,float]] = None,
                 zaxis: Union[Vector3, Tuple[float,float,float]] = None):
        # Add underscore suffix to match C++ naming convention
        if origin is None:
            self.origin_ = np.array([0.0, 0.0, 0.0])
        else:
            self.origin_ = np.asarray(origin, dtype=float)

        # Default setting (matches C++ default constructor)
        if all(v is None for v in (xaxis, yaxis, zaxis)):
            self.xaxis_ = np.array([1.0, 0.0, 0.0])
            self.yaxis_ = np.array([0.0, 1.0, 0.0])
            self.zaxis_ = np.array([0.0, 0.0, 1.0])

        # Full constructor case
        elif all(v is not None for v in (xaxis, yaxis, zaxis)):
            self.xaxis_ = np.asarray(xaxis, dtype=float)
            self.yaxis_ = np.asarray(yaxis, dtype=float)
            self.zaxis_ = np.asarray(zaxis, dtype=float)
            # Normalize axes
            self.xaxis_ = self.xaxis_ / np.linalg.norm(self.xaxis_)
            self.yaxis_ = self.yaxis_ / np.linalg.norm(self.yaxis_)
            self.zaxis_ = self.zaxis_ / np.linalg.norm(self.zaxis_)

        # Origin and zaxis constructor case
        elif zaxis is not None:
            self.zaxis_ = np.asarray(zaxis, dtype=float)
            self.zaxis_ = self.zaxis_ / np.linalg.norm(self.zaxis_)

            temp_v = np.array([1.0, 0.0, 0.0])
            self.xaxis_ = np.cross(self.zaxis_, temp_v)
            if np.linalg.norm(self.xaxis_) < 1e-3:
                temp_v = np.array([0.0, 1.0, 0.0])
                self.xaxis_ = np.cross(self.zaxis_, temp_v)
            self.xaxis_ = self.xaxis_ / np.linalg.norm(self.xaxis_)

            self.yaxis_ = np.cross(self.zaxis_, self.xaxis_)
            self.yaxis_ = self.yaxis_ / np.linalg.norm(self.yaxis_)
        else:
            raise ValueError("Either provide all axes (x,y,z) or only z axis, or none!")

    def copy(self) -> 'CoordinateSystem':
        """Copy constructor equivalent"""
        return CoordinateSystem(
            origin=self.origin_.copy(),
            xaxis=self.xaxis_.copy(),
            yaxis=self.yaxis_.copy(),
            zaxis=self.zaxis_.copy()
        )


