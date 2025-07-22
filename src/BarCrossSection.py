from enum import Enum
import numpy as np
import math
#import json
from typing import List, Dict, Any

class CrossSectionType(Enum):
    None_ = 0
    ROUND = 1
    Rectangle = 2

class BarCrossSection:

    def __init__(self):
        self.Ax_ = 0.0 # area m^2
        self.Jxx_ = 0.0 # torsional constant m^4 (local beam axis --> x axis)
        self.Iyy_ = 0.0 # second axial moment of area
        self.Izz_ = 0.0
        self.type_ = CrossSectionType.None_
        self.scale_ = 1.0

    def get_points(self) -> List[np.ndarray]:
        return []

    def read(self, json_node:Dict[str,any]) -> None:
        """read cross section info from json"""
        if "A" in json_node:
            self.Ax_ = float(json_node["A"])
        if "Jx" in json_node:
            self.Jxx_ = float(json_node["Jx"])
        if "Iy" in json_node:
            self.Iyy_ = float(json_node["Iy"])
        if "Iz" in json_node:
            self.Izz_ = float(json_node["Iz"])
        if "type" in json_node:
            if json_node["type"] == "rectangle":
                self.type_ = CrossSectionType.Rectangle
            if json_node["type"] == "round":
                self.type_ = CrossSectionType.ROUND
        else:
            self.type_ = CrossSectionType.ROUND

    def dump(self,json_node:Dict[str,Any]) -> None:
        json_node["A"] = self.Ax_
        json_node["Jx"] = self.Jxx_
        json_node["Iy"] = self.Iyy_
        json_node["Iz"] = self.Izz_

class BarCrossSectionRound(BarCrossSection):

    def __init__(self, radius: float):
        super().__init__()
        self.radius_ = radius
        self.n_ = 10 # default discretization
        self.Ax_ = math.pi * radius ** 2
        self.Iyy_ = math.pi / 4.0 * radius ** 4
        self.Izz_ = math.pi / 4.0 * radius ** 4
        self.Jxx_ = math.pi / 2.0 * radius ** 4
        self.scale_ = 1.0
        self.type_ = CrossSectionType.ROUND


    def get_points(self) -> List[np.ndarray]:

        points = []
        for kd in range(self.n_):
            angle = float(kd) / self.n_ * math.pi * 2
            pt = np.array([math.cos(angle) * self.radius(), math.sin(angle) * self.radius()])

            points.append(pt)
        return points

    def dump(self, json_node: Dict[str, Any]) -> None:
        super().dump(json_node)
        json_node["type"] = "round"
        json_node["radius"] = self.radius()

    def radius(self) -> float:
        return self.scale_ * self.radius_


class BarCrossSectionRectangle(BarCrossSection):

    def __init__(self, width: float, height: float):
        super().__init__()
        self.width_ = width
        self.height_ = height
        self.Ax_ = width * height
        self.Iyy_ = 1.0 / 12.0 * width * height ** 3
        self.Izz_ = 1.0 / 12.0 * height * width ** 3
        self.Jxx_ = self.Izz_ + self.Iyy_
        self.scale_ = 1.0
        self.type_ = CrossSectionType.Rectangle

    def get_points(self) -> List[np.ndarray]:
        points = []
        points.append(np.array([-self.width() / 2, -self.height() / 2]))
        points.append(np.array([self.width() / 2, -self.height() / 2]))
        points.append(np.array([self.width() / 2, self.height() / 2]))
        points.append(np.array([-self.width() / 2, self.height() / 2]))
        return points

    def dump(self, json_node: Dict[str, Any]) -> None:
        super().dump(json_node)
        json_node["type"] = "rectangle"
        json_node["width"] = self.width()
        json_node["height"] = self.height()

    def width(self):
        return self.scale_ * self.width_

    def height(self):
        return self.scale_ * self.height_
