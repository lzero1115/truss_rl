from typing import Dict, Any, Optional
from .BarCrossSection import BarCrossSection


class BarMaterial:
    """Material properties of the beam"""

    def __init__(self,
                 E: Optional[float] = None,
                 mu: Optional[float] = None,
                 rho: Optional[float] = None,
                 section: Optional[BarCrossSection] = None):
        """Initialize bar material properties.

        Args:
            E: Young's modulus (Pa)
            mu: Poisson's ratio
            rho: Density (N/m³)
            section: Cross section properties
        """
        self.E_: float = E
        self.mu_: float = mu
        self.rho_: float = rho
        self.mat_section_: BarCrossSection = section

    @classmethod
    def from_material(cls, mat: 'BarMaterial') -> 'BarMaterial':
        """Create a new material by copying an existing one."""
        return cls(E=mat.E_, mu=mat.mu_, rho=mat.rho_, section=mat.mat_section_)

    def dump(self, json_node: Dict[str, Any]) -> None:
        """Dump material properties to JSON."""
        json_node["E"] = self.E_
        json_node["G12"] = self.E_ / (2 * (1 + self.mu_))
        json_node["density"] = self.rho_
        json_cross_sec = {}
        self.mat_section_.dump(json_cross_sec)
        json_node["cross_sec"] = json_cross_sec

    def Ax(self) -> float:
        """Get cross-sectional area."""
        return self.mat_section_.Ax_

    def Jxx(self) -> float:
        """Get torsional constant."""
        return self.mat_section_.Jxx_

    def Iyy(self) -> float:
        """Get second moment of area about y-axis."""
        return self.mat_section_.Iyy_

    def Izz(self) -> float:
        """Get second moment of area about z-axis."""
        return self.mat_section_.Izz_