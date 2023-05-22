from dataclasses import dataclass

@dataclass
class SimulationParams:
    nqva: int = 0
    nqv: int = 0
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nee: int = 0
    nsim: int = 0
    ntime: int= 0
    dt: float = 0.01
