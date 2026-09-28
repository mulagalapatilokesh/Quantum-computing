"""QUBO problem definition and QUBO -> Ising conversion."""
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class QUBO:
    n: int                                                        # number of binary variables
    linear: Dict[int, float] = field(default_factory=dict)        # {i: cost of x_i}
    quadratic: Dict[Tuple[int, int], float] = field(default_factory=dict)  # {(i,j): cost of x_i*x_j}
    offset: float = 0.0

    def energy(self, bits: str) -> float:
        """Cost of a bitstring like '011'."""
        x = [int(b) for b in bits]
        e = self.offset
        e += sum(c * x[i] for i, c in self.linear.items())
        e += sum(c * x[i] * x[j] for (i, j), c in self.quadratic.items())
        return e


def qubo_to_ising(q: QUBO):
    """Rewrite with spins: x = (1 - z) / 2.
    Result: H = sum h_i Z_i + sum J_ij Z_i Z_j + offset
    """
    h, J, off = {}, {}, q.offset
    for i, c in q.linear.items():
        off += c / 2
        h[i] = h.get(i, 0.0) - c / 2
    for (i, j), c in q.quadratic.items():
        off += c / 4
        h[i] = h.get(i, 0.0) - c / 4
        h[j] = h.get(j, 0.0) - c / 4
        J[(i, j)] = J.get((i, j), 0.0) + c / 4
    return h, J, off


def ising_energy(h, J, off, bits: str) -> float:
    """Energy of a bitstring using the Ising form (bit 0 -> z=+1, bit 1 -> z=-1)."""
    z = [1 - 2 * int(b) for b in bits]
    return off + sum(c * z[i] for i, c in h.items()) + sum(c * z[i] * z[j] for (i, j), c in J.items())