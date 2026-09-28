"""Classical brute-force solver: our ground truth."""
import itertools, time
from qaoa.qubo import QUBO


def brute_force(qubo: QUBO):
    t0 = time.perf_counter()
    best = min((''.join(b) for b in itertools.product('01', repeat=qubo.n)), key=qubo.energy)
    return {"solution": best, "energy": qubo.energy(best), "time": time.perf_counter() - t0}
