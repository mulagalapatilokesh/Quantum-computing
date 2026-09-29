from qaoa.qubo import QUBO, qubo_to_ising, ising_energy
from qaoa.classical import brute_force
from qaoa.engine import run_qaoa
import itertools

# ---- A tiny problem: 2 variables ----
tiny = QUBO(n=2, linear={0: -1, 1: -1}, quadratic={(0, 1): 2})

print("STEP A: energy of every bitstring (the computer does the 'paper work')")
for bits in itertools.product('01', repeat=2):
    b = ''.join(bits)
    print(f"  {b} -> {tiny.energy(b)}")

print("\nSTEP B: Ising conversion gives the SAME energies?")
h, J, off = qubo_to_ising(tiny)
for bits in itertools.product('01', repeat=2):
    b = ''.join(bits)
    print(f"  {b}: qubo={tiny.energy(b)}  ising={ising_energy(h, J, off, b)}")

# ---- Max-Cut on a 6-node graph ----
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4)]
q = QUBO(n=6)
for i, j in edges:
    q.linear[i] = q.linear.get(i, 0) - 1
    q.linear[j] = q.linear.get(j, 0) - 1
    q.quadratic[(i, j)] = q.quadratic.get((i, j), 0) + 2

print("\nSTEP C: QAOA vs brute force on Max-Cut (6 nodes)")
classical = brute_force(q)
quantum = run_qaoa(q, p=2)
print(f"  brute force : {classical['solution']}  energy={classical['energy']}  time={classical['time']:.4f}s")
print(f"  QAOA        : {quantum['solution']}  energy={quantum['energy']}  time={quantum['time']:.4f}s")
print(f"  optimizer steps: {len(quantum['history'])}   first={quantum['history'][0]:.3f}  last={quantum['history'][-1]:.3f}")
print("  QAOA found the best answer:", abs(quantum['energy'] - classical['energy']) < 1e-9)
