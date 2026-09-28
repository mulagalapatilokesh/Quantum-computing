# QAOA Project

A step-by-step build of QAOA with NVIDIA CUDA-Q. Every step done in the Codespace is logged here.

Setup: GitHub Codespaces, Python 3.14, CUDA-Q 0.16.0, NumPy, SciPy, CPU simulator qpp-cpu.

## Step 2: Problem definition, `qaoa/qubo.py` (done)
Why: the computer needs a way to store a problem and score an answer. This file has no quantum code,
which keeps the problem separate from the quantum part.

A QUBO is a problem where each item is taken (1) or left (0). The total cost has a cost for each item
alone (`linear`) and an extra cost for pairs (`quadratic`). The cost of a whole answer such as `01` is
called its energy. Lower is better.

The file contains:
- `QUBO` class: holds `n`, `linear`, `quadratic` and `offset`
- `energy(bits)`: the cost of a bitstring. All solvers are scored with it
- `qubo_to_ising(q)`: rewrites the problem with spins (+1/-1) using `x = (1 - z) / 2`, because quantum
  circuits need that form. It returns `h` (single terms), `J` (pair terms) and an offset
- `ising_energy(...)`: energy from the spin form, used to prove the conversion is correct

Check (run from inside `qaoa_project`):
    python3 -c "
    from qaoa.qubo import QUBO, qubo_to_ising, ising_energy
    q = QUBO(n=2, linear={0:-1, 1:-1}, quadratic={(0,1):2})
    h, J, off = qubo_to_ising(q)
    for b in ['00','01','10','11']:
        print(b, 'qubo =', q.energy(b), ' ising =', ising_energy(h, J, off, b))
    "
Result: the QUBO and Ising columns match on every line (00 -> 0, 01 -> -1, 10 -> -1, 11 -> 0),
so the conversion is correct.

## Step 3: Classical answer key, `qaoa/classical.py` (done)
Why: to know whether a quantum answer is good, we need the true best answer. Brute force tries every
possible bitstring and keeps the one with the lowest energy. It is always correct on small problems, so
it is our ground truth. Later it is the classical side of the quantum vs classical comparison.

How it works: `itertools.product('01', repeat=n)` makes all `2^n` bitstrings, `min(..., key=qubo.energy)`
keeps the cheapest, and the function records the run time. The count doubles with every extra variable
(20 variables is about a million answers), so brute force cannot scale to large problems.

Result of the check:
    tiny problem  -> 01 -1.0
    max-cut graph -> 010101 -8.0
Energy -8 on the 8-edge graph means every edge is cut, the best possible.

## Progress
- [x] Step 0: Codespace and CUDA-Q setup
- [x] Step 1: Folders and helper files
- [x] Step 2: qubo.py
- [x] Step 3: classical.py
- [x] Step 4: engine.py (the QAOA circuit)
- [ ] Step 5: run_demo.py
- [ ] Step 6: tests

## Step 4: The QAOA engine, `qaoa/engine.py` (done)
Why: this is QAOA itself. It builds the quantum circuit, lets a classical optimizer tune it, and reads out
the answer.

The idea is a hybrid loop. A quantum circuit cannot tune its own angles, so a classical optimizer
proposes angles, the circuit runs and reports the average energy, and the optimizer proposes better
angles. This repeats. At the end the circuit runs many times and each run gives one bitstring (an
answer). The best one is kept.

The file contains three functions:
- `build_circuit(n, p, h, J)`: Hadamard gates start every qubit in an equal mix of 0 and 1. Then `p`
  layers follow. Each layer has a cost part (`rz` for single terms, `cx-rz-cx` for pair terms, controlled
  by the angle `gamma`) and a mixer part (`rx` on every qubit, controlled by the angle `beta`). All angles
  come from one list so the optimizer can change them.
- `build_hamiltonian(h, J)`: the energy operator, built from Pauli-Z terms, that CUDA-Q uses to compute
  the average energy.
- `run_qaoa(...)`: converts the QUBO to Ising, builds the circuit, runs the COBYLA optimizer with
  `cudaq.observe` as the objective, samples the final circuit with `cudaq.sample`, and returns the lowest-energy
  bitstring together with the optimizer history.

Settings of `run_qaoa`: `p` (layers, default 2), `shots` (final measurements, default 2000), `max_iter`
(optimizer steps, default 150), `seed` (default 42), `backend` (default `qpp-cpu`).

Result of the check on the 2-variable problem: solution `10` (or `01`), energy -1.0, and the counts split
about 50/50 between `10` and `01`, the two best answers. On the 6-node Max-Cut graph QAOA found
`010101` with energy -8.0, the same as brute force. The optimizer lowered the average energy from -3.649
to -4.854 over 142 steps. Brute force is far faster on problems this small, which is expected.
