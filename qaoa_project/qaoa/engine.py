"""QAOA on CUDA-Q: circuit + classical optimizer + sampling."""
import time, warnings
warnings.filterwarnings("ignore")
import cudaq
from qaoa.qubo import QUBO, qubo_to_ising


def build_circuit(n, p, h, J):
    """p layers of (cost, mixer) after an initial Hadamard layer."""
    kernel, theta = cudaq.make_kernel(list)
    q = kernel.qalloc(n)
    for i in range(n):
        kernel.h(q[i])                                        # equal superposition of all answers
    for layer in range(p):
        gamma, beta = theta[2 * layer], theta[2 * layer + 1]
        for i, c in h.items():
            kernel.rz(2.0 * gamma * c, q[i])                  # cost: single-variable terms
        for (i, j), c in J.items():
            kernel.cx(q[i], q[j])
            kernel.rz(2.0 * gamma * c, q[j])                  # cost: pair terms
            kernel.cx(q[i], q[j])
        for i in range(n):
            kernel.rx(2.0 * beta, q[i])                       # mixer
    return kernel


def build_hamiltonian(h, J):
    """The energy operator that CUDA-Q measures."""
    H = 0.0 * cudaq.spin.z(0)
    for i, c in h.items():
        H += c * cudaq.spin.z(i)
    for (i, j), c in J.items():
        H += c * cudaq.spin.z(i) * cudaq.spin.z(j)
    return H


def run_qaoa(qubo: QUBO, p=2, shots=2000, max_iter=150, seed=42, backend="qpp-cpu"):
    t0 = time.perf_counter()
    try:
        cudaq.set_target(backend)
    except Exception:
        backend = "qpp-cpu"
        cudaq.set_target("qpp-cpu")
    cudaq.set_random_seed(seed)

    h, J, off = qubo_to_ising(qubo)
    kernel = build_circuit(qubo.n, p, h, J)
    H = build_hamiltonian(h, J)

    history = []                                              # energy at every optimizer step

    def objective(params):
        value = float(cudaq.observe(kernel, H, params).expectation()) + off
        history.append(value)
        return value

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = max_iter
    optimizer.initial_parameters = [0.4, 0.7] * p
    best_expectation, best_params = optimizer.optimize(2 * p, objective)

    counts = cudaq.sample(kernel, best_params, shots_count=shots)
    best_bits = min(counts.items(), key=lambda kv: qubo.energy(kv[0]))[0]

    return {
        "solution": best_bits,
        "energy": qubo.energy(best_bits),
        "expectation": best_expectation,
        "params": list(best_params),
        "history": history,
        "counts": dict(counts.items()),
        "p": p,
        "backend": backend,
        "time": time.perf_counter() - t0,
    }
