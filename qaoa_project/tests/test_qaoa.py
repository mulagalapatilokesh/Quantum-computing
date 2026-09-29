import itertools, unittest
from qaoa.qubo import QUBO, qubo_to_ising, ising_energy
from qaoa.classical import brute_force
from qaoa.engine import run_qaoa


def maxcut_qubo(n, edges):
    q = QUBO(n=n)
    for i, j in edges:
        q.linear[i] = q.linear.get(i, 0) - 1
        q.linear[j] = q.linear.get(j, 0) - 1
        q.quadratic[(i, j)] = q.quadratic.get((i, j), 0) + 2
    return q


class TestQUBO(unittest.TestCase):
    def test_energy_by_hand(self):
        q = QUBO(n=2, linear={0: -1, 1: -1}, quadratic={(0, 1): 2})
        self.assertEqual(q.energy("00"), 0)
        self.assertEqual(q.energy("01"), -1)
        self.assertEqual(q.energy("10"), -1)
        self.assertEqual(q.energy("11"), 0)


class TestIsing(unittest.TestCase):
    def test_ising_matches_qubo_for_every_bitstring(self):
        q = maxcut_qubo(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        q.offset = 1.5
        h, J, off = qubo_to_ising(q)
        for bits in itertools.product("01", repeat=4):
            b = "".join(bits)
            self.assertAlmostEqual(q.energy(b), ising_energy(h, J, off, b))


class TestBruteForce(unittest.TestCase):
    def test_finds_minimum(self):
        q = QUBO(n=2, linear={0: -1, 1: -1}, quadratic={(0, 1): 2})
        self.assertEqual(brute_force(q)["energy"], -1)


class TestQAOA(unittest.TestCase):
    def test_reaches_optimum_on_small_maxcut(self):
        q = maxcut_qubo(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        best = brute_force(q)["energy"]
        result = run_qaoa(q, p=2)
        self.assertAlmostEqual(result["energy"], best)


if __name__ == "__main__":
    unittest.main()
