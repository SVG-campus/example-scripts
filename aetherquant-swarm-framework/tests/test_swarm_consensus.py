import unittest
import numpy as np
import sys
import os

# Adjust path to import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.quantum_swarm_consensus import QuantumSwarmConsensusEngine

class TestQuantumSwarmConsensus(unittest.TestCase):
    def setUp(self):
        self.engine = QuantumSwarmConsensusEngine(target_safety_ratio=0.80)

    def test_holographic_consensus_basic(self):
        # 3 candidate strategies, 6D belief states
        belief_vectors = np.array([
            [3.5, 4.0, 2.1, 1.2, 0.5, 0.2],  # Strategy 0: Strong metrics
            [1.2, 8.0, 0.4, 0.2, 0.9, 0.6],  # Strategy 1: Weak/high breakeven
            [2.8, 5.0, 1.8, 1.0, 0.6, 0.3]   # Strategy 2: Moderate metrics
        ])
        expected_returns = np.array([1500.0, -200.0, 800.0])

        best_idx, confidence = self.engine.holographic_consensus(belief_vectors, expected_returns)
        
        # Strategy 0 should be selected as it has best LTV/CAC and returns
        self.assertEqual(best_idx, 0)
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_hpiv_critic_nominal(self):
        # Good expected return, low risk
        is_safe, safety_ratio = self.engine.hamiltonian_path_integral_critic(
            expected_return=1200.0,
            current_volatility=0.05,
            max_historical_drawdown=0.08
        )
        self.assertTrue(is_safe)
        self.assertTrue(safety_ratio >= 0.80)

    def test_hpiv_critic_unsafe(self):
        # Negative expected return, high risk (high volatility and drawdown exceed nominal bounds)
        is_safe, safety_ratio = self.engine.hamiltonian_path_integral_critic(
            expected_return=-500.0,
            current_volatility=4.5,
            max_historical_drawdown=4.5
        )
        self.assertFalse(is_safe)
        self.assertTrue(safety_ratio < 0.80)

    def test_holographic_consensus_arbitrary_dimensions(self):
        # 4D belief vectors
        b4 = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
        r4 = np.array([100.0, 50.0])
        idx4, conf4 = self.engine.holographic_consensus(b4, r4)
        self.assertEqual(idx4, 0)

        # 8D belief vectors with distinct directions
        b8 = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0]])
        r8 = np.array([50.0, 200.0])
        idx8, conf8 = self.engine.holographic_consensus(b8, r8)
        self.assertEqual(idx8, 1)

    def test_hierarchical_swarm_consensus(self):
        subgroups = [
            {
                "name": "SubgroupAlpha",
                "belief_vectors": [[3.5, 4.0, 2.1, 1.2], [1.2, 8.0, 0.4, 0.2]],
                "expected_returns": [1500.0, -200.0]
            },
            {
                "name": "SubgroupBeta",
                "subgroups": [
                    {
                        "name": "SubSubGroup_1",
                        "belief_vectors": [[5.0, 2.0, 3.0, 1.0], [2.0, 1.0, 1.0, 0.5]],
                        "expected_returns": [2000.0, 500.0]
                    }
                ]
            }
        ]
        res = self.engine.hierarchical_swarm_consensus(subgroups)
        self.assertIn("winning_subgroup_name", res)
        self.assertEqual(res["winning_subgroup_name"], "SubgroupBeta")

if __name__ == '__main__':
    unittest.main()
