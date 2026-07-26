import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mera_kmpa_mcp_server import (
    handle_holographic_consensus,
    handle_hierarchical_consensus,
    handle_betti_topology,
    handle_hpiv_critic,
    handle_dro_critic,
    handle_mine_apriori,
    handle_greedy_stack,
    handle_evolve_features
)

class TestMcpServerTools(unittest.TestCase):
    def test_mcp_consensus_tool(self):
        args = {
            "belief_vectors": [[1.0, 2.0, 3.0, 0.1], [0.5, 0.2, 0.1, 0.05]],
            "expected_returns": [100.0, 50.0]
        }
        res = handle_holographic_consensus(args)
        self.assertIn("winning_index", res)
        self.assertIn("confidence", res)

    def test_mcp_betti_topology_tool(self):
        args = {
            "point_cloud": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "epsilon": 1.05
        }
        res = handle_betti_topology(args)
        self.assertEqual(res["betti_numbers"]["beta_1"], 1)

    def test_mcp_hierarchical_consensus_tool(self):
        subgroups = [
            {
                "name": "QuantGroup",
                "belief_vectors": [[4.0, 5.0, 2.0, 1.0], [3.5, 4.0, 1.8, 0.9]],
                "expected_returns": [1200.0, 900.0]
            },
            {
                "name": "SaaSGroup",
                "belief_vectors": [[3.8, 4.0, 1.9, 0.8], [3.0, 3.2, 1.5, 0.7]],
                "expected_returns": [850.0, 700.0]
            }
        ]
        res = handle_hierarchical_consensus({"subgroup_tree": subgroups})
        self.assertIn("winning_subgroup_name", res)

if __name__ == '__main__':
    unittest.main()
