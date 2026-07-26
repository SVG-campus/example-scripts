import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.betti_topological_analyzer import BettiTopologicalAnalyzer

class TestBettiTopologicalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = BettiTopologicalAnalyzer()

    def test_betti_point_cloud_basic(self):
        # 4 points forming a square (1-loop cycle)
        square_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])

        # At epsilon = 1.05, edges exist along square boundary but not diagonals (length ~1.414)
        betti_res = self.analyzer.compute_betti_numbers(square_points, epsilon=1.05)
        self.assertEqual(betti_res["beta_0"], 1) # 1 connected component
        self.assertEqual(betti_res["beta_1"], 1) # 1 topological loop/hole

        # At larger epsilon = 1.5, diagonals join, forming 2 triangles (fills hole)
        betti_filled = self.analyzer.compute_betti_numbers(square_points, epsilon=1.5)
        self.assertEqual(betti_filled["beta_0"], 1)
        self.assertEqual(betti_filled["beta_1"], 0) # Hole filled

    def test_persistent_homology_sweep(self):
        pts = np.random.normal(0, 1, (10, 3))
        sweep = self.analyzer.persistent_homology_sweep(pts, num_steps=5)
        self.assertEqual(len(sweep), 5)
    def test_betti_fubini_study(self):
        # 5 complex vectors on CP^1 creating a 5-cycle loop (beta_1 = 1)
        angles = [0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5]
        complex_pts = np.array([[np.cos(a), np.sin(a)] for a in angles], dtype=complex)
        
        fs_analyzer = BettiTopologicalAnalyzer(metric="fubini_study")
        res = fs_analyzer.compute_betti_numbers(complex_pts, epsilon=0.70)
        self.assertEqual(res["beta_0"], 1)
        self.assertEqual(res["beta_1"], 1) # Phase loop detected on CP^1 under Fubini-Study metric

if __name__ == '__main__':
    unittest.main()
