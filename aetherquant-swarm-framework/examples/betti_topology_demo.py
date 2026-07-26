#!/usr/bin/env python3
"""
Example demonstrating Betti Loop Topological Analysis (Persistent Homology)
to detect topological holes and cycles in high-dimensional agent belief spaces.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.betti_topological_analyzer import BettiTopologicalAnalyzer

def main():
    print("====================================================")
    print("      Betti Loop Topological Analyzer Demo          ")
    print("====================================================")

    analyzer = BettiTopologicalAnalyzer()

    # Generate 8 agents whose belief vectors form a topological 1-loop (ring) in 4D space
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ring_points = np.stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(8),
        np.zeros(8)
    ], axis=1)

    print("\n--- Computing Betti Numbers for Agent Belief Ring ---")
    betti_single = analyzer.compute_betti_numbers(ring_points, epsilon=0.9)
    print(f"Betti Numbers at filtration epsilon=0.9:")
    print(f"  - Beta_0 (Connected Components): {betti_single['beta_0']}")
    print(f"  - Beta_1 (1D Topological Loops/Holes): {betti_single['beta_1']}")
    print(f"  - Beta_2 (2D Voids): {betti_single['beta_2']}")

    print("\n--- Persistent Homology Filtration Sweep ---")
    sweep = analyzer.persistent_homology_sweep(ring_points, num_steps=6)
    for step in sweep:
        print(f"Eps: {step['epsilon']:.4f} | Beta_0: {step['beta_0']} | Beta_1: {step['beta_1']} | Edges: {step['num_edges']} | Triangles: {step['num_triangles']}")

    print("====================================================")

if __name__ == "__main__":
    main()
