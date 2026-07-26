#!/usr/bin/env python3
"""
Example demonstrating Hierarchical N-th Subgroup Swarm Consensus.
Recursively contracts agent belief vectors from sub-subgroups -> subgroups -> global root consensus.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.quantum_swarm_consensus import QuantumSwarmConsensusEngine

def main():
    print("====================================================")
    print("     Hierarchical N-th Subgroup Consensus Demo      ")
    print("====================================================")

    engine = QuantumSwarmConsensusEngine()

    # Define nested subgroup tree structure down to sub-subgroups
    subgroups = [
        {
            "name": "Quantitative High-Frequency Subgroup",
            "belief_vectors": [
                [4.2, 5.0, 2.8, 1.2, 0.4, 0.1],  # Agent 1
                [3.9, 4.8, 2.5, 1.1, 0.3, 0.1]   # Agent 2
            ],
            "expected_returns": [1500.0, 1350.0]
        },
        {
            "name": "Fundamental SaaS Subgroup",
            "subgroups": [
                {
                    "name": "Enterprise B2B Sub-Subgroup",
                    "belief_vectors": [
                        [3.8, 4.0, 1.9, 0.8, 0.2, 0.05],
                        [3.5, 3.8, 1.7, 0.7, 0.2, 0.04]
                    ],
                    "expected_returns": [900.0, 800.0]
                },
                {
                    "name": "PLG Developer Tools Sub-Subgroup",
                    "belief_vectors": [
                        [5.2, 3.1, 3.2, 1.4, 0.3, 0.06],
                        [4.8, 3.0, 3.0, 1.3, 0.3, 0.05]
                    ],
                    "expected_returns": [1800.0, 1650.0]
                }
            ]
        }
    ]

    print("\n--- Contracting Hierarchical Subgroup Consensus ---")
    res = engine.hierarchical_swarm_consensus(subgroups)

    print(f"Global Winning Subgroup: {res['winning_subgroup_name']}")
    print(f"Root Consensus Confidence: {res['root_confidence']:.6f}")
    print(f"Winning Consensus Return: ${res['consensus_return']:.2f}")
    print("====================================================")

if __name__ == "__main__":
    main()
