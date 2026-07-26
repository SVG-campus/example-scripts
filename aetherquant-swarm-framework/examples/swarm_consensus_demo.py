#!/usr/bin/env python3
"""
Example demonstrating the Holographic MERA Swarm Consensus Engine,
Hamiltonian Path-Integral validation (HPIV), and Wasserstein DRO.
"""
import sys
import os
import numpy as np

# Adjust path to import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.quantum_swarm_consensus import QuantumSwarmConsensusEngine

def main():
    print("====================================================")
    print("      Quantum Swarm Consensus & Critic Demo         ")
    print("====================================================")

    # Initialize Engine
    engine = QuantumSwarmConsensusEngine(h_bar=1.0, dt=0.05, target_safety_ratio=0.80)

    # Define candidate business strategies
    candidates = [
        "AetherQuant (High Returns, Medium Volatility)",
        "BlockSpread (Negative Returns due to Gas Drag)",
        "NicheStack SaaS (Stable Cashflow, Low Volatility)",
        "DeFi L2 Arbitrage Bot (High Returns, Low Volatility)"
    ]

    # Formulate 6D belief states: [LTV/CAC, breakeven_months, Sortino, mean, std, cvar]
    belief_vectors = np.array([
        [4.2, 5.0, 2.8, 1.2, 0.4, 0.1],  # AetherQuant
        [0.8, 12.0, -0.5, -0.2, 0.8, 0.7], # BlockSpread (MEV drag)
        [3.8, 4.0, 1.9, 0.8, 0.2, 0.05], # NicheStack SaaS
        [5.5, 3.0, 3.4, 1.5, 0.3, 0.08]  # DeFi L2 Arbitrage Bot
    ])

    expected_returns = np.array([1200.0, -180.0, 850.0, 1600.0])

    print("\n--- Running Holographic MERA Consensus (with KMPA & CYCF) ---")
    best_idx, confidence = engine.holographic_consensus(belief_vectors, expected_returns)

    print(f"Optimal Consensus Choice Resolved: {candidates[best_idx]}")
    print(f"Consensus Confidence: {confidence:.6f}")

    # Hamiltonian Path-Integral Verification (HPIV)
    print("\n--- Running HPIV Critic Checks on the Consensus Choice ---")
    is_safe_hpiv, hpiv_ratio = engine.hamiltonian_path_integral_critic(
        expected_return=expected_returns[best_idx],
        current_volatility=0.08,
        max_historical_drawdown=0.10
    )
    print(f"HPIV Validation Status: {'SAFE (PASS)' if is_safe_hpiv else 'RISKY (FAIL)'}")
    print(f"HPIV Safety Ratio: {hpiv_ratio:.6f}")

    # Wasserstein Distributionally Robust Optimization (DRO)
    print("\n--- Running Wasserstein DRO Robustness Critic ---")
    mock_returns_history = np.random.normal(expected_returns[best_idx] / 12.0, 20.0, 100)
    is_safe_dro, dro_loss = engine.wasserstein_dro_critic(
        returns_history=mock_returns_history,
        expected_return=expected_returns[best_idx]
    )
    print(f"Wasserstein DRO Status: {'ROBUST (PASS)' if is_safe_dro else 'VOLATILE (FAIL)'}")
    print(f"Worst-Case Perturbed Return Loss: {dro_loss:.6f}")
    print("====================================================")

if __name__ == "__main__":
    main()
