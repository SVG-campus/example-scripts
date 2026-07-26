#!/usr/bin/env python3
"""
TDA Persistence Diagram & Betti Barcode Exporter for MERA-KMPA Framework.
Computes multi-scale Vietoris-Rips filtration, Betti curves (\beta_0, \beta_1, \beta_2),
and exports persistence barcodes as JSON for real-time AetherQuant API integration.
"""

import os
import sys
import json
import numpy as np

# Adjust path to import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_swarm.betti_topological_analyzer import BettiTopologicalAnalyzer

def generate_tda_persistence_report(point_cloud: np.ndarray, max_eps: float = 2.0, num_steps: int = 10) -> dict:
    """
    Computes Betti filtration across epsilon scale parameter [0.1, max_eps].
    Returns persistence barcode metadata and Betti numbers.
    """
    analyzer = BettiTopologicalAnalyzer()
    epsilons = np.linspace(0.1, max_eps, num_steps)
    betti_curves = []

    for eps in epsilons:
        res = analyzer.compute_betti_numbers(point_cloud, epsilon=float(eps))
        betti_curves.append({
            "epsilon": round(float(eps), 4),
            "betti_0": int(res["beta_0"]),
            "betti_1": int(res["beta_1"]),
            "betti_2": int(res["beta_2"])
        })

    # Summary persistence statistics
    b0_max = max(b["betti_0"] for b in betti_curves)
    b1_max = max(b["betti_1"] for b in betti_curves)
    b2_max = max(b["betti_2"] for b in betti_curves)

    return {
        "framework": "MERA-KMPA TDA Engine v1.0",
        "point_count": len(point_cloud),
        "dimensions": point_cloud.shape[1] if point_cloud.ndim > 1 else 1,
        "max_betti_0_components": b0_max,
        "max_betti_1_cycles": b1_max,
        "max_betti_2_voids": b2_max,
        "betti_filtration_curve": betti_curves,
        "market_regime_classification": "STABLE_HOMOLOGY" if b1_max <= 2 else "HIGH_TURBULENCE_PHASE_TRANSITION"
    }

def main():
    print("====================================================")
    print("      TDA Persistence Barcode & Betti Exporter      ")
    print("====================================================")

    # Simulate 4D financial manifold point cloud (Return, Volatility, Skewness, Liquidity)
    np.random.seed(42)
    sample_point_cloud = np.random.normal(0, 1, (10, 4))

    report = generate_tda_persistence_report(sample_point_cloud, max_eps=2.5, num_steps=10)
    print(json.dumps(report, indent=2))

    output_path = os.path.join(os.path.dirname(__file__), "tda_persistence_sample.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Persistence Barcode Exported Successfully: {output_path}")
    print("====================================================")

if __name__ == "__main__":
    main()
