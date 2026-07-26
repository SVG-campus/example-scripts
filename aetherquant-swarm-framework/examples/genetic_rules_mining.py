#!/usr/bin/env python3
"""
Example demonstrating Boolean Apriori rule mining,
Greedy Rule Stacking (GRS), and non-linear Genetic program feature synthesis.
"""
import sys
import os
import pandas as pd
import numpy as np

# Adjust path to import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_discovery.apriori_miner import mine_itemsets
from causal_discovery.greedy_stacker import greedy_stack
from causal_discovery.genetic_synthesizer import GeneticFeatureSynthesizer

def main():
    print("====================================================")
    print("       Causal Discovery & Rule Mining Demo          ")
    print("====================================================")

    # Generate synthetic binary indicators
    np.random.seed(42)
    n_days = 200
    
    # 5 boolean parameters
    vix_high = np.random.choice([0.0, 1.0], n_days, p=[0.7, 0.3])
    sentiment_neg = np.random.choice([0.0, 1.0], n_days, p=[0.8, 0.2])
    has_unsub = np.random.choice([0.0, 1.0], n_days, p=[0.3, 0.7])
    cmo_bullish = np.random.choice([0.0, 1.0], n_days, p=[0.6, 0.4])
    pvi_active = np.random.choice([0.0, 1.0], n_days, p=[0.5, 0.5])

    df = pd.DataFrame({
        "VIX_HIGH": vix_high,
        "SENTIMENT_NEG": sentiment_neg,
        "HAS_UNSUB": has_unsub,
        "CMO_BULLISH": cmo_bullish,
        "PVI_ACTIVE": pvi_active
    })

    # High conversions are driven by HAS_UNSUB and CMO_BULLISH
    conversions = 0.05 + 0.12 * df["HAS_UNSUB"] + 0.08 * df["CMO_BULLISH"] + np.random.normal(0, 0.02, n_days)
    df["CONVERSION_HIGH"] = (conversions > 0.09).astype(float)

    # 1. Apriori Boolean Miner
    print("\n--- Running Apriori Boolean Rule Miner ---")
    itemsets = mine_itemsets(df.drop(columns=["CONVERSION_HIGH"]), min_support=0.10, max_k=3)
    for k, entries in itemsets.items():
        print(f"Frequent {k}-Itemsets:")
        for item, support in entries[:5]:
            print(f"  - Features: {list(item)} | Support: {support:.4f}")

    # 2. Greedy Rule Stacking (GRS)
    print("\n--- Running Greedy Rule Stacking ---")
    stack_res = greedy_stack(
        df=df,
        features=["VIX_HIGH", "SENTIMENT_NEG", "HAS_UNSUB", "CMO_BULLISH", "PVI_ACTIVE"],
        target="CONVERSION_HIGH",
        min_support=0.05
    )
    print(f"Stacked Conjunction Rule: {' AND '.join(stack_res['rule'])}")
    print(f"Mean Target Outcome: {stack_res['mean_outcome']:.4f}")
    print(f"Rule Support: {stack_res['support']:.4f}")
    print(f"Lift vs Baseline: {stack_res['lift_vs_baseline']:.4f}")

    # 3. Agentic Genetic Feature Synthesizer
    print("\n--- Running Agentic Genetic Feature Synthesizer ---")
    synthesizer = GeneticFeatureSynthesizer(
        base_features=["VIX_HIGH", "SENTIMENT_NEG", "HAS_UNSUB", "CMO_BULLISH", "PVI_ACTIVE"],
        max_depth=2,
        population_size=15,
        generations=3,
        seed=42
    )
    best_genetic_features = synthesizer.evolve(df, target="CONVERSION_HIGH")
    print("Top evolved feature formulas (AST representations):")
    for tree, correlation in best_genetic_features[:3]:
        print(f"  - Formula: `{tree.to_string()}` | Correlation: {correlation:.6f}")
    print("====================================================")

if __name__ == "__main__":
    main()
