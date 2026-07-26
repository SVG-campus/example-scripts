import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_discovery.apriori_miner import mine_itemsets
from causal_discovery.greedy_stacker import greedy_stack
from causal_discovery.genetic_synthesizer import GeneticFeatureSynthesizer

class TestCausalDiscovery(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.df = pd.DataFrame({
            "A": np.random.choice([0.0, 1.0], self.n_samples, p=[0.4, 0.6]),
            "B": np.random.choice([0.0, 1.0], self.n_samples, p=[0.3, 0.7]),
            "C": np.random.choice([0.0, 1.0], self.n_samples, p=[0.5, 0.5])
        })
        # Target strongly correlated with A and B
        conversions = 0.05 + 0.15 * self.df["A"] + 0.10 * self.df["B"] + np.random.normal(0, 0.01, self.n_samples)
        self.df["TARGET"] = (conversions > 0.10).astype(float)

    def test_apriori_miner(self):
        results = mine_itemsets(self.df.drop(columns=["TARGET"]), min_support=0.10, max_k=2)
        self.assertIn(1, results)
        self.assertIn(2, results)
        # Verify support bounds
        for k, items in results.items():
            for itemset, support in items:
                self.assertTrue(0.10 <= support <= 1.0)

    def test_greedy_stacker(self):
        res = greedy_stack(
            df=self.df,
            features=["A", "B", "C"],
            target="TARGET",
            min_support=0.05
        )
        self.assertIn("rule", res)
        self.assertIn("lift_vs_baseline", res)
        # A and/or B should be part of the optimal rule
        self.assertTrue(len(res["rule"]) > 0)

    def test_genetic_synthesizer(self):
        synthesizer = GeneticFeatureSynthesizer(
            base_features=["A", "B", "C"],
            max_depth=2,
            population_size=10,
            generations=2,
            seed=42
        )
        best_features = synthesizer.evolve(self.df, target="TARGET")
        self.assertTrue(len(best_features) > 0)
        # Check fitness format
        for node, score in best_features:
            self.assertTrue(0.0 <= score <= 1.0)
            self.assertTrue(isinstance(node.to_string(), str))

if __name__ == '__main__':
    unittest.main()
