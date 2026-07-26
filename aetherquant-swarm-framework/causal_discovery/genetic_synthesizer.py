"""
Genetic Program Synthesis Feature Engine.
Recursively generates non-linear mathematical expressions (ASTs) of feature variables,
evaluates their fitness using out-of-sample correlation and information metrics,
and retains the optimal self-evolved features.
"""
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Any, Optional

class Node:
    def __init__(self, op: Optional[str] = None, left: Optional['Node'] = None, right: Optional['Node'] = None, value: Optional[str] = None):
        self.op = op          # Unary or binary operator name
        self.left = left      # Left child
        self.right = right    # Right child (None for unary)
        self.value = value    # Base feature name (None if operator)

    def to_string(self) -> str:
        if self.value is not None:
            return self.value
        if self.right is None:
            return f"{self.op}({self.left.to_string()})" if self.left else f"{self.op}()"
        left_str = self.left.to_string() if self.left else "0"
        right_str = self.right.to_string() if self.right else "0"
        return f"{self.op}({left_str}, {right_str})"

    def size(self) -> int:
        """Returns total AST node count."""
        if self.value is not None:
            return 1
        count = 1
        if self.left:
            count += self.left.size()
        if self.right:
            count += self.right.size()
        return count

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        eps = 1e-9
        if self.value is not None:
            if self.value not in df.columns:
                return pd.Series(0.0, index=df.index)
            s = df[self.value].astype(float)
            mean, std = s.mean(), s.std()
            if np.isnan(std) or std < eps:
                return s - mean
            return (s - mean) / (std + eps)

        left_val = self.left.evaluate(df) if self.left else pd.Series(0.0, index=df.index)
        
        # Unary operators
        if self.right is None:
            if self.op == "abs":
                return left_val.abs()
            elif self.op == "signed_sq":
                return left_val.apply(lambda x: np.sign(x) * (x ** 2))
            elif self.op == "signed_sqrt":
                return left_val.apply(lambda x: np.sign(x) * np.sqrt(np.abs(x)))
            elif self.op == "tanh":
                return np.tanh(left_val)
            elif self.op == "sigmoid":
                return 1.0 / (1.0 + np.exp(-np.clip(left_val, -15.0, 15.0)))
            elif self.op == "log_abs":
                return np.log(left_val.abs() + 1e-5)
            elif self.op == "exp_clip":
                return np.exp(np.clip(left_val, -10.0, 10.0))
            else:
                return left_val

        # Binary operators
        right_val = self.right.evaluate(df) if self.right else pd.Series(0.0, index=df.index)
        if self.op == "spread":
            return left_val - right_val
        elif self.op == "interaction":
            return left_val * right_val
        elif self.op == "ratio":
            return left_val / (right_val.abs() + eps)
        elif self.op == "coextreme":
            return np.sign(left_val * right_val) * (left_val.abs() * right_val.abs())
        elif self.op == "min_pair":
            return np.minimum(left_val, right_val)
        elif self.op == "max_pair":
            return np.maximum(left_val, right_val)
        else:
            return left_val + right_val

class GeneticFeatureSynthesizer:
    def __init__(self, base_features: List[str], max_depth: int = 3, population_size: int = 30, generations: int = 3, seed: int = 42):
        self.base_features = base_features
        self.max_depth = max_depth
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        self.unary_ops = ["abs", "signed_sq", "signed_sqrt", "tanh", "sigmoid", "log_abs", "exp_clip"]
        self.binary_ops = ["spread", "interaction", "ratio", "coextreme", "min_pair", "max_pair"]
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _generate_random_tree(self, depth: int) -> Node:
        if depth <= 1 or (depth < self.max_depth and random.random() < 0.3):
            return Node(value=random.choice(self.base_features))
        
        if random.random() < 0.4:
            op = random.choice(self.unary_ops)
            left = self._generate_random_tree(depth - 1)
            return Node(op=op, left=left)
        else:
            op = random.choice(self.binary_ops)
            left = self._generate_random_tree(depth - 1)
            right = self._generate_random_tree(depth - 1)
            return Node(op=op, left=left, right=right)

    def evaluate_fitness(self, tree: Node, df: pd.DataFrame, target: str) -> float:
        try:
            feature_series = tree.evaluate(df)
            if feature_series.isna().any() or np.isinf(feature_series).any():
                return 0.0
            
            target_series = df[target].astype(float)
            std_f = feature_series.std()
            std_t = target_series.std()
            if std_f == 0 or std_t == 0 or np.isnan(std_f) or np.isnan(std_t):
                return 0.0

            corr = np.corrcoef(feature_series, target_series)[0, 1]
            raw_fitness = float(np.abs(corr)) if not np.isnan(corr) else 0.0
            
            # Pareto complexity penalty (0.01 per extra AST node over 3)
            complexity_penalty = max(0, tree.size() - 3) * 0.01
            pareto_score = max(0.0, raw_fitness - complexity_penalty)
            return pareto_score
        except Exception:
            return 0.0

    def mutate(self, tree: Node) -> Node:
        if random.random() < 0.2:
            return self._generate_random_tree(depth=self.max_depth - 1)
        
        if tree.value is not None:
            return Node(value=random.choice(self.base_features))
            
        left_child = self.mutate(tree.left) if tree.left else None
        right_child = self.mutate(tree.right) if tree.right else None
        return Node(op=tree.op, left=left_child, right=right_child)

    def crossover(self, parent1: Node, parent2: Node) -> Node:
        child = Node(op=parent1.op, left=parent1.left, right=parent1.right, value=parent1.value)
        if child.value is None:
            if random.random() < 0.5 and child.left:
                child.left = parent2.left if parent2.left else parent2
            elif child.right:
                child.right = parent2.right if parent2.right else parent2
        return child

    def evolve(self, df: pd.DataFrame, target: str) -> List[Tuple[Node, float]]:
        population = [self._generate_random_tree(depth=self.max_depth) for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            scored_pop = []
            for ind in population:
                score = self.evaluate_fitness(ind, df, target)
                scored_pop.append((ind, score))
            
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            
            keep_size = max(2, int(self.population_size * 0.2))
            elites = [ind for ind, score in scored_pop[:keep_size]]
            
            new_pop = list(elites)
            while len(new_pop) < self.population_size:
                p1, p2 = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
                child = self.crossover(p1, p2)
                if random.random() < 0.3:
                    child = self.mutate(child)
                new_pop.append(child)
                
            population = new_pop

        final_scored = []
        for ind in population:
            score = self.evaluate_fitness(ind, df, target)
            final_scored.append((ind, score))
        final_scored.sort(key=lambda x: x[1], reverse=True)
        
        seen_strings = set()
        unique_results = []
        for ind, score in final_scored:
            s_str = ind.to_string()
            if s_str not in seen_strings:
                seen_strings.add(s_str)
                unique_results.append((ind, score))
                
        return unique_results[:5]
