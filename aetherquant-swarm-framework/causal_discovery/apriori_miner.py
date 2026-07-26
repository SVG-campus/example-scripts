import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

def discretize_dataframe(df: pd.DataFrame, bins: int = 2, strategy: str = 'quantile') -> pd.DataFrame:
    """
    Discretizes continuous numerical columns into binary indicator variables (0.0 or 1.0).
    """
    df_disc = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        # Check if already binary
        unique_vals = set(s.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "true", "false", "True", "False"}):
            df_disc[col] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            median_val = s.median()
            df_disc[f"{col}_HIGH"] = (s > median_val).astype(float)
            df_disc[f"{col}_LOW"] = (s <= median_val).astype(float)
        else:
            # Categorical one-hot encoding
            dummies = pd.get_dummies(s, prefix=col, dtype=float)
            df_disc = pd.concat([df_disc, dummies], axis=1)
    return df_disc

def mine_itemsets(df: pd.DataFrame, min_support: float = 0.05, max_k: int = 3, auto_discretize: bool = True) -> Dict[int, List[Tuple[Tuple[str, ...], float]]]:
    """
    Mines boolean itemsets up to depth max_k using an Apriori-like pruning search.
    Supports boolean, numeric, and continuous discretized columns.
    Returns a dictionary mapping size k to a list of (itemset, support) tuples.
    """
    if auto_discretize:
        df_proc = discretize_dataframe(df)
    else:
        df_proc = df.copy()

    # Cast binary columns to float 0.0/1.0
    cols = []
    for col in df_proc.columns:
        unique_vals = set(df_proc[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            df_proc[col] = df_proc[col].astype(float)
            cols.append(col)

    n_samples = len(df_proc)
    if n_samples == 0 or not cols:
        return {}

    df_binary = df_proc[cols]
    results = {}

    # 1-Itemsets (k=1)
    k1_candidates = [((col,), float(df_binary[col].mean())) for col in cols]
    frequent_k1 = [item for item in k1_candidates if item[1] >= min_support]
    results[1] = frequent_k1

    # Track frequent itemsets for joint checks
    active_sets: Set[Tuple[str, ...]] = {item[0] for item in frequent_k1}

    for k in range(2, max_k + 1):
        prev_sets = sorted(list(active_sets))
        candidates = set()

        for i in range(len(prev_sets)):
            s1 = prev_sets[i]
            for j in range(i + 1, len(prev_sets)):
                s2 = prev_sets[j]
                if s1[:-1] == s2[:-1]:
                    union_set = tuple(sorted(set(s1).union(s2)))
                    subsets_ok = True
                    for item in union_set:
                        subset = tuple(x for x in union_set if x != item)
                        if subset not in active_sets:
                            subsets_ok = False
                            break
                    if subsets_ok:
                        candidates.add(union_set)
                else:
                    break

        frequent_k = []
        df_values = df_binary.values
        col_to_idx = {col: i for i, col in enumerate(df_binary.columns)}

        for candidate in candidates:
            idxs = [col_to_idx[col] for col in candidate]
            support = float(np.all(df_values[:, idxs] == 1.0, axis=1).mean())
            if support >= min_support:
                frequent_k.append((candidate, support))

        if not frequent_k:
            break

        results[k] = frequent_k
        active_sets = {item[0] for item in frequent_k}

    return results
