import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

def greedy_stack(df: pd.DataFrame, features: List[str], target: str, min_support: float = 0.05, eps: float = 0.0001) -> Dict[str, Any]:
    """
    Sequentially builds a logical conjunction (AND) of boolean features that maximizes the mean target outcome.
    Prunes additions that drop support below min_support or yield marginal lift < eps.
    """
    df_clean = df[[target] + features].dropna()
    n_samples = len(df_clean)
    if n_samples == 0:
        return {"rule": [], "mean_outcome": 0.0, "support": 0.0}
        
    baseline_mean = float(df_clean[target].mean())
    active_rule: List[str] = []
    
    # Convert features to numpy 2D boolean array and target to 1D float array
    feat_matrix = df_clean[features].values == 1.0
    target_values = df_clean[target].values
    
    current_mask = np.ones(n_samples, dtype=bool)
    current_mean = baseline_mean
    current_support = 1.0
    
    remaining_indices = list(range(len(features)))
    
    while remaining_indices:
        best_candidate_idx = None
        best_lift = -np.inf
        best_mask = None
        best_mean = current_mean
        best_support = current_support
        
        for idx in remaining_indices:
            # Try combining the active rule with this candidate
            candidate_mask = current_mask & feat_matrix[:, idx]
            support = float(np.sum(candidate_mask)) / n_samples
            
            if support < min_support:
                continue
                
            mean_val = float(np.mean(target_values[candidate_mask]))
            lift = mean_val - current_mean
            
            if lift > best_lift:
                best_lift = lift
                best_candidate_idx = idx
                best_mask = candidate_mask
                best_mean = mean_val
                best_support = support
                
        # If we found a candidate that improves the KPI by at least eps
        if best_candidate_idx is not None and best_lift >= eps:
            best_feature = features[best_candidate_idx]
            active_rule.append(best_feature)
            current_mask = best_mask
            current_mean = best_mean
            current_support = best_support
            remaining_indices.remove(best_candidate_idx)
        else:
            break
            
    return {
        "rule": active_rule,
        "mean_outcome": current_mean,
        "support": current_support,
        "lift_vs_baseline": current_mean - baseline_mean
    }
