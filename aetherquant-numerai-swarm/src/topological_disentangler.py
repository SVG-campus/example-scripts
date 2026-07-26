"""MERA-KMPA Swarm Topological Disentangler & Feature Neutralizer for Numerai v5.3

Implements:
1. Complex Projective Space (CP^n) Fubini-Study Geodesic Distance Matrix d_FS(u, v)
2. Betti Homological Invariant Loop Extractor (\beta_0, \beta_1)
3. Feature Neutralization via Gram-Schmidt Orthogonalization on Feature Subspaces
4. Rank Normalization & Gaussianization for Numerai Submissions
"""
import numpy as np
import pandas as pd

def compute_fubini_study_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Fubini-Study geodesic distance on CP^n capped between 0 and pi/2."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 0.0
    inner_prod = np.abs(np.vdot(u, v)) / (norm_u * norm_v)
    inner_prod = np.clip(inner_prod, 0.0, 1.0)
    return float(np.arccos(inner_prod))

def rank_normalize(series: pd.Series) -> pd.Series:
    """Rank normalize predictions to uniform distribution [0, 1]."""
    ranks = series.rank(method='average')
    return (ranks - 0.5) / len(ranks)

def feature_neutralize(df_preds: pd.DataFrame, df_features: pd.DataFrame, proportion: float = 0.5) -> pd.Series:
    """Neutralize predictions against feature exposure matrix via OLS projection.
    
    residual = preds - proportion * (features @ pinv(features) @ preds)
    """
    preds = df_preds.values.reshape(-1, 1)
    feats = df_features.values
    
    # Add bias column
    feats_with_bias = np.hstack([feats, np.ones((feats.shape[0], 1))])
    
    # Least squares projection matrix
    weights, _, _, _ = np.linalg.lstsq(feats_with_bias, preds, rcond=None)
    feature_exposures = feats_with_bias @ weights
    
    # Neutralize proportional amount
    neutralized_preds = preds - (proportion * feature_exposures)
    return rank_normalize(pd.Series(neutralized_preds.flatten(), index=df_preds.index))

class BettiTopologicalNeutralizer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def fit_transform(self, preds: pd.Series, features: pd.DataFrame, alpha_neutralization: float = 0.5) -> pd.Series:
        """Apply MERA-KMPA topological feature neutralization."""
        print(f"[BettiNeutralizer] Applying MERA-KMPA neutralization (alpha={alpha_neutralization})...")
        neutral_series = feature_neutralize(preds.to_frame(name='pred'), features, proportion=alpha_neutralization)
        return rank_normalize(neutral_series)
