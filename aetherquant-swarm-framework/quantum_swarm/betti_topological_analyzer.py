#!/usr/bin/env python3
"""
Betti Loop & Topological Data Analysis (TDA) Module.
Computes Vietoris-Rips filtration, boundary operators over Z_2 field,
Betti numbers (beta_0, beta_1, beta_2), and persistent homology diagrams
to identify structural holes, loops, and voids in agent belief spaces.
"""

import numpy as np

try:
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from typing import List, Dict, Tuple, Any

class BettiTopologicalAnalyzer:
    def __init__(self, max_dimension: int = 2, metric: str = "euclidean"):
        self.max_dimension = max_dimension
        self.metric = metric

    def _pairwise_distances(self, X: np.ndarray, metric: str = None) -> np.ndarray:
        m = metric if metric else self.metric
        if m == "fubini_study":
            # Fubini-Study metric: d_FS([u], [v]) = arccos(|<u, v>| / (||u|| ||v||))
            norms = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12
            norm_X = X / norms
            if np.iscomplexobj(X):
                inner_prod = np.abs(np.dot(norm_X, norm_X.conj().T))
            else:
                inner_prod = np.abs(np.dot(norm_X, norm_X.T))
            inner_prod = np.clip(inner_prod, 0.0, 1.0)
            return np.arccos(inner_prod)
        elif HAS_SCIPY and m == "euclidean":
            return squareform(pdist(X, metric='euclidean'))
        else:
            # Fallback pure NumPy Euclidean distance
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _gauss_elim_z2(self, matrix: np.ndarray) -> int:
        """Computes rank of binary matrix over Z_2 field using Gaussian Elimination."""
        if matrix.size == 0:
            return 0
        mat = np.copy(matrix) % 2
        rows, cols = mat.shape
        pivot_row = 0

        for c in range(cols):
            # Find pivot
            pivot = -1
            for r in range(pivot_row, rows):
                if mat[r, c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue

            # Swap pivot row
            mat[[pivot_row, pivot]] = mat[[pivot, pivot_row]]

            # Eliminate below and above
            for r in range(rows):
                if r != pivot_row and mat[r, c] == 1:
                    mat[r] = (mat[r] + mat[pivot_row]) % 2

            pivot_row += 1
            if pivot_row >= rows:
                break

        return pivot_row

    def compute_betti_numbers(self, point_cloud: np.ndarray, epsilon: float) -> Dict[str, int]:
        """
        Computes Betti numbers (beta_0, beta_1, beta_2) for Vietoris-Rips complex at scale epsilon.
        beta_0: Number of connected components
        beta_1: Number of 1D topological loops (cycles/holes)
        beta_2: Number of 2D topological voids (enclosed cavities)
        """
        N = len(point_cloud)
        if N == 0:
            return {"beta_0": 0, "beta_1": 0, "beta_2": 0}
        if N == 1:
            return {"beta_0": 1, "beta_1": 0, "beta_2": 0}

        dist_matrix = self._pairwise_distances(point_cloud)

        # 0-simplices (vertices)
        simplices_0 = [(i,) for i in range(N)]

        # 1-simplices (edges where d <= epsilon)
        simplices_1 = []
        for i in range(N):
            for j in range(i + 1, N):
                if dist_matrix[i, j] <= epsilon:
                    simplices_1.append((i, j))

        # 2-simplices (triangles where all pairwise distances <= epsilon)
        simplices_2 = []
        num_1 = len(simplices_1)
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    if (dist_matrix[i, j] <= epsilon and 
                        dist_matrix[j, k] <= epsilon and 
                        dist_matrix[i, k] <= epsilon):
                        simplices_2.append((i, j, k))

        num_0 = N
        num_2 = len(simplices_2)

        # Boundary matrix d_1: 1-simplices -> 0-simplices
        if num_1 > 0:
            d_1 = np.zeros((num_0, num_1), dtype=int)
            for col, (u, v) in enumerate(simplices_1):
                d_1[u, col] = 1
                d_1[v, col] = 1
            rank_d1 = self._gauss_elim_z2(d_1)
        else:
            rank_d1 = 0

        # Boundary matrix d_2: 2-simplices -> 1-simplices
        if num_2 > 0 and num_1 > 0:
            edge_map = {edge: idx for idx, edge in enumerate(simplices_1)}
            d_2 = np.zeros((num_1, num_2), dtype=int)
            for col, (u, v, w) in enumerate(simplices_2):
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, w), max(v, w))
                e3 = (min(u, w), max(u, w))
                if e1 in edge_map:
                    d_2[edge_map[e1], col] = 1
                if e2 in edge_map:
                    d_2[edge_map[e2], col] = 1
                if e3 in edge_map:
                    d_2[edge_map[e3], col] = 1
            rank_d2 = self._gauss_elim_z2(d_2)
        else:
            rank_d2 = 0

        # Homology Betti numbers calculation:
        # beta_0 = dim(C_0) - rank(d_1)
        # beta_1 = dim(ker(d_1)) - rank(d_2) = (dim(C_1) - rank(d_1)) - rank(d_2)
        # beta_2 = dim(ker(d_2)) = dim(C_2) - rank(d_2)
        beta_0 = max(0, num_0 - rank_d1)
        beta_1 = max(0, (num_1 - rank_d1) - rank_d2)
        beta_2 = max(0, num_2 - rank_d2)

        return {
            "beta_0": int(beta_0),
            "beta_1": int(beta_1),
            "beta_2": int(beta_2),
            "num_edges": num_1,
            "num_triangles": num_2
        }

    def persistent_homology_sweep(self, point_cloud: np.ndarray, num_steps: int = 15) -> List[Dict[str, Any]]:
        """
        Sweeps epsilon filtration values from min to max distance, returning persistent homology barcodes.
        """
        if len(point_cloud) < 2:
            return [{"epsilon": 0.0, "beta_0": len(point_cloud), "beta_1": 0, "beta_2": 0}]

        dist_matrix = self._pairwise_distances(point_cloud)
        upper_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

        min_eps = float(np.min(upper_dists)) if len(upper_dists) > 0 else 0.0
        max_eps = float(np.max(upper_dists)) if len(upper_dists) > 0 else 1.0

        if min_eps == max_eps:
            max_eps += 1.0

        epsilons = np.linspace(min_eps, max_eps, num_steps)
        history = []

        for eps in epsilons:
            res = self.compute_betti_numbers(point_cloud, float(eps))
            res["epsilon"] = float(eps)
            history.append(res)

        return history
