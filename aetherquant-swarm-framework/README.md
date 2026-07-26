# MERA-KMPA Swarm & Causal Discovery Framework

A unified, self-contained mathematical framework for quantum-cognitive swarm consensus, persistent homology Betti loop topological analysis, causal discovery, and non-linear genetic feature synthesis. Designed for local quantized LLMs, Antigravity Swarms, FastMCP servers, and autonomous agent loops.

---

## 🌌 1. Core Mathematical Foundations

### A. MERA (Multiscale Entanglement Renormalization Ansatz)
Maps multi-dimensional belief vectors of arbitrary real dimension $d$ into complex Hilbert spaces $\mathbb{C}^{\lceil d/2 \rceil}$. Performs **Variational MERA Disentangling Contractions** layer-by-layer to minimize entanglement entropy:
$$S(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
Projects bulk states down to a unified consensus vector without losing topological correlation. Supports **Hierarchical $N$-th Subgroup Swarm Consensus** for nested agent sub-subgroups.

### B. KMPA (Kähler Manifold Phase Alignment)
Resolves phase alignments on complex projective spaces $\mathbb{CP}^n$ using the **Fubini-Study Geodesic Distance**:
$$d(\psi, \phi) = \arccos\left(\frac{|\langle\psi, \phi\rangle|}{\|\psi\|\|\phi\|}\right)$$
Flows tangent vectors iteratively toward the barycenter to establish a phase-aligned consensus state.

### C. CYCF (Calabi-Yau Cohomology Flow)
Deforms Kähler metrics by solving the complex **Monge-Ampère Equation**:
$$\det\left(g_{i\bar{j}} + \partial_i \bar{\partial}_j \phi\right) = e^F \det(g_{i\bar{j}})$$
Numeric-gradient-deforms boundary states for $K$-dimensional complex spaces to match volume distributions of target weights.

### D. Betti Loop Persistent Homology (TDA)
Computes Vietoris-Rips filtration and boundary matrix reduction over the $\mathbb{Z}_2$ field to extract **Betti Numbers**:
- $\beta_0$: Number of connected agent components.
- $\beta_1$: 1D topological cycles / feedback loops / structural holes in belief landscapes.
- $\beta_2$: 2D enclosed cavities and voids.

### E. HPIV (Hamiltonian Path-Integral Verification)
Evaluates classical Lagrangian Action $S = \int (T - V) \, dt$ along candidate trajectories to compute partition functions:
$$P(\text{safe}) = \frac{e^{-S_{\text{nominal}} / \hbar}}{\sum e^{-S_i / \hbar}}$$

### F. Wasserstein DRO (Distributionally Robust Optimization)
Certifies out-of-distribution (OOD) robustness under worst-case perturbations within a Wasserstein ball of radius $\epsilon$:
$$\min_{P: W(P, P_0) \le \epsilon} \mathbb{E}_P[f(X)]$$

---

## 🔍 2. Causal Discovery & Rule Mining

### A. Agentic Genetic Feature Synthesizer (AGFS)
Recursively evolves AST feature expressions composed of unary (`abs`, `signed_sq`, `signed_sqrt`, `tanh`, `sigmoid`, `log_abs`, `exp_clip`) and binary (`spread`, `interaction`, `ratio`, `coextreme`, `min_pair`, `max_pair`) operators using Pareto complexity-penalized fitness.

### B. Apriori Boolean Miner & Auto-Discretization
Discretizes continuous variables and mines frequent boolean itemsets via Apriori pruning search.

### C. Greedy Rule Stacking (GRS)
Sequentially stacks logical `AND` constraints to form rules maximizing target metric lift.

---

## 🔌 3. Model Context Protocol (MCP) Server

Mount the stdio FastMCP server to grant LLM agents direct tool access to MERA consensus, Betti loop topology, HPIV, DRO, and genetic feature evolution:

```json
{
  "mcpServers": {
    "mera-kmpa-swarm-framework": {
      "command": "python",
      "args": [
        "C:/Users/svillalobosgonzalez1/Documents/GitHub/mera-kmpa-swarm-framework/mera_kmpa_mcp_server.py"
      ]
    }
  }
}
```

Available MCP Tools:
- `holographic_consensus`
- `hierarchical_consensus`
- `betti_topology`
- `hpiv_critic`
- `dro_critic`
- `mine_apriori`
- `greedy_stack`
- `evolve_features`

---

## 🛠️ 4. Installation & Usage

### Setup
```bash
git clone https://github.com/SVG-campus/mera-kmpa-swarm-framework.git
cd mera-kmpa-swarm-framework
pip install -r requirements.txt
```

### Running Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### CLI Interface
```bash
# Run MCP server
python cli.py serve

# Run Betti loop analysis on 2D/4D point cloud
python cli.py betti --points "[[0,0],[1,0],[1,1],[0,1]]" --epsilon 1.05
```
