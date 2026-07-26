#!/usr/bin/env python3
"""
MERA-KMPA Swarm Framework FastMCP Server.
Exposes holographic MERA consensus, hierarchical subgroup consensus,
Betti loop persistent homology, HPIV critic, Wasserstein DRO critic,
Apriori itemset mining, Greedy Rule Stacking, and Genetic Feature Synthesis over MCP JSON-RPC.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_swarm.quantum_swarm_consensus import QuantumSwarmConsensusEngine
from quantum_swarm.betti_topological_analyzer import BettiTopologicalAnalyzer
from causal_discovery.apriori_miner import mine_itemsets, discretize_dataframe
from causal_discovery.greedy_stacker import greedy_stack
from causal_discovery.genetic_synthesizer import GeneticFeatureSynthesizer

consensus_engine = QuantumSwarmConsensusEngine()
betti_analyzer = BettiTopologicalAnalyzer()

def handle_holographic_consensus(args: Dict[str, Any]) -> Dict[str, Any]:
    belief_vectors = np.array(args["belief_vectors"], dtype=float)
    expected_returns = np.array(args["expected_returns"], dtype=float)
    best_idx, confidence = consensus_engine.holographic_consensus(belief_vectors, expected_returns)
    return {
        "winning_index": best_idx,
        "confidence": confidence,
        "selected_vector": belief_vectors[best_idx].tolist()
    }

def handle_hierarchical_consensus(args: Dict[str, Any]) -> Dict[str, Any]:
    subgroup_tree = args["subgroup_tree"]
    res = consensus_engine.hierarchical_swarm_consensus(subgroup_tree)
    # Convert numpy arrays in result to lists for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        return obj
    return _convert(res)

def handle_betti_topology(args: Dict[str, Any]) -> Dict[str, Any]:
    point_cloud = np.array(args["point_cloud"], dtype=float)
    epsilon = args.get("epsilon")
    if epsilon is not None:
        betti_res = betti_analyzer.compute_betti_numbers(point_cloud, float(epsilon))
        return {"mode": "single_scale", "epsilon": float(epsilon), "betti_numbers": betti_res}
    else:
        num_steps = args.get("num_steps", 10)
        sweep = betti_analyzer.persistent_homology_sweep(point_cloud, num_steps=num_steps)
        return {"mode": "persistent_sweep", "filtration": sweep}

def handle_hpiv_critic(args: Dict[str, Any]) -> Dict[str, Any]:
    expected_return = float(args["expected_return"])
    current_volatility = float(args["current_volatility"])
    max_historical_drawdown = float(args["max_historical_drawdown"])
    is_safe, safety_ratio = consensus_engine.hamiltonian_path_integral_critic(
        expected_return, current_volatility, max_historical_drawdown
    )
    return {
        "is_safe": is_safe,
        "safety_ratio": safety_ratio
    }

def handle_dro_critic(args: Dict[str, Any]) -> Dict[str, Any]:
    returns_history = np.array(args["returns_history"], dtype=float)
    expected_return = float(args["expected_return"])
    pert_radius = float(args.get("pert_radius", 0.02))
    is_safe, loss = consensus_engine.wasserstein_dro_critic(
        returns_history, expected_return, pert_radius
    )
    return {
        "is_safe": is_safe,
        "perturbed_return_loss": loss
    }

def handle_mine_apriori(args: Dict[str, Any]) -> Dict[str, Any]:
    data_dict = args["data_dict"]
    min_support = float(args.get("min_support", 0.05))
    max_k = int(args.get("max_k", 3))
    df = pd.DataFrame(data_dict)
    itemsets = mine_itemsets(df, min_support=min_support, max_k=max_k)
    formatted = {}
    for k, entries in itemsets.items():
        formatted[str(k)] = [{"itemset": list(items), "support": supp} for items, supp in entries]
    return {"frequent_itemsets": formatted}

def handle_greedy_stack(args: Dict[str, Any]) -> Dict[str, Any]:
    data_dict = args["data_dict"]
    features = args["features"]
    target = args["target"]
    min_support = float(args.get("min_support", 0.05))
    df = pd.DataFrame(data_dict)
    return greedy_stack(df, features, target, min_support=min_support)

def handle_evolve_features(args: Dict[str, Any]) -> Dict[str, Any]:
    data_dict = args["data_dict"]
    base_features = args["base_features"]
    target = args["target"]
    max_depth = int(args.get("max_depth", 2))
    population_size = int(args.get("population_size", 15))
    generations = int(args.get("generations", 3))
    df = pd.DataFrame(data_dict)

    synthesizer = GeneticFeatureSynthesizer(
        base_features=base_features,
        max_depth=max_depth,
        population_size=population_size,
        generations=generations
    )
    best_results = synthesizer.evolve(df, target=target)
    out = []
    for node, score in best_results:
        out.append({
            "formula": node.to_string(),
            "pareto_fitness_score": score
        })
    return {"evolved_features": out}

TOOLS_SPEC = [
    {
        "name": "holographic_consensus",
        "description": "Computes holographic MERA tensor consensus over agent belief vectors",
        "inputSchema": {
            "type": "object",
            "properties": {
                "belief_vectors": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "expected_returns": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["belief_vectors", "expected_returns"]
        }
    },
    {
        "name": "hierarchical_consensus",
        "description": "Recursively computes swarm consensus across N-th level subgroup hierarchies",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subgroup_tree": {"type": "array"}
            },
            "required": ["subgroup_tree"]
        }
    },
    {
        "name": "betti_topology",
        "description": "Computes Vietoris-Rips Betti numbers (beta_0, beta_1, beta_2) and persistent homology",
        "inputSchema": {
            "type": "object",
            "properties": {
                "point_cloud": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "epsilon": {"type": "number"},
                "num_steps": {"type": "integer"}
            },
            "required": ["point_cloud"]
        }
    },
    {
        "name": "hpiv_critic",
        "description": "Validates Lagrangian action trajectory safety using path integrals",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expected_return": {"type": "number"},
                "current_volatility": {"type": "number"},
                "max_historical_drawdown": {"type": "number"}
            },
            "required": ["expected_return", "current_volatility", "max_historical_drawdown"]
        }
    },
    {
        "name": "dro_critic",
        "description": "Evaluates Wasserstein DRO worst-case perturbation loss",
        "inputSchema": {
            "type": "object",
            "properties": {
                "returns_history": {"type": "array", "items": {"type": "number"}},
                "expected_return": {"type": "number"},
                "pert_radius": {"type": "number"}
            },
            "required": ["returns_history", "expected_return"]
        }
    },
    {
        "name": "mine_apriori",
        "description": "Mines frequent boolean itemsets using Apriori pruning search",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_dict": {"type": "object"},
                "min_support": {"type": "number"},
                "max_k": {"type": "integer"}
            },
            "required": ["data_dict"]
        }
    },
    {
        "name": "greedy_stack",
        "description": "Stacks logical AND boolean rules to maximize mean target lift",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_dict": {"type": "object"},
                "features": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "min_support": {"type": "number"}
            },
            "required": ["data_dict", "features", "target"]
        }
    },
    {
        "name": "evolve_features",
        "description": "Synthesizes mathematical AST features using genetic evolution and Pareto scoring",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_dict": {"type": "object"},
                "base_features": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "max_depth": {"type": "integer"},
                "population_size": {"type": "integer"},
                "generations": {"type": "integer"}
            },
            "required": ["data_dict", "base_features", "target"]
        }
    }
]

def main():
    sys.stderr.write("[MERA-KMPA MCP Server] Started.\n")
    sys.stderr.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            msg_id = request.get("id")

            response = {"jsonrpc": "2.0", "id": msg_id}

            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {
                        "name": "mera-kmpa-swarm-mcp-server",
                        "version": "1.0.0"
                    }
                }
            elif method == "tools/list":
                response["result"] = {"tools": TOOLS_SPEC}
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                if tool_name == "holographic_consensus":
                    res = handle_holographic_consensus(tool_args)
                elif tool_name == "hierarchical_consensus":
                    res = handle_hierarchical_consensus(tool_args)
                elif tool_name == "betti_topology":
                    res = handle_betti_topology(tool_args)
                elif tool_name == "hpiv_critic":
                    res = handle_hpiv_critic(tool_args)
                elif tool_name == "dro_critic":
                    res = handle_dro_critic(tool_args)
                elif tool_name == "mine_apriori":
                    res = handle_mine_apriori(tool_args)
                elif tool_name == "greedy_stack":
                    res = handle_greedy_stack(tool_args)
                elif tool_name == "evolve_features":
                    res = handle_evolve_features(tool_args)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                response["result"] = {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                }
            else:
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"[ERROR] {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    main()
