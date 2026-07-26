#!/usr/bin/env python3
"""
Command Line Interface for MERA-KMPA Swarm Framework.
Supports consensus resolution, Betti loop persistent homology, causal discovery, and MCP server launching.
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_swarm.quantum_swarm_consensus import QuantumSwarmConsensusEngine
from quantum_swarm.betti_topological_analyzer import BettiTopologicalAnalyzer

def main():
    parser = argparse.ArgumentParser(description="MERA-KMPA Swarm & Causal Discovery Framework CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Serve MCP
    mcp_parser = subparsers.add_parser("serve", help="Launch stdio FastMCP server")

    # Run Consensus
    consensus_parser = subparsers.add_parser("consensus", help="Resolve swarm consensus")
    consensus_parser.add_argument("--beliefs", type=str, required=True, help="JSON array of belief vectors")
    consensus_parser.add_argument("--returns", type=str, required=True, help="JSON array of expected returns")

    # Run Betti Topology
    betti_parser = subparsers.add_parser("betti", help="Analyze point cloud persistent homology Betti loops")
    betti_parser.add_argument("--points", type=str, required=True, help="JSON array of N-dimensional points")
    betti_parser.add_argument("--epsilon", type=float, default=None, help="Specific filtration distance epsilon")

    args = parser.parse_args()

    if args.command == "serve":
        from mera_kmpa_mcp_server import main as mcp_main
        mcp_main()
    elif args.command == "consensus":
        engine = QuantumSwarmConsensusEngine()
        b_vecs = np.array(json.loads(args.beliefs), dtype=float)
        rets = np.array(json.loads(args.returns), dtype=float)
        best_idx, confidence = engine.holographic_consensus(b_vecs, rets)
        print(f"Optimal Strategy Index: {best_idx}")
        print(f"Consensus Confidence: {confidence:.6f}")
    elif args.command == "betti":
        analyzer = BettiTopologicalAnalyzer()
        pts = np.array(json.loads(args.points), dtype=float)
        if args.epsilon is not None:
            res = analyzer.compute_betti_numbers(pts, args.epsilon)
            print(json.dumps(res, indent=2))
        else:
            sweep = analyzer.persistent_homology_sweep(pts)
            print(json.dumps(sweep, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
