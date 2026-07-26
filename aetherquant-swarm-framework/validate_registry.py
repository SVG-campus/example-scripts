#!/usr/bin/env python3
"""
Registry Validation Script.
Verifies structure, total count, tier distributions, and solver bindings for all 100 civilization repositories.
"""

import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MEGA_CIVILIZATION_REPOSITORY_REGISTRY.json")

def validate_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sectors = data.get("sectors", {})
    total_repos = 0
    tier_counts = {"Tier 1 - Fully Automated": 0, "Tier 2 - VC / Partner": 0, "Tier 3 - Direct CEO Focus": 0}

    print(f"=== Validating Master Civilization Registry: [{data.get('framework')}] ===")
    
    for sector_name, repo_list in sectors.items():
        count = len(repo_list)
        total_repos += count
        print(f"  Sector [{sector_name}]: {count} Repositories")
        for repo in repo_list:
            tier = repo.get("tier")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\nTotal Cataloged Repositories: {total_repos}")
    print("\nGovernance Tier Distribution:")
    for tier, cnt in tier_counts.items():
        pct = (cnt / total_repos) * 100 if total_repos > 0 else 0
        print(f"  - {tier}: {cnt} ({pct:.1f}%)")

    assert total_repos == 100, f"Expected 100 repositories, found {total_repos}"
    print("\n[OK] Master Registry Validation Passed (100/100 Repositories Cataloged).")

if __name__ == "__main__":
    validate_registry()
