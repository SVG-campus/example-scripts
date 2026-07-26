import os
from numerapi import NumerAPI

PUBLIC_ID = "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V"
SECRET_KEY = "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6"

napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# addModel mutation with tournament: 8 (Classic Numerai Tournament)
q = """
mutation AddModel($name: String!, $tournament: Int!) {
    addModel(name: $name, tournament: $tournament) {
        id
        name
    }
}
"""

models_to_create = [
    "aetherquant_fn",
    "aetherquant_te",
    "aetherquant_lgb",
    "aetherquant_swarm"
]

for model_name in models_to_create:
    try:
        res = napi.raw_query(q, variables={"name": model_name, "tournament": 8})
        print(f"Result for '{model_name}':", res)
    except Exception as e:
        print(f"Error creating '{model_name}': {e}")

print("\nFinal models list:", napi.get_models())
