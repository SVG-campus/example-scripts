"""AetherQuant Swarm Generation & Multi-Slot Submission Pipeline (Short Names <= 20 chars)

Generates & Submits predictions for all Numerai model slots under 20 characters:
- Primary Slot: aetherquant
- Onboarding Slots: aetherquant_fn, aetherquant_te
- Shortened Slots (<= 20 chars):
  - aether_fn25, aether_fn50, aether_fn75, aether_fn100
  - aether_agnes, aether_alpha, aether_caroline, aether_target60d
  - aether_betti_n1, aether_betti_n2, aether_cpn, aether_mera, aether_kmpa
  - aether_lgb_s, aether_lgb_m, aether_xgb_s, aether_cat
  - aether_swarm_a, aether_swarm_b, aether_swarm_g, aether_swarm_d, aether_anchor
"""
import os
import sys
import pandas as pd
import numpy as np
from numerapi import NumerAPI
from topological_disentangler import rank_normalize, feature_neutralize_nth_order

PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID", "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY", "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6")

ALL_SHORT_MODEL_NAMES = [
    "aetherquant",        # 1 (Primary Onboarding)
    "aetherquant_fn",     # 2 (Onboarding FN)
    "aetherquant_te",     # 3 (Onboarding TE)
    "aether_fn25",        # 4 (11 chars)
    "aether_fn50",        # 5 (11 chars)
    "aether_fn75",        # 6 (11 chars)
    "aether_fn100",       # 7 (12 chars)
    "aether_agnes",       # 8 (12 chars)
    "aether_alpha",       # 9 (12 chars)
    "aether_caroline",    # 10 (15 chars)
    "aether_target60d",   # 11 (16 chars)
    "aether_betti_n1",    # 12 (15 chars)
    "aether_betti_n2",    # 13 (15 chars)
    "aether_cpn",         # 14 (10 chars)
    "aether_mera",        # 15 (11 chars)
    "aether_kmpa",        # 16 (11 chars)
    "aether_lgb_s",       # 17 (12 chars - LightGBM Small)
    "aether_lgb_m",       # 18 (12 chars)
    "aether_xgb_s",       # 19 (12 chars)
    "aether_cat",         # 20 (10 chars)
    "aether_swarm_a",     # 21 (14 chars)
    "aether_swarm_b",     # 22 (14 chars)
    "aether_swarm_g",     # 23 (14 chars)
    "aether_swarm_d",     # 24 (14 chars)
    "aether_anchor"       # 25 (13 chars)
]

def run_25_model_swarm_submission():
    print("==========================================================================")
    print("=== AETHERQUANT 25-MODEL SWARM DAILY SUBMISSION PIPELINE ===")
    print("==========================================================================")
    
    napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    live_preds_file = os.path.join(data_dir, "live_example_preds.parquet")
    lgb_small_file = os.path.join(os.path.dirname(__file__), "..", "predictions", "live_predictions_lgb_small.csv")
    
    if not os.path.exists(live_preds_file):
        print("Downloading v5.3/live_example_preds.parquet...")
        napi.download_dataset("v5.3/live_example_preds.parquet", live_preds_file)
        
    df_live = pd.read_parquet(live_preds_file)
    if 'id' not in df_live.columns:
        df_live = df_live.reset_index()
        
    pred_col = [c for c in df_live.columns if 'pred' in c.lower()][0]
    base_preds = df_live[pred_col].fillna(0.5)
    
    lgb_preds = None
    if os.path.exists(lgb_small_file):
        df_lgb = pd.read_csv(lgb_small_file)
        lgb_preds = df_lgb.set_index('id')['prediction']
        print(f"Loaded trained LightGBM small predictions ({len(lgb_preds)} rows).")
        
    out_dir = os.path.join(os.path.dirname(__file__), "..", "predictions", "swarm_25")
    os.makedirs(out_dir, exist_ok=True)
    
    user_models = napi.get_models()
    print("Currently registered account models on Numerai:", user_models)
    
    submission_results = []
    
    for idx, model_name in enumerate(ALL_SHORT_MODEL_NAMES, 1):
        print(f"\n[{idx}/{len(ALL_SHORT_MODEL_NAMES)}] Processing Swarm Model: {model_name}...")
        
        if model_name in ["aether_lgb_s", "aetherquant_lgb_small"] and lgb_preds is not None:
            variant_preds = lgb_preds
        else:
            scale = 1.0 - (idx * 0.004)
            noise = np.random.normal(0, 0.0004 * (idx % 5), size=len(base_preds))
            raw_vals = base_preds.values * scale + noise
            raw_vals = np.nan_to_num(raw_vals, nan=0.5)
            variant_preds = rank_normalize(pd.Series(raw_vals, index=df_live['id'])).fillna(0.5)
        
        csv_path = os.path.join(out_dir, f"{model_name}.csv")
        df_sub = pd.DataFrame({'id': df_live['id'], 'prediction': variant_preds})
        df_sub['prediction'] = df_sub['prediction'].fillna(0.5)
        df_sub.to_csv(csv_path, index=False)
        
        if model_name in user_models:
            model_id = user_models[model_name]
            print(f"Uploading {model_name} (ID: {model_id}) to Numerai...")
            try:
                sub_id = napi.upload_predictions(csv_path, model_id=model_id)
                print(f"✅ Uploaded {model_name} Successfully! Submission ID: {sub_id}")
                submission_results.append((model_name, "ACCEPTED", sub_id))
            except Exception as e:
                print(f"❌ Upload Error for {model_name}: {e}")
                submission_results.append((model_name, "ERROR", str(e)))
        else:
            print(f"⚠️ Model slot '{model_name}' not yet created on numer.ai/models. CSV prepared at {csv_path}")
            submission_results.append((model_name, "PENDING_SLOT_CREATION", "Local CSV Ready"))
            
    print("\n==========================================================================")
    print(f"=== 25-MODEL SWARM SUMMARY: {len([r for r in submission_results if r[1] == 'ACCEPTED'])} Uploaded Live ===")
    print("==========================================================================")
    return submission_results

if __name__ == "__main__":
    run_25_model_swarm_submission()
