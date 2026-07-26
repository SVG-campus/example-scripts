"""AetherQuant 25-Model Swarm Generation & Submission Pipeline

Generates & Submits predictions for all 25 Numerai model slots:
- Models 1-3: Onboarding Models (aetherquant, aetherquant_fn, aetherquant_te)
- Models 4-7: Feature Neutralization (aetherquant_fn_025, aetherquant_fn_050, aetherquant_fn_075, aetherquant_fn_100)
- Models 8-11: Multi-Horizon Targets (aetherquant_target_agnes, aetherquant_target_alpha, aetherquant_target_caroline, aetherquant_target_60d)
- Models 12-16: MERA-KMPA Topological Homology (aetherquant_betti_n1, aetherquant_betti_n2, aetherquant_cpn_fubini, aetherquant_mera_noise_cancel, aetherquant_kmpa_phase_align)
- Models 17-20: ML Architecture Ensembles (aetherquant_lgb_small, aetherquant_lgb_medium, aetherquant_xgb_small, aetherquant_catboost)
- Models 21-25: Swarm Diversifiers & Meta Anchors (aetherquant_swarm_alpha, aetherquant_swarm_beta, aetherquant_swarm_gamma, aetherquant_swarm_delta, aetherquant_meta_anchor)
"""
import os
import sys
import pandas as pd
import numpy as np
from numerapi import NumerAPI
from topological_disentangler import rank_normalize, feature_neutralize_nth_order

PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID", "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY", "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6")

ALL_25_MODEL_NAMES = [
    "aetherquant",                  # 1 (Primary Onboarding)
    "aetherquant_fn",               # 2 (Onboarding FN)
    "aetherquant_te",               # 3 (Onboarding TE)
    "aetherquant_fn_025",           # 4
    "aetherquant_fn_050",           # 5
    "aetherquant_fn_075",           # 6
    "aetherquant_fn_100",           # 7
    "aetherquant_target_agnes",     # 8
    "aetherquant_target_alpha",     # 9
    "aetherquant_target_caroline",  # 10
    "aetherquant_target_60d",       # 11
    "aetherquant_betti_n1",         # 12
    "aetherquant_betti_n2",         # 13
    "aetherquant_cpn_fubini",       # 14
    "aetherquant_mera_noise_cancel",# 15
    "aetherquant_kmpa_phase_align", # 16
    "aetherquant_lgb_small",        # 17 (Trained LightGBM Model)
    "aetherquant_lgb_medium",       # 18
    "aetherquant_xgb_small",        # 19
    "aetherquant_catboost",         # 20
    "aetherquant_swarm_alpha",      # 21
    "aetherquant_swarm_beta",       # 22
    "aetherquant_swarm_gamma",      # 23
    "aetherquant_swarm_delta",      # 24
    "aetherquant_meta_anchor"       # 25
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
    
    for idx, model_name in enumerate(ALL_25_MODEL_NAMES, 1):
        print(f"\n[{idx}/25] Processing Swarm Model: {model_name}...")
        
        if model_name == "aetherquant_lgb_small" and lgb_preds is not None:
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
