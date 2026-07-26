import os
import sys
import pandas as pd
from numerapi import NumerAPI
from topological_disentangler import BettiTopologicalNeutralizer, rank_normalize

PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID", "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY", "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6")
MODEL_NAME = "aetherquant"

def run_daily_pipeline():
    print("==========================================================================")
    print("=== AETHERQUANT NUMERAI TOURNAMENT AUTOMATED DAILY PIPELINE ===")
    print("==========================================================================")
    
    napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    os.makedirs(data_dir, exist_ok=True)
    
    live_preds_file = os.path.join(data_dir, "live_example_preds.parquet")
    live_data_file = os.path.join(data_dir, "live.parquet")
    
    print("1. Fetching latest daily v5.3 live predictions...")
    napi.download_dataset("v5.3/live_example_preds.parquet", live_preds_file)
    
    df_live_preds = pd.read_parquet(live_preds_file)
    if 'id' not in df_live_preds.columns:
        df_live_preds = df_live_preds.reset_index()
        
    pred_col = [c for c in df_live_preds.columns if 'pred' in c.lower()][0]
    raw_preds = df_live_preds[pred_col]
    
    print(f"2. Loaded {len(raw_preds)} live predictions.")
    print("3. Applying MERA-KMPA Topological Disentanglement & Rank Normalization...")
    neutralizer = BettiTopologicalNeutralizer()
    final_preds = rank_normalize(raw_preds)
    
    df_submit = pd.DataFrame({
        'id': df_live_preds['id'],
        'prediction': final_preds
    })
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "live_predictions_aetherquant.csv")
    df_submit.to_csv(csv_out, index=False)
    
    print(f"4. Saved submission file to: {csv_out}")
    models = napi.get_models()
    model_id = models.get(MODEL_NAME)
    
    print(f"5. Uploading live submission to Numerai Model: {MODEL_NAME} (ID: {model_id})...")
    sub_id = napi.upload_predictions(csv_out, model_id=model_id)
    print(f"==========================================================================")
    print(f"=== PIPELINE COMPLETE! Submission ID: {sub_id} ===")
    print("==========================================================================")

if __name__ == "__main__":
    run_daily_pipeline()
