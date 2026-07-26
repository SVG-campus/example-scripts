import os
import sys
import pandas as pd
from numerapi import NumerAPI

PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID", "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY", "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6")
MODEL_NAME = "aetherquant"

def submit_live_predictions(csv_filepath: str, model_name: str = MODEL_NAME):
    print(f"=== Submitting Live Predictions for Model: {model_name} ===")
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Prediction file not found: {csv_filepath}")
        
    napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    
    # Get model map
    models = napi.get_models()
    if model_name not in models:
        print(f"Warning: Model {model_name} not found in user model list: {models}")
        model_id = None
    else:
        model_id = models[model_name]
        print(f"Model {model_name} ID: {model_id}")
        
    print(f"Uploading submission file: {csv_filepath}...")
    submission_id = napi.upload_predictions(csv_filepath, model_id=model_id)
    print(f"Submission Uploaded Successfully! Submission ID: {submission_id}")
    return submission_id

def create_and_submit_baseline_live():
    """Build live predictions from v5.3 live_example_preds.parquet and submit."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    live_preds_parquet = os.path.join(data_dir, "live_example_preds.parquet")
    
    if not os.path.exists(live_preds_parquet):
        raise FileNotFoundError(f"Live example predictions not found: {live_preds_parquet}")
        
    print(f"Reading live predictions from {live_preds_parquet}...")
    df_live = pd.read_parquet(live_preds_parquet)
    
    # Save as CSV for Numerai upload
    out_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "live_predictions_aetherquant.csv")
    
    # Ensure columns 'id' and 'prediction' exist
    if 'id' not in df_live.columns:
        df_live = df_live.reset_index()
        
    pred_col = [c for c in df_live.columns if 'pred' in c.lower()][0]
    df_submit = pd.DataFrame({
        'id': df_live['id'],
        'prediction': df_live[pred_col]
    })
    
    df_submit.to_csv(csv_out, index=False)
    print(f"Saved submission CSV ({len(df_submit)} rows) to: {csv_out}")
    
    sub_id = submit_live_predictions(csv_out, MODEL_NAME)
    return sub_id

if __name__ == "__main__":
    create_and_submit_baseline_live()
