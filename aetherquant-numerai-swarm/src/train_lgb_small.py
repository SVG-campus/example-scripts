import os
import sys
import json
import joblib
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI
from topological_disentangler import rank_normalize

def train_small_feature_model():
    print("=== Training LightGBM Model on Numerai v5.3 Small Feature Set ===")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download train.parquet if missing
    train_parquet = os.path.join(data_dir, "train.parquet")
    napi = NumerAPI()
    if not os.path.exists(train_parquet):
        print("Downloading v5.3/train.parquet...")
        napi.download_dataset("v5.3/train.parquet", train_parquet)
        
    # 2. Load features.json
    features_json = os.path.join(data_dir, "features.json")
    with open(features_json) as f:
        feature_data = json.load(f)
        
    small_features = feature_data["feature_sets"]["small"]
    target_col = "target"
    print(f"Using {len(small_features)} features from 'small' set to predict '{target_col}'.")
    
    # 3. Read train data
    print("Loading train dataset...")
    df_train = pd.read_parquet(train_parquet, columns=small_features + [target_col])
    
    X_train = df_train[small_features]
    y_train = df_train[target_col]
    
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lgb_small_v53.pkl")
    joblib.dump(model, model_path)
    print(f"Saved trained LightGBM model to {model_path}!")
    
    # 4. Predict on live data
    live_parquet = os.path.join(data_dir, "live.parquet")
    if not os.path.exists(live_parquet):
        napi.download_dataset("v5.3/live.parquet", live_parquet)
        
    print("Generating live predictions...")
    df_live = pd.read_parquet(live_parquet, columns=small_features)
    raw_preds = model.predict(df_live[small_features])
    
    if 'id' in df_live.columns:
        ids = df_live['id']
    else:
        ids = df_live.index
        
    df_submit = pd.DataFrame({
        'id': ids,
        'prediction': rank_normalize(pd.Series(raw_preds, index=ids))
    })
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "live_predictions_lgb_small.csv")
    df_submit.to_csv(csv_out, index=False)
    print(f"Saved live prediction CSV ({len(df_submit)} rows) to: {csv_out}")
    print("=== Training & Prediction Complete! ===")

if __name__ == "__main__":
    train_small_feature_model()
