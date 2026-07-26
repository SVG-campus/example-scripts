import os
import sys
import numpy as np
import pandas as pd

def numerai_corr(preds: pd.Series, targets: pd.Series) -> float:
    """Compute Numerai Spearman correlation rank score."""
    ranked_preds = preds.rank(pct=True, method="first")
    return float(np.corrcoef(ranked_preds, targets)[0, 1])

def evaluate_validation_performance(df_val_preds: pd.DataFrame, target_col: str = "target"):
    """Evaluate era-by-era correlation performance on validation dataset."""
    print("=== Evaluating Era-by-Era Historical Validation Performance ===")
    
    if "era" not in df_val_preds.columns:
        raise KeyError("Validation DataFrame must contain 'era' column.")
        
    eras = df_val_preds["era"].unique()
    era_corrs = []
    
    for era in eras:
        df_era = df_val_preds[df_val_preds["era"] == era]
        if len(df_era) > 0:
            corr = numerai_corr(df_era["prediction"], df_era[target_col])
            era_corrs.append(corr)
            
    era_corrs = np.array(era_corrs)
    mean_corr = np.mean(era_corrs)
    std_corr = np.std(era_corrs)
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0
    max_drawdown = float(np.min(np.cumsum(era_corrs)))
    
    print("--------------------------------------------------------------------------")
    print(f"Mean Era CORR:     {mean_corr:+.5f}")
    print(f"Std Era CORR:      {std_corr:.5f}")
    print(f"Validation Sharpe: {sharpe:+.4f}")
    print(f"Max Drawdown:      {max_drawdown:+.5f}")
    print("--------------------------------------------------------------------------")
    
    return {
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    val_preds_file = os.path.join(data_dir, "validation_example_preds.parquet")
    
    if os.path.exists(val_preds_file):
        df_val = pd.read_parquet(val_preds_file)
        if "prediction" not in df_val.columns:
            pred_col = [c for c in df_val.columns if "pred" in c.lower()][0]
            df_val["prediction"] = df_val[pred_col]
        if "target" not in df_val.columns:
            df_val["target"] = 0.5
        evaluate_validation_performance(df_val)
    else:
        print(f"Validation predictions file missing: {val_preds_file}")
