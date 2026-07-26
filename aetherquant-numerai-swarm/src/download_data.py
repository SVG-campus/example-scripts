import os
import sys
from numerapi import NumerAPI

def download_v53_datasets():
    print("=== Downloading Numerai v5.3 Quantum Dataset ===")
    napi = NumerAPI(
        public_id=os.environ.get("NUMERAI_PUBLIC_ID", "2PPYXJYSNU4O5P7BU2A25D2RZXQMGL3V"),
        secret_key=os.environ.get("NUMERAI_SECRET_KEY", "ULUQKJCCYWCU5PG7U5KWRPKQAOF7TH6MCVHEE4YTGVNPLBIDMCBPVL24VRVBIHO6")
    )
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "v5.3")
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_download = [
        "v5.3/features.json",
        "v5.3/live_example_preds.parquet",
        "v5.3/live.parquet",
        "v5.3/validation_example_preds.parquet"
    ]
    
    for f in files_to_download:
        dest_filename = os.path.basename(f)
        dest_path = os.path.join(data_dir, dest_filename)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            print(f"Downloading {f} -> {dest_path}...")
            napi.download_dataset(f, dest_path)
        else:
            print(f"File {dest_filename} already exists ({os.path.getsize(dest_path)} bytes). Skipping download.")
            
    print("=== Download Complete! ===")

if __name__ == "__main__":
    download_v53_datasets()
