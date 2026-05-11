# ai/download_dataset.py
import os
import requests
import pandas as pd

def download_file(url, filename):
    print(f"[AI] Downloading {url} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"[AI] Saved to {filename}")
        return True
    else:
        print(f"[AI] ERROR: Could not download file (Status: {response.status_code})")
        return False

def main():
    os.makedirs("ai/data", exist_ok=True)
    
    # Primary Source: Ergo-Net Public Dataset (CSV with 3D landmarks + RULA scores)
    # This dataset is specifically designed for multi-output ergonomic regression.
    url = "https://raw.githubusercontent.com/vigneshwarr/Ergo-Net-v2/master/data/processed_ergo.csv"
    target = "ai/data/dataset.csv"
    
    if download_file(url, target):
        df = pd.read_csv(target)
        print(f"[AI] Dataset loaded successfully: {len(df)} samples found.")
    else:
        print("[AI] Falling back to high-fidelity synthetic data generation.")
        from ai.synthetic_gen import generate_ergonomic_dataset
        generate_ergonomic_dataset(num_samples=15000)

if __name__ == "__main__":
    main()
