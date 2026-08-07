import os
os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("brmil07/gas-classification-dataset")
print("Path:", path)
files = os.listdir(path)
print("Files:", files)

for f in files:
    if f.endswith('.csv'):
        filepath = os.path.join(path, f)
        df = pd.read_csv(filepath, nrows=5)
        print(f"\n--- {f} ---")
        print("Shape:", pd.read_csv(filepath).shape)
        print("Columns:", df.columns.tolist()[:50], "...", df.columns.tolist()[-5:])
        print(df.head())
