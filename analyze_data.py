import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("husamksalih/array-of-gas-sensors-data")
csv_file = os.path.join(path, "All Samples.csv")

df = pd.read_csv(csv_file)

with open("analysis.txt", "w", encoding="utf-8") as f:
    f.write("Dataset Information:\n")
    f.write("This is a dataset containing sensor readings from an array of gas sensors for various substances (like Coffee, Garlic, Gasoline, etc.).\n\n")
    
    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    f.write("\n")
    
    f.write("Data Sample (First 5 rows):\n")
    f.write(df.head().to_string())
    f.write("\n\n")
    
    f.write("Data Statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\n")

print("Columns:", df.columns.tolist())
