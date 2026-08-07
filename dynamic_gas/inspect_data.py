import os
os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
import kagglehub

path = kagglehub.dataset_download("uciml/gas-sensor-array-under-dynamic-gas-mixtures")
print("Path:", path)

files = os.listdir(path)
print("Files:", files)

for f in files:
    if f.endswith('.txt'):
        print(f"\n--- First 5 lines of {f} ---")
        filepath = os.path.join(path, f)
        with open(filepath, 'r') as file:
            for _ in range(5):
                print(file.readline().strip())
