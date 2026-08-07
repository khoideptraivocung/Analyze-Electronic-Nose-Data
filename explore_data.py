import kagglehub
import os

path = kagglehub.dataset_download("husamksalih/array-of-gas-sensors-data")
print("Path to dataset files:", path)
print("Files:", os.listdir(path))
