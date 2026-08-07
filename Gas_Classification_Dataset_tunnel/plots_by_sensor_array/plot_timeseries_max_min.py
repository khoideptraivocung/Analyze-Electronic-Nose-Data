import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
import kagglehub

path = kagglehub.dataset_download("brmil07/gas-classification-dataset")

files = [f for f in os.listdir(path) if f.endswith('.csv')]
dfs = []
for f in files:
    filepath = os.path.join(path, f)
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

sensor_arrays = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
classes = df_all['Chemical'].unique()

out_dir = "plots_by_sensor_array"
os.makedirs(out_dir, exist_ok=True)

# Tính trung bình toàn bộ (kể cả max, min, std thô của data)
df_mean = df_all.groupby('Chemical').mean()

stats = ['mean', 'max', 'min', 'std']
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#7f7f7f'] # Xanh lam, Đỏ, Vàng Cam, Xám

for chem in classes:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=True)
    fig.suptitle(f'Comprehensive Fingerprint (Mean, Max, Min, Std) - {chem}', fontsize=22, fontweight='bold')
    
    for i, array_prefix in enumerate(sensor_arrays):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        channels = np.arange(1, 9)
        width = 0.2 # Độ rộng của mỗi cột con
        
        for idx, stat in enumerate(stats):
            values = []
            for j in range(1, 9):
                feature = f'{stat}_{array_prefix}{j}'
                if feature in df_mean.columns:
                    values.append(df_mean.loc[chem, feature])
                else:
                    values.append(0)
            
            # Tính toán vị trí lệch (offset) để 4 cột đứng cạnh nhau
            x_pos = channels + (idx - 1.5) * width
            ax.bar(x_pos, values, width=width, color=colors[idx], edgecolor='black')
            
        ax.set_title(f'Sensor Array {array_prefix}', fontsize=14)
        ax.set_xticks(channels)
        ax.set_xticklabels([f'Ch{j}' for j in range(1, 9)])
        if col == 0:
            ax.set_ylabel('Average Amplitude')
            
    # Tạo chú thích chung (Legend)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[idx], edgecolor='black', label=stats[idx].upper()) for idx in range(len(stats))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize='x-large')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    safe_chem_name = str(chem).replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(out_dir, f'grouped_fingerprint_{safe_chem_name}.png'))
    plt.close()

print(f"Generated 11 grouped fingerprint charts in '{out_dir}' directory!")
