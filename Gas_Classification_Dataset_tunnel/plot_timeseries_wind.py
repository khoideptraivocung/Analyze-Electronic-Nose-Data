import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

out_dir = "plots_by_class_300"
os.makedirs(out_dir, exist_ok=True)

# Vẽ theo Class: Mỗi hình là 1 loại khí, gồm 9 subplots (cho 9 cụm cảm biến)
for chem in classes:
    # Trích xuất 300 mẫu đầu tiên để nhìn rõ hơn
    df_class = df_all[df_all['Chemical'] == chem].head(300).reset_index(drop=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharex=True)
    fig.suptitle(f'Chemical: {chem} - Sensor Arrays Response (First 300 samples)', fontsize=22, fontweight='bold')
    
    for i, array_prefix in enumerate(sensor_arrays):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        for j in range(1, 9):
            feature = f'mean_{array_prefix}{j}'
            if feature in df_class.columns:
                ax.plot(df_class.index, df_class[feature], alpha=0.8, label=f'Ch {j}')
                
        ax.set_title(f'Sensor Array {array_prefix}', fontsize=16)
        if col == 0:
            ax.set_ylabel('Mean Value', fontsize=12)
        if row == 2:
            ax.set_xlabel('Sample Index (0-300)', fontsize=12)
            
    # Tạo 1 Legend chung ở dưới cùng của hình
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, fontsize='x-large')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    safe_chem_name = str(chem).replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(out_dir, f'timeseries_{safe_chem_name}.png'))
    plt.close()

print(f"Generated 11 class plots in '{out_dir}' directory!")
