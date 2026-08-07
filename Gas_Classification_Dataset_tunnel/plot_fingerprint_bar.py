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

out_dir = "fingerprints_bar"
os.makedirs(out_dir, exist_ok=True)

# Lấy giá trị trung bình của toàn bộ các mẫu cho mỗi hóa chất
# (Lúc này mỗi hóa chất chỉ còn lại 1 con số duy nhất cho mỗi feature)
df_mean = df_all.groupby('Chemical').mean()

print(df_all['Chemical'].value_counts())
# Vẽ biểu đồ Bar Chart cho từng hóa chất
for chem in classes:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    fig.suptitle(f'Chemical Fingerprint (Bar Chart) - {chem}', fontsize=22, fontweight='bold')
    
    for i, array_prefix in enumerate(sensor_arrays):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Tạo nhãn trục x (Ch1 đến Ch8)
        channels = [f'Ch{j}' for j in range(1, 9)]
        features = [f'mean_{array_prefix}{j}' for j in range(1, 9)]
        
        # Lấy 8 giá trị trung bình tương ứng
        values = []
        for feature in features:
            if feature in df_mean.columns:
                values.append(df_mean.loc[chem, feature])
            else:
                values.append(0)
                
        # Vẽ bar chart với màu sắc đẹp mắt
        colors = sns.color_palette("Set2", 8)
        ax.bar(channels, values, color=colors, edgecolor='black')
        
        ax.set_title(f'Sensor Array {array_prefix}', fontsize=14)
        if col == 0:
            ax.set_ylabel('Mean Amplitude')
            
        # Quay nhãn 45 độ để không bị đè lên nhau
        ax.tick_params(axis='x', rotation=45)
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_chem_name = str(chem).replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(out_dir, f'fingerprint_{safe_chem_name}.png'))
    plt.close()

print(f"Generated 11 fingerprint bar charts in '{out_dir}' directory!")
