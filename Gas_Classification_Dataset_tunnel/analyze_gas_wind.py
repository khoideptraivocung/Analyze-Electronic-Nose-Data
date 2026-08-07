import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
import kagglehub

path = kagglehub.dataset_download("brmil07/gas-classification-dataset")

files = [f for f in os.listdir(path) if f.endswith('.csv')]
dfs = []
for f in files:
    filepath = os.path.join(path, f)
    df = pd.read_csv(filepath)
    # Loại bỏ cột Unnamed: 0 nếu có
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print(f"Total records: {len(df_all)}")
print(f"Chemical classes: {df_all['Chemical'].unique()}")

# 1. Vẽ biểu đồ phân bố số lượng mẫu của mỗi loại hóa chất
plt.figure(figsize=(14, 8))
sns.countplot(data=df_all, y='Chemical', order=df_all['Chemical'].value_counts().index, palette='viridis')
plt.title("Distribution of samples for each chemical class (Classes)")
plt.xlabel("Samples")
plt.ylabel("Chemical")
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# 2. Rút gọn chiều dữ liệu (PCA) để xem khả năng phân loại
# Dữ liệu chứa các cột features và cột 'Chemical'
features = df_all.drop(columns=['Chemical']).fillna(0)
labels = df_all['Chemical']

# Chuẩn hóa dữ liệu trước khi PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

df_pca = pd.DataFrame(data=features_pca, columns=['PC1', 'PC2'])
df_pca['Chemical'] = labels

plt.figure(figsize=(14, 10))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Chemical', palette='tab10', alpha=0.7)
plt.title("PCA 2D Clustering - 288 Features")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('pca_clusters.png')
plt.close()

# 3. Biểu đồ Heatmap trung bình của 8 tính năng đầu tiên (mean_A1 -> mean_A8) theo Chemical
cols_to_plot = [f'mean_A{i}' for i in range(1, 9)]
if all(c in df_all.columns for c in cols_to_plot):
    grouped = df_all.groupby('Chemical')[cols_to_plot].mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(grouped, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title("Average of Sensor Array A (mean_A1 to mean_A8) by Chemical")
    plt.ylabel("Chemical")
    plt.xlabel("Sensor Array A")
    plt.tight_layout()
    plt.savefig('sensor_heatmap.png')
    plt.close()

print("Finished generating plots!")