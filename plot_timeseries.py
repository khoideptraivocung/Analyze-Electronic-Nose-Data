import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("husamksalih/array-of-gas-sensors-data")
csv_file = os.path.join(path, "All Samples.csv")

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# Chọn các cảm biến chính
sensor_cols = ['MQ_2', 'MQ_4', 'MQ_135', 'MQ_136', 'MQ_8']

# Vẽ biểu đồ lineplot để thấy sự khác biệt của tín hiệu theo thời gian (hoặc theo index) cho từng loại chất
classes = df['Sampe/Class'].unique()

plt.figure(figsize=(20, 15))
for i, cls in enumerate(classes):
    plt.subplot(4, 3, i + 1) # Có khoảng 10 class
    subset = df[df['Sampe/Class'] == cls].reset_index()
    for sensor in sensor_cols:
        plt.plot(subset.index, subset[sensor], label=sensor)
    plt.title(f"Class: {cls}")
    plt.xlabel("Sample Time/Index")
    plt.ylabel("Sensor Value")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('timeseries_by_class.png')
plt.close()

# Vẽ biểu đồ Radar/Spider (hoặc PCA) để xem sự phân biệt các cụm (tùy chọn)
# Ở đây ta sẽ vẽ phân phối của 2 cảm biến mạnh nhất (VD: MQ_4 vs MQ_8)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='MQ_4', y='MQ_8', hue='Sampe/Class', palette='tab10', alpha=0.7)
plt.title("Phân loại các chất dựa trên MQ_4 và MQ_8")
plt.tight_layout()
plt.savefig('scatter_mq4_mq8.png')
plt.close()

print("Timeseries and scatter plots generated!")
