import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("husamksalih/array-of-gas-sensors-data")

# Bỏ qua file All Samples.csv vì nó không chứa đủ dữ liệu
files = [f for f in os.listdir(path) if f.endswith('.csv') and f != 'All Samples.csv']

plt.figure(figsize=(24, 20))

# ĐÃ CẬP NHẬT: Thêm đầy đủ các biến liên quan đến Df-NH3 và MQ-136 (volt, ratio, thô)
sensor_cols = [
    # 1. Các yếu tố môi trường (Raw)
    
    
    # 2. Cụm NH3
    'volt_NH3',          # Tính toán & Điện áp thô NH3
    
    # 3. Cụm MQ-136
    'Vmq136', 'volt_MQ_136', # Tính toán & các bản thể điện áp thô của MQ-136
    
    # 4. Thêm toàn bộ các cột điện áp (Volt) của các cảm biến còn lại:
    'Vmq2', 'volt_MQ_2','Vmq4', 'volt_MQ_4','Vmq135', 'volt_MQ_135','Vmq8', 'volt_MQ_8',
]

for i, file in enumerate(files):
    csv_file = os.path.join(path, file)
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    cls = file.replace('.csv', '')
    
    plt.subplot(4, 3, i + 1)
    
    # Kiểm tra xem các cột khai báo có thực sự tồn tại trong file CSV này không
    existing_cols = [c for c in sensor_cols if c in df.columns]
    
    for sensor in existing_cols:
        df[sensor] = pd.to_numeric(df[sensor], errors='coerce')
        plt.plot(df.index, df[sensor], label=sensor, alpha=0.85)
        
    plt.title(f"Class: {cls}", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Time/Index", fontsize=10)
    plt.ylabel("Sensor Value", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5) # Thêm lưới cho dễ nhìn xu hướng
    
    # ĐỂ LEGEND Ở ĐÂY: Hiển thị chú thích cho TẤT CẢ các ô để dễ phân biệt màu từng sensor
    plt.legend(loc='upper right', fontsize=8, framealpha=0.6)

plt.tight_layout()
plt.savefig('timeseries_all_sample_updated.png', dpi=300) # Tăng dpi lên 300 cho ảnh nét căng
plt.close()

print("Generated updated timeseries plots with NH3 and MQ136 parameters successfully!")