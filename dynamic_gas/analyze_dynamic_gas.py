import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.environ["KAGGLEHUB_CACHE"] = "D:\\kagglehub_cache"
import kagglehub

path = kagglehub.dataset_download("uciml/gas-sensor-array-under-dynamic-gas-mixtures")

txt_files = ['ethylene_CO.txt', 'ethylene_methane.txt']

cols = ['Time', 'Gas1_Conc', 'Ethylene_Conc']
sensor_cols = [
    'TGS2602_1', 'TGS2602_2', 'TGS2600_1', 'TGS2600_2',
    'TGS2610_1', 'TGS2610_2', 'TGS2620_1', 'TGS2620_2',
    'TGS2602_3', 'TGS2602_4', 'TGS2600_3', 'TGS2600_4',
    'TGS2610_3', 'TGS2610_4', 'TGS2620_3', 'TGS2620_4'
]
all_cols = cols + sensor_cols

import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

for file in txt_files:
    filepath = os.path.join(path, file)
    
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, header=None, names=all_cols)
    
    # Downsample lấy 1/100 (tương đương 1 điểm mỗi giây nếu gốc là 0.01s)
    df = df.iloc[::100, :].reset_index(drop=True)
    
    gas1_name = 'CO' if 'CO' in file else 'Methane'
    
    # Chia dataframe thành 3 phần: 0-3000, 3000-7000, 7000-hết
    df_part1 = df.iloc[:3000]
    df_part2 = df.iloc[3000:7000]
    df_part3 = df.iloc[7000:12000]
    df_part4 = df.iloc[12000:17000]
        
    fig, axes = plt.subplots(2, 4, figsize=(32, 12), sharey='row')
    
    # --- Cột 1: 0 -> 3000 ---
    axes[0, 0].plot(df_part1['Time'], df_part1['Gas1_Conc'], label=f'{gas1_name} conc', color='red')
    axes[0, 0].plot(df_part1['Time'], df_part1['Ethylene_Conc'], label='Ethylene conc', color='blue')
    axes[0, 0].set_title(f'Gas Conc (0 - 3000s)')
    axes[0, 0].set_ylabel('Concentration (ppm)')
    axes[0, 0].legend()
    
    for col in sensor_cols:
        axes[1, 0].plot(df_part1['Time'], df_part1[col], label=col, alpha=0.7)
    axes[1, 0].set_title('Sensors (0 - 3000s)')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('S_i')
    axes[1, 0].legend(loc='upper right', ncol=4, fontsize='small')
    
    # --- Cột 2: 3000 -> 7000 ---
    axes[0, 1].plot(df_part2['Time'], df_part2['Gas1_Conc'], label=f'{gas1_name} conc', color='red')
    axes[0, 1].plot(df_part2['Time'], df_part2['Ethylene_Conc'], label='Ethylene conc', color='blue')
    axes[0, 1].set_title(f'Gas Conc (3000 - 7000s)')
    axes[0, 1].legend()
    
    for col in sensor_cols:
        axes[1, 1].plot(df_part2['Time'], df_part2[col], label=col, alpha=0.7)
    axes[1, 1].set_title('Sensors (3000 - 7000s)')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].legend(loc='upper right', ncol=4, fontsize='small')
    
    # --- Cột 3: 7000 -> 12000 ---
    axes[0, 2].plot(df_part3['Time'], df_part3['Gas1_Conc'], label=f'{gas1_name} conc', color='red')
    axes[0, 2].plot(df_part3['Time'], df_part3['Ethylene_Conc'], label='Ethylene conc', color='blue')
    axes[0, 2].set_title(f'Gas Conc (7000 - 12000s)')
    axes[0, 2].legend()
    
    for col in sensor_cols: 
        axes[1, 2].plot(df_part3['Time'], df_part3[col], label=col, alpha=0.7)
    axes[1, 2].set_title('Sensors (7000 - 12000s)')
    axes[1, 2].set_xlabel('Time (seconds)')
    axes[1, 2].legend(loc='upper right', ncol=4, fontsize='small')

    # --- Cột 4: 12000 -> 17000 ---
    axes[0, 3].plot(df_part4['Time'], df_part4['Gas1_Conc'], label=f'{gas1_name} conc', color='red')
    axes[0, 3].plot(df_part4['Time'], df_part4['Ethylene_Conc'], label='Ethylene conc', color='blue')
    axes[0, 3].set_title(f'Gas Conc (12000 - 17000s)')
    axes[0, 3].legend()
    
    for col in sensor_cols: 
        axes[1, 3].plot(df_part4['Time'], df_part4[col], label=col, alpha=0.7)
    axes[1, 3].set_title('Sensors (12000 - 17000s)')
    axes[1, 3].set_xlabel('Time (seconds)')
    axes[1, 3].legend(loc='upper right', ncol=4, fontsize='small')
    
    plt.tight_layout()
    out_img = f"{file.replace('.txt', '')}_4_cols_plot.png"
    plt.savefig(out_img)
    plt.close()

print("Generated split plots (0-3000 and 3000-end) for dynamic gas mixtures!")
