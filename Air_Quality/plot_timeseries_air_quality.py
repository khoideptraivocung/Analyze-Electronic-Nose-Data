import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "air+quality", "AirQualityUCI.csv")
OUTPUT_DIR = os.path.dirname(__file__)

def load_and_preprocess_data():
    print(f"Loading data from: {DATA_PATH}")
    # Read CSV
    df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
    
    # Drop completely empty columns/rows
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="all", axis=0)
    
    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    
    # Filter valid date/time rows
    df = df[df['Date'].notna() & df['Time'].notna()].copy()
    
    # Combine Date and Time into Datetime
    # Date format: DD/MM/YYYY, Time format: HH.MM.SS
    df['DatetimeStr'] = df['Date'] + ' ' + df['Time']
    df['Datetime'] = pd.to_datetime(df['DatetimeStr'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    
    # Drop rows where Datetime parsing failed
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    df.set_index('Datetime', inplace=True)
    
    # Replace -200 sentinel missing values with NaN
    numeric_cols = [c for c in df.columns if c not in ['Date', 'Time', 'DatetimeStr']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(-200, np.nan)
        
    print(f"Successfully processed {len(df)} records from {df.index.min()} to {df.index.max()}")
    return df

def plot_mox_sensors(df):
    plt.figure(figsize=(14, 8))
    sensor_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
    sensor_labels = [
        'PT08.S1 (Tin Oxide - CO targeted)',
        'PT08.S2 (Titania - NMHC targeted)',
        'PT08.S3 (Tungsten Oxide - NOx targeted)',
        'PT08.S4 (Tungsten Oxide - NO2 targeted)',
        'PT08.S5 (Indium Oxide - O3 targeted)'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for col, label, color in zip(sensor_cols, sensor_labels, colors):
        # Apply 24h rolling mean for smoother trend visualization alongside raw signal
        smoothed = df[col].rolling(window=24, min_periods=6).mean()
        plt.plot(df.index, smoothed, label=label, color=color, alpha=0.85, linewidth=1.5)
        
    plt.title("Time Series (1-Year 24h Rolling Avg): 5 Metal Oxide Chemical Sensors", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Sensor Response / Resistance Signal", fontsize=11)
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "air_quality_timeseries_sensors.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_ground_truth(df):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    gt_configs = [
        ('CO(GT)', 'CO Concentration (mg/m³)', '#1f77b4'),
        ('C6H6(GT)', 'Benzene C6H6 Concentration (µg/m³)', '#d62728'),
        ('NOx(GT)', 'NOx Concentration (ppb)', '#2ca02c'),
        ('NO2(GT)', 'NO2 Concentration (µg/m³)', '#ff7f0e')
    ]
    
    for ax, (col, ylabel, color) in zip(axes, gt_configs):
        smoothed = df[col].rolling(window=24, min_periods=6).mean()
        ax.plot(df.index, df[col], alpha=0.25, color=color, linewidth=0.75, label='Hourly Raw')
        ax.plot(df.index, smoothed, alpha=0.9, color=color, linewidth=1.8, label='24h Rolling Avg')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[0].set_title("Ground Truth Air Pollutant Concentrations (Reference Analyzer)", fontsize=14, fontweight='bold', pad=12)
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "air_quality_timeseries_ground_truth.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_environmental(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    env_configs = [
        ('T', 'Temperature (°C)', '#d62728'),
        ('RH', 'Relative Humidity (%)', '#1f77b4'),
        ('AH', 'Absolute Humidity', '#2ca02c')
    ]
    
    for ax, (col, ylabel, color) in zip(axes, env_configs):
        smoothed = df[col].rolling(window=24, min_periods=6).mean()
        ax.plot(df.index, smoothed, color=color, linewidth=1.8, label=f'24h Avg {col}')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[0].set_title("Environmental Parameters Time Series (Temperature & Humidity)", fontsize=14, fontweight='bold', pad=12)
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "air_quality_timeseries_env.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_diurnal_profile(df):
    df_hourly = df.copy()
    df_hourly['Hour'] = df_hourly.index.hour
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Ground Truth Diurnal Profile
    gt_cols = ['CO(GT)', 'C6H6(GT)', 'NO2(GT)']
    gt_means = df_hourly.groupby('Hour')[gt_cols].mean()
    
    # Normalize for comparison
    gt_norm = (gt_means - gt_means.min()) / (gt_means.max() - gt_means.min())
    
    ax1.plot(gt_norm.index, gt_norm['CO(GT)'], marker='o', label='CO(GT)', linewidth=2)
    ax1.plot(gt_norm.index, gt_norm['C6H6(GT)'], marker='s', label='C6H6(GT)', linewidth=2)
    ax1.plot(gt_norm.index, gt_norm['NO2(GT)'], marker='^', label='NO2(GT)', linewidth=2)
    ax1.set_title("Normalized 24-Hour Diurnal Cycle: Ground Truth Pollutants", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Hour of Day", fontsize=11)
    ax1.set_ylabel("Normalized Concentration (0-1)", fontsize=11)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # Highlight traffic peak hours
    ax1.axvspan(7, 10, color='yellow', alpha=0.2, label='Morning Rush Hour')
    ax1.axvspan(18, 21, color='orange', alpha=0.2, label='Evening Rush Hour')
    
    # 2. Sensor Diurnal Profile
    sensor_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)']
    sensor_means = df_hourly.groupby('Hour')[sensor_cols].mean()
    sensor_norm = (sensor_means - sensor_means.min()) / (sensor_means.max() - sensor_means.min())
    
    ax2.plot(sensor_norm.index, sensor_norm['PT08.S1(CO)'], marker='o', label='PT08.S1 (CO)', linewidth=2)
    ax2.plot(sensor_norm.index, sensor_norm['PT08.S2(NMHC)'], marker='s', label='PT08.S2 (NMHC)', linewidth=2)
    ax2.plot(sensor_norm.index, sensor_norm['PT08.S4(NO2)'], marker='^', label='PT08.S4 (NO2)', linewidth=2)
    ax2.set_title("Normalized 24-Hour Diurnal Cycle: Chemical Sensors", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Hour of Day", fontsize=11)
    ax2.set_ylabel("Normalized Sensor Response (0-1)", fontsize=11)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")
    
    ax2.axvspan(7, 10, color='yellow', alpha=0.2)
    ax2.axvspan(18, 21, color='orange', alpha=0.2)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "air_quality_diurnal_profile.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_raw_sensor_vs_groundtruth(df, days=14):
    # Select a representative multi-week sample period (e.g., April 2004) for clear hourly visualization
    start_date = '2004-04-01'
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
    sample_df = df.loc[start_date:end_date].copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    pairs = [
        ('PT08.S1(CO)', 'CO(GT)', 'CO Sensor (PT08.S1) vs Ground Truth CO', '#1f77b4', '#d62728', 'Sensor Tins (CO)', 'CO (mg/m³)'),
        ('PT08.S2(NMHC)', 'C6H6(GT)', 'NMHC Sensor (PT08.S2) vs Ground Truth Benzene (C6H6)', '#ff7f0e', '#9467bd', 'Sensor Titania (NMHC)', 'C6H6 (µg/m³)'),
        ('PT08.S3(NOx)', 'NOx(GT)', 'NOx Sensor (PT08.S3) vs Ground Truth NOx', '#2ca02c', '#8c564b', 'Sensor Tungsten (NOx)', 'NOx (ppb)'),
        ('PT08.S4(NO2)', 'NO2(GT)', 'NO2 Sensor (PT08.S4) vs Ground Truth NO2', '#d62728', '#e377c2', 'Sensor Tungsten (NO2)', 'NO2 (µg/m³)')
    ]
    
    for ax1, (sensor_col, gt_col, title, color_s, color_gt, s_label, gt_label) in zip(axes, pairs):
        ax2 = ax1.twinx()
        
        # Plot raw hourly data directly without smoothing
        line1 = ax1.plot(sample_df.index, sample_df[sensor_col], color=color_s, marker='o', markersize=3, 
                         linewidth=1.2, alpha=0.85, label=f'Raw Sensor: {sensor_col}')
        line2 = ax2.plot(sample_df.index, sample_df[gt_col], color=color_gt, marker='x', markersize=4, 
                         linewidth=1.2, linestyle='--', alpha=0.85, label=f'Raw GT: {gt_col}')
        
        ax1.set_ylabel(s_label, color=color_s, fontweight='bold', fontsize=10)
        ax2.set_ylabel(gt_label, color=color_gt, fontweight='bold', fontsize=10)
        ax1.tick_params(axis='y', labelcolor=color_s)
        ax2.tick_params(axis='y', labelcolor=color_gt)
        
        ax1.set_title(title, fontsize=11, fontweight='bold', pad=4)
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', facecolor='white', framealpha=0.9, fontsize=9)
        
    axes[-1].set_xlabel("Date (Hourly Raw Measurements - 14 Days Window)", fontsize=11, fontweight='bold')
    fig.suptitle("Raw Hourly Data Comparison: MOX Sensors vs Ground Truth Reference Analyzer (No Smoothing)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "air_quality_raw_sensor_vs_groundtruth.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_raw_full_year(df):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Raw Sensors Full Year
    sensor_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
    for col in sensor_cols:
        axes[0].plot(df.index, df[col], linewidth=0.5, alpha=0.7, label=col)
    axes[0].set_title("Full 1-Year Unfiltered Raw Data: 5 MOX Sensors", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Sensor Resistance / Signal", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # Raw Ground Truth Full Year
    gt_cols = ['C6H6(GT)', 'CO(GT)', 'NO2(GT)']
    for col in gt_cols:
        axes[1].plot(df.index, df[col], linewidth=0.5, alpha=0.7, label=col)
    axes[1].set_title("Full 1-Year Unfiltered Raw Data: Ground Truth Concentrations", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Concentration (Raw Units)", fontsize=10)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "air_quality_raw_full_year.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_monthly_trends(df):
    numeric_df = df.select_dtypes(include=[np.number])
    df_monthly = numeric_df.resample('ME').mean()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color1 = '#d62728'
    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_ylabel('Benzene C6H6 (µg/m³) & CO (mg/m³)', color=color1, fontsize=11, fontweight='bold')
    l1 = ax1.plot(df_monthly.index.strftime('%Y-%m'), df_monthly['C6H6(GT)'], marker='o', color=color1, linewidth=2, label='C6H6 (GT)')
    l2 = ax1.plot(df_monthly.index.strftime('%Y-%m'), df_monthly['CO(GT)'], marker='s', color='#ff7f0e', linewidth=2, label='CO (GT)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.set_ylabel('Temperature (°C)', color=color2, fontsize=11, fontweight='bold')
    l3 = ax2.plot(df_monthly.index.strftime('%Y-%m'), df_monthly['T'], marker='^', color=color2, linewidth=2, linestyle='--', label='Temperature (°C)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True)
    
    plt.title("12-Month Seasonal Average Trends: Benzene, CO vs Temperature", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "air_quality_monthly_trends.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("Generating time series plots...")
    plot_raw_sensor_vs_groundtruth(df, days=30)
    plot_raw_full_year(df)
    plot_mox_sensors(df)
    plot_ground_truth(df)
    plot_environmental(df)
    plot_diurnal_profile(df)
    plot_monthly_trends(df)
    print("All time series plots successfully created!")
