import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})

# Paths
OUTPUT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(
    OUTPUT_DIR,
    "zip",
    "Advanced Gas Detection and Classification using MQ Series Sensors Integrated with Machine Learning and Deep Learning Techniques",
    "MQ135SensorData.csv"
)

def load_data():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    class_counts = df['Class'].value_counts().sort_index()
    
    colors = sns.color_palette("Set2", len(class_counts))
    bars = plt.bar(class_counts.index.astype(str), class_counts.values, color=colors, edgecolor='black', alpha=0.85)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 300,
                 f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title("Sample Distribution Across 7 Gas Target Classes (Class 0 - 6)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Gas Target Class", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Samples", fontsize=12, fontweight='bold')
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_class_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_raw_timeseries(df, num_samples=300):
    # Select a tight 300-sample window for ultra-clear point-by-point inspection
    sample_df = df.iloc[10000:10000+num_samples].copy()
    
    fig, axes = plt.subplots(6, 1, figsize=(15, 12), sharex=True)
    gas_cols = [f'Gas{i}' for i in range(1, 7)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for ax, col, color in zip(axes, gas_cols, colors):
        ax.plot(sample_df.index, sample_df[col], color=color, linewidth=1.5, marker='o', markersize=4, alpha=0.9, label=f"Raw {col}")
        ax.set_ylabel(f"{col} (ADC/Volt)", fontsize=9, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    axes[0].set_title(f"High-Detail Ultra-Short Time Series ({num_samples} Consecutive Samples): Raw MQ Sensor Outputs", fontsize=14, fontweight='bold', pad=12)
    axes[-1].set_xlabel("Sample Index", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_raw_timeseries.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_ppm_timeseries(df, num_samples=300):
    # Select the same tight 300-sample window for direct comparison
    sample_df = df.iloc[10000:10000+num_samples].copy()
    
    fig, axes = plt.subplots(6, 1, figsize=(15, 12), sharex=True)
    ppm_cols = [f'Gas{i} PPM' for i in range(1, 7)]
    colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e']
    
    for ax, col, color in zip(axes, ppm_cols, colors):
        ax.plot(sample_df.index, sample_df[col], color=color, linewidth=1.5, marker='s', markersize=4, alpha=0.9, label=col)
        ax.set_ylabel(f"{col}", fontsize=9, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
        
    axes[0].set_title(f"High-Detail Ultra-Short Time Series ({num_samples} Consecutive Samples): Converted Gas PPM", fontsize=14, fontweight='bold', pad=12)
    axes[-1].set_xlabel("Sample Index", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_ppm_timeseries.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_raw_by_class(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    gas_cols = [f'Gas{i}' for i in range(1, 7)]
    palette = sns.color_palette("Set2", 7)
    
    for ax, col in zip(axes, gas_cols):
        # Clip extreme visualization outliers for clean boxplot representation if any
        sns.boxplot(data=df, x='Class', y=col, ax=ax, palette=palette, fliersize=1)
        ax.set_title(f"Distribution of {col} across Gas Classes", fontsize=11, fontweight='bold')
        ax.set_xlabel("Gas Target Class", fontsize=10)
        ax.set_ylabel("Raw Sensor Value", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        
    fig.suptitle("Raw MQ Sensor Values Disaggregated by Gas Target Class (Class 0 - 6)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_raw_by_class.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    feature_cols = [f'Gas{i}' for i in range(1, 7)] + [f'Gas{i} PPM' for i in range(1, 7)] + ['Class']
    corr = df[feature_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    
    plt.title("Correlation Matrix: Raw MQ Sensors, PPM Concentrations & Target Class", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_correlation_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_sample_period_zoom(df, num_samples=3000):
    # Take a 3,000 sample segment to show fine time-series dynamics and class transitions
    sample_df = df.iloc[10000:10000+num_samples].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    gas_cols = [f'Gas{i}' for i in range(1, 7)]
    for col in gas_cols:
        ax1.plot(sample_df.index, sample_df[col], label=col, linewidth=1.2, alpha=0.85)
        
    ax1.set_title(f"High-Resolution Segment (3,000 Consecutive Samples): Raw Sensors vs Class Transitions", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Raw Sensor Signal (ADC)", fontsize=10, fontweight='bold')
    ax1.legend(loc="upper right", frameon=True, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot Class timeline on second subplot
    ax2.plot(sample_df.index, sample_df['Class'], color='black', drawstyle='steps-post', linewidth=1.5, label='Class Target')
    ax2.set_ylabel("Gas Class", fontsize=10, fontweight='bold')
    ax2.set_yticks(range(7))
    ax2.set_xlabel("Sample Index", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "advanced_gas_sample_period.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    df = load_data()
    print("Generating Advanced Gas Detection visual plots...")
    plot_class_distribution(df)
    plot_raw_timeseries(df)
    plot_ppm_timeseries(df)
    plot_raw_by_class(df)
    plot_correlation_matrix(df)
    plot_sample_period_zoom(df)
    print("All Advanced Gas Detection plots successfully created!")
