import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("husamksalih/array-of-gas-sensors-data")
csv_file = os.path.join(path, "All Samples.csv")

df = pd.read_csv(csv_file)
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Set up the plotting style
sns.set_theme(style="whitegrid")

# 1. Plot Average Sensor Values per Class (MQ_2, MQ_4, MQ_135, MQ_136, MQ_8)
sensor_cols = ['MQ_2', 'MQ_4', 'MQ_135', 'MQ_136', 'MQ_8','Temp','Hum']
if all(col in df.columns for col in sensor_cols) and 'Sampe/Class' in df.columns:
    plt.figure(figsize=(14, 8))
    # Melt the dataframe for easier plotting with seaborn
    df_melt = df.melt(id_vars=['Sampe/Class'], value_vars=sensor_cols, var_name='Sensor', value_name='Value')
    sns.barplot(data=df_melt, x='Sampe/Class', y='Value', hue='Sensor', errorbar=None)
    plt.title('Average Gas Sensor Readings by Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('average_sensors_by_class.png')
    plt.close()

# 2. Plot Temperature and Humidity by Class
if 'temp' in df.columns and 'Hum' in df.columns:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Sampe/Class', y='temp')
    plt.title('Temperature Distribution by Class')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Sampe/Class', y='Hum')
    plt.title('Humidity Distribution by Class')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('temp_hum_by_class.png')
    plt.close()

# Append to analysis.txt
with open("analysis.txt", "a", encoding="utf-8") as f:
    f.write("\nData Analysis & Visualizations Generated:\n")
    f.write("1. average_sensors_by_class.png: Shows the average reading of MQ_2, MQ_4, MQ_135, MQ_136, and MQ_8 gas sensors for each sample class (Coffee, Garlic, etc.). Different substances trigger different sensor responses.\n")
    f.write("2. temp_hum_by_class.png: Shows the distribution of temperature and humidity during the measurement of each substance class. Notice how some experiments might have been conducted under slightly different environmental conditions.\n")
    
print("Plots generated successfully!")
