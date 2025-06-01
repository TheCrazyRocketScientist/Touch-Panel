import pandas as pd
import matplotlib.pyplot as plt
import sys

# List your CSV file paths here or pass as command-line arguments
file_paths = sys.argv[1:]  # python plot_time_series.py file1.csv file2.csv file3.csv

if len(file_paths) < 1:
    print("Usage: python plot_time_series.py file1.csv file2.csv file3.csv")
    sys.exit(1)

# Optional colors if you want to customize per file
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axs[0].set_title("Accelerometer X")
axs[1].set_title("Accelerometer Y")
axs[2].set_title("Accelerometer Z")

for i, file in enumerate(file_paths):
    df = pd.read_csv(file)

    # Convert 'timestamp' to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    label = f"Sensor {i+1} ({file})"
    color = colors[i % len(colors)]

    axs[0].plot(df.index, df['x'], label=label, color=color)
    axs[1].plot(df.index, df['y'], label=label, color=color)
    axs[2].plot(df.index, df['z'], label=label, color=color)

for ax in axs:
    ax.legend()
    ax.grid(True)

axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()
