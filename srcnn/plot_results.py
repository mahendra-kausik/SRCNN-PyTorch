import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
from itertools import cycle

sns.set(style="whitegrid", font_scale=1.2)

# Base folder containing all experiments
BASE_DIR = "outputs"

plt.figure(figsize=(12, 6))

# Define colors and line styles
palette = sns.color_palette("tab10")  # up to 10 colors
line_styles = ['-', '--', '-.', ':']  # line styles
color_cycle = cycle(palette)
style_cycle = cycle(line_styles)

# Recursively find all training_log.csv files
for log in glob.glob(os.path.join(BASE_DIR, "**/training_log.csv"), recursive=True):
    df = pd.read_csv(log)
    # Go up two levels to get the experiment folder as label
    label = os.path.basename(os.path.dirname(os.path.dirname(log)))
    color = next(color_cycle)
    style = next(style_cycle)
    plt.plot(df["epoch"], df["val_psnr"], label=label, color=color, linestyle=style, linewidth=2)

plt.title("Number of Epochs vs. Average Validation PSNR")
plt.xlabel("Number of Epochs")
plt.ylabel("Average Validation PSNR (dB)")
plt.legend()
plt.tight_layout()

# Ensure results folder exists
os.makedirs("results", exist_ok=True)
plt.savefig("results/psnr_comparison.png")
plt.show()