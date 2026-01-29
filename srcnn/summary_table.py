import os
import pandas as pd
from tabulate import tabulate
import glob

# Base folder containing all experiments
BASE_DIR = "outputs"

summaries = []

# Recursively find all training_log.csv files
for log in glob.glob(os.path.join(BASE_DIR, "**/training_log.csv"), recursive=True):
    df = pd.read_csv(log)
    # Go up two levels to get the experiment folder as model name
    model_name = os.path.basename(os.path.dirname(os.path.dirname(log)))
    best_psnr = df["val_psnr"].max()
    best_epoch = df["val_psnr"].idxmax()
    final_loss = df["train_loss"].iloc[-1]
    summaries.append({
        "Model": model_name,
        "Best PSNR": best_psnr,
        "Best Epoch": best_epoch,
        "Final Loss": final_loss
    })

# Convert to DataFrame
summary_df = pd.DataFrame(summaries)

# Sort by Best PSNR descending
summary_df = summary_df.sort_values(by="Best PSNR", ascending=False)

# Display table
print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False))