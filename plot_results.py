import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/media/disk3/mateusz/ppo/results/ppo_mjx_losses_20251007_163736.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Drop duplicate iterations (keep first occurrence)
df_unique = df.drop_duplicates(subset=["iteration"])

# Plot
plt.plot(df_unique["iteration"], df_unique["avg_return"])
plt.xlabel("Iteration")
plt.ylabel("Average Return")
plt.title("PPO MJX: Average Return vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_return_vs_iteration.png")
plt.show()