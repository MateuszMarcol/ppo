import pandas as pd
import matplotlib.pyplot as plt

# Paths to your files
file1 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251007_163736.csv"
file2 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251010_094126.csv"  # Change to your second file

def get_avg_return_per_iteration(path):
    df = pd.read_csv(path)
    # Take the first row for each iteration
    df_unique = df.drop_duplicates(subset=["iteration"])
    return df_unique["iteration"], df_unique["avg_return"]

iter1, avg1 = get_avg_return_per_iteration(file1)
iter2, avg2 = get_avg_return_per_iteration(file2)

plt.plot(iter1, avg1, label="No entropy", marker="o")
plt.plot(iter2, avg2, label="Entropy", marker="s")
plt.xlabel("Iteration")
plt.ylabel("Average Return")
plt.title("PPO Average return Hopper no entropy vs. entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_return_comparison.png")
plt.show()