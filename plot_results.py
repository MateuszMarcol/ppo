import pandas as pd
import matplotlib.pyplot as plt

# Paths to your files
file1 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251014_153107.csv"
file2 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251014_153122.csv" 
file3 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251015_103210.csv" 
file4 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251015_105735.csv" 
# file5 = "/media/disk3/mateusz/ppo_clean/results/ppo_mjx_losses_20251014_100829.csv" 


def get_avg_return_per_iteration(path):
    df = pd.read_csv(path)
    # Take the first row for each iteration
    df_unique = df.drop_duplicates(subset=["iteration"])
    return df_unique["iteration"], df_unique["avg_return"]

iter1, avg1 = get_avg_return_per_iteration(file1)
iter2, avg2 = get_avg_return_per_iteration(file2)
iter3, avg3 = get_avg_return_per_iteration(file3)
iter4, avg4 = get_avg_return_per_iteration(file4)
# iter5, avg5 = get_avg_return_per_iteration(file5)

plt.figure(figsize=(15, 10))
plt.plot(iter1, avg1, label="baseline")
plt.plot(iter2, avg2, label="no entropy")
plt.plot(iter3, avg3, label="distributional critic")
plt.plot(iter4, avg4, label="distributional critic explore")
# plt.plot(iter5, avg5, label="Entropy + hyperparameters")
plt.xlabel("Iteration")
plt.ylabel("Average Return")
plt.title("PPO Average return Hopper, different settings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_return_comparison.png")
plt.show()