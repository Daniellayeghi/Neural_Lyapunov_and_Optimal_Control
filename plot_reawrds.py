import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_rewards_with_positive_variance(task_name, total_timesteps, sac_csv_path, ppo_csv_path):
    """
    Plots the rewards for SAC and PPO, ensuring the shaded variance region does not imply negative rewards.

    Parameters:
    task_name (str): The title of the plot, indicating the task name.
    total_timesteps (int): The total number of timesteps to scale the data.
    sac_csv_path (str): The file path to the SAC rewards CSV.
    ppo_csv_path (str): The file path to the PPO rewards CSV.
    """
    # Load and process the datasets
    sac_data = pd.read_csv(sac_csv_path).abs() /100
    ppo_data = pd.read_csv(ppo_csv_path).abs()

    # Compute mean and standard deviation for both SAC and PPO
    mean_sac = sac_data.mean(axis=1)
    std_dev_sac = sac_data.std(axis=1)
    mean_ppo = ppo_data.mean(axis=1)
    std_dev_ppo = ppo_data.std(axis=1)

    # Rescale the index to fit the total timesteps
    x = np.linspace(0, total_timesteps, len(sac_data))

    # Plotting adjustments
    plt.figure(figsize=(3.5, 2.8))
    plt.plot(x, mean_sac, label='Average SAC Costs x 1e-2', color='green', linewidth=1)
    plt.fill_between(x, np.maximum(mean_sac - std_dev_sac, 0), mean_sac + std_dev_sac, color='green', alpha=0.2)
    plt.plot(x, mean_ppo, label='Average PPO Costs', color='orange', linewidth=1)
    plt.fill_between(x, np.maximum(mean_ppo - std_dev_ppo, 0), mean_ppo + std_dev_ppo, color='orange', alpha=0.2)
    plt.title(f'{task_name} Costs', fontsize=10)
    plt.xlabel('Total Timesteps', fontsize=8)
    plt.ylabel('Costs', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(loc='upper right', fontsize=6)
    plt.grid(True, linestyle='--')
    plt.gca().set_facecolor('#f2f2f2')
    plt.tight_layout(pad=0.5)

    file_name = f'{task_name.replace(" ", "_").lower()}_rewards_ieee.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')

    plt.show()


def main():
    """
    Main function to be called with arguments from the terminal.
    Expected arguments: task_name, total_timesteps, sac_csv_path, ppo_csv_path.
    """
    if len(sys.argv) != 5:
        print("Usage: python script.py <task_name> <total_timesteps> <sac_csv_path> <ppo_csv_path>")
        sys.exit(1)
    print(sys.argv)
    task_name = sys.argv[1]
    total_timesteps = int(sys.argv[2])
    sac_csv_path = sys.argv[3]
    ppo_csv_path = sys.argv[4]

    plot_rewards_with_positive_variance(task_name, total_timesteps, sac_csv_path, ppo_csv_path)


if __name__ == "__main__":
    main()
