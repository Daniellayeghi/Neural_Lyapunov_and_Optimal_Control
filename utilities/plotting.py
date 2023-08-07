import matplotlib.pyplot as plt
import pandas as pd


def plot_reward_graph_multi(csv_path: str, experiment_name: str):
    # Load the data from the CSV
    data = pd.read_csv(csv_path)

    # Compute mean and standard deviation
    mean_rewards = data.mean(axis=1)
    std_rewards = data.std(axis=1)

    # Create a new figure instance
    plt.figure(figsize=(12, 7))

    plt.plot(mean_rewards, label=f"Mean Reward ({experiment_name})", color="dodgerblue")
    plt.fill_between(data.index,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     color="dodgerblue", alpha=0.3)

    plt.title(f"Mean Reward for {experiment_name}")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
