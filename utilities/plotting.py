
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Circle



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


# Modify the function to save the plot as a high-quality image
def save_phase_plane_from_variable_data(trajectories, losses, save_path):
    """
    Save the phase plane of trajectories with variable lengths as a high-quality image.

    Parameters:
    - trajectories (list): A list of numpy arrays, where each array is a trajectory.
    - losses (list): A list of loss values corresponding to each trajectory.
    - save_path (str): The path to save the high-quality image.
    """

    # Initialize high-quality 2D plot with LaTeX annotations in legends
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, color='black')

    # Lists to store initial conditions
    safe_starts = []
    unsafe_starts = []

    # Flags to control legend entry addition
    add_safe_legend = True
    add_unsafe_legend = True

    # Plot trajectories and collect initial conditions
    for i in range(len(losses)):
        trajectory = trajectories[i]
        loss = losses[i]
        positions = trajectory[::2]
        velocities = trajectory[1::2]
        if loss == 0:
            ax.plot(positions, velocities, c='g', label='Safe Trajectory' if add_safe_legend else "")
            ax.scatter(positions[0], velocities[0], c='g', marker='o')
            safe_starts.append((positions[0], velocities[0]))
            add_safe_legend = False  # Only add 'Safe Trajectory' to the legend once
        else:
            ax.plot(positions, velocities, c='r', linestyle='--',
                    label='Unsafe Trajectory' if add_unsafe_legend else "")
            ax.scatter(positions[0], velocities[0], c='r', marker='o')
            unsafe_starts.append((positions[0], velocities[0]))
            add_unsafe_legend = False  # Only add 'Unsafe Trajectory' to the legend once

    # Add shaded regions
    if len(safe_starts) >= 3:
        hull = ConvexHull(np.array(safe_starts))
        vertices = hull.vertices
        polygon = Polygon(np.array(safe_starts)[vertices], closed=True, facecolor='g', alpha=0.2,
                          label=r'Safe Start Region: $\frac{dv(\mathbf{x}, t)}{dt} \leq -\ell(\mathbf{x}, t)$')
        ax.add_patch(polygon)
    ax.set_facecolor('mistyrose')

    # Add legend for unsafe start region
    unsafe_region_patch = Polygon([[0, 0]], closed=True, facecolor='mistyrose', alpha=0.2,
                                  label=r'Unsafe Start Region: $\frac{dv(\mathbf{x}, t)}{dt} \geq -\ell(\mathbf{x}, t)$')
    ax.add_patch(unsafe_region_patch)

    # Labels, title, and legend with LaTeX annotations
    ax.set_xlabel(r'$\theta$ (Position)')
    ax.set_ylabel(r'$\dot{\theta}$ (Velocity)')
    ax.set_title('Phase Plane of Trajectories')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Save the plot as a high-quality image
    plt.savefig(save_path, dpi=300, format='png')

    # Function to plot filled-in plots for mean and standard deviation of loss values


def plot_filled_loss(steps, loss_values, time_steps_per_iteration, title, smoothing_alpha=None):
    # Apply exponential smoothing if the alpha value is provided
    if smoothing_alpha is not None:
        loss_values = [exponential_smoothing(loss, smoothing_alpha) for loss in loss_values]

    # Calculate the mean and standard deviation for the loss at each step
    mean_loss = np.mean(loss_values, axis=0)
    std_loss = np.std(loss_values, axis=0)

    # Calculate the total time steps for each iteration step
    total_time_steps = steps * time_steps_per_iteration

    # Create the plot
    plt.figure(figsize=(14, 8), dpi=300)

    # Plot the mean loss
    plt.plot(total_time_steps, mean_loss, label='Mean Total Cost', color='b', linewidth=2)

    # Add shaded region for standard deviation
    plt.fill_between(total_time_steps, mean_loss - std_loss, mean_loss + std_loss, color='b', alpha=0.2)

    # Add labels, title, and legend
    plt.xlabel('Total Time Steps', fontsize=14)
    plt.ylabel('Total Cost', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()


def exponential_smoothing(series, alpha):

    smoothed_series = []
    for i in range(len(series)):
        if i == 0:
            smoothed_series.append(series[i])
        else:
            smoothed_series.append(alpha * series[i] + (1 - alpha) * smoothed_series[-1])
    return smoothed_series


import torch

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def plot_trajectories_on_level_sets(model, csv_file_path):
    plt.figure(figsize=(7, 5))  # Adjust figure size to fit double-column IEEE paper

    model.eval()

    df = pd.read_csv(csv_file_path)
    trajectories = df.values

    time_tensor = torch.tensor(0.0)

    x_range = np.linspace(-1.5, 1.5, 20)
    y_range = np.linspace(-1.5, 1.5, 20)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            state = torch.tensor((x, y)).float().to('cpu').unsqueeze(0)
            Z[i, j] = model(time_tensor, state).item()

    plt.contourf(X, Y, Z, levels=50, cmap='viridis')

    safe_initial_conditions = []
    legend_handles = []

    for traj in trajectories:
        x_vals = traj[:-1:2]
        y_vals = traj[1:-1:2]
        if traj[-1] == 0:
            color = 'g'
            safe_initial_conditions.append((x_vals[0], y_vals[0]))
        else:
            color = 'r'

        plt.plot(x_vals, y_vals, color=color)
        plt.scatter(x_vals[0], y_vals[0], color=color, s=5)  # Smaller bullet points at the starting point

    legend_handles.extend([
        plt.Line2D([0], [0], color='g', label=r'$\frac{dv(\mathbf{x}, t)}{dt} \leq -\ell(\mathbf{x}, t)$ trajectories'),
        plt.Line2D([0], [0], color='r', label=r'$\frac{dv(\mathbf{x}, t)}{dt} \geq -\ell(\mathbf{x}, t)$ trajectories')
    ])

    if len(safe_initial_conditions) > 2:
        hull = ConvexHull(safe_initial_conditions)
        for simplex in hull.simplices:
            plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], "k-")

    legend_handles.append(plt.Line2D([0], [0], color='k', label='Convex hull of satisfied initial conditions'))

    plt.legend(handles=legend_handles, fontsize=8)

    plt.xlabel(r'$q$', fontsize=10)
    plt.ylabel(r'$\dot{q}$', fontsize=10)
    plt.title('Double Integrator Trajectories on Lyapunov Level Sets', fontsize=10)
    plt.tight_layout()
    plt.savefig("high_quality_plot.png", dpi=300)  # Save figure in high quality
    plt.show()

# Example usage:
# Define your neural network architecture in a class, let's say it's called MyLyapunovNN
# model = MyLyapunovNN()
# model.load_state_dict(torch.load('your_model_state_dict.pt'))
# plot_trajectories_on_level_sets(model, 'your_trajectories.csv')


# if __name__ == "__main__":
#     # # Test the function with the newly uploaded CSV file
#     # # save_phase_plane_from_variable_data("../data/CP_balancing_LYAP.csv")
#     # from PSDNets import ICNN
#     # import torch.functional as F
#     # nn_value_func = ICNN([3, 64, 64, 1], F.softplus).to('cpu')
#     # nn_value_func.load_state_dict(torch.load('DI_LYAP_m-fwd_d-1_s-0.005_seed4.pt'))
#     # plot_trajectories_on_level_sets(nn_value_func, '../data/DI_balancing_traj4.csv')