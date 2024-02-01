import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_ieee_style_plot(task_name, data_type, total_timesteps, data):
    """
    Saves a plot styled for IEEE two-column paper, with high DPI and minimal boundaries.

    Parameters:
    task_name (str): The name of the task.
    data_type (str): 'cost' or 'loss' to specify the type of data to plot.
    total_timesteps (int): The total number of timesteps to scale the data.
    data_cost (DataFrame): The dataset containing cost information.
    data_loss (DataFrame): The dataset containing loss information.
    """

    # Select the appropriate dataset
    if data_type == 'cost':
        data = data
        ylabel = 'Cost'
        color = 'red'
        title = f'{task_name} Trajectory {ylabel}'
    elif data_type == 'loss':
        data = data
        ylabel = 'Loss'
        color = 'blue'
        title = f'{task_name} Constraint {ylabel}'
    else:
        raise ValueError("data_type must be 'cost' or 'loss'")

    # Drop the 'Step' column and rescale the data to fit the total timesteps
    data_rescaled = data.drop('Step', axis=1)
    data_rescaled.index = np.linspace(0, total_timesteps, len(data_rescaled))

    # Compute mean and standard deviation across different runs
    mean = data_rescaled.mean(axis=1)
    std_dev = data_rescaled.std(axis=1)

    # Plotting with IEEE style formatting
    plt.figure(figsize=(3.5, 2.8))  # Width of a column in a two-column format is usually around 3.5 inches
    plt.plot(mean, label=f'Mean {ylabel}', color=color, linewidth=1)
    plt.fill_between(mean.index, mean - std_dev, mean + std_dev, color=color, alpha=0.2)
    plt.title(title, fontsize=10)
    plt.xlabel('Total Timesteps', fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(loc='upper right', fontsize=6)
    plt.grid(True, linestyle='--')
    plt.gca().set_facecolor('#f2f2f2')  # Grey background for the plot area
    plt.tight_layout(pad=0.5)  # Reduce padding/margin

    # Saving the plot with high DPI and minimal boundary
    file_name = f'{task_name.replace(" ", "_").lower()}_trajectory_{ylabel.lower()}_ieee.png'
    plt.savefig(file_name, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the plot to prevent displaying it inline

    return file_name


# Save the IEEE styled plot with the specified settings
import sys


def main():
    """
    Main function to be called with arguments from the terminal.
    Expected arguments: task_name, data_type, total_timesteps, csv_path_cost, csv_path_loss.
    """
    if len(sys.argv) != 5:
        print("Usage: python script.py <task_name> <data_type> <total_timesteps> <csv_path_cost> <csv_path_loss>")
        sys.exit(1)

    # Parse arguments
    task_name = sys.argv[1]
    data_type = sys.argv[2]
    try:
        total_timesteps = int(sys.argv[3])
    except ValueError:
        print("The total_timesteps argument must be an integer.")
        sys.exit(1)

    # CSV file paths for cost and loss data
    path = sys.argv[4]

    # Load data
    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"An error occurred while loading the CSV files: {e}")
        sys.exit(1)

    # Save the plot
    file_path = save_ieee_style_plot(task_name, data_type, total_timesteps, data)
    print(f"Plot saved as {file_path}")


# This check ensures that the main function is not called when the script is imported as a module
if __name__ == "__main__":
    main()



