import csv


def save_trajectory_to_csv(traj, loss, filename, state_elements=None):
    traj_np = traj.detach().numpy()
    loss_np = loss.detach().numpy()

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trajectory", "Loss"])

        for i in range(traj_np.shape[1]):  # Loop over batch size
            # Optionally filter out specific state elements
            if state_elements is not None:
                traj_row = traj_np[:, i, 0, state_elements].flatten()
            else:
                traj_row = traj_np[:, i, 0, :].flatten()

            loss_value = loss_np[i]
            writer.writerow([*traj_row, loss_value])


import pandas as pd


# Update the function to include the value of the cost at the midpoints and the average final loss at the last iteration
def analyze_chunk_gradients_with_cost(csv_path):
    data = pd.read_csv(csv_path)
    data['mean_value'] = data.mean(axis=1)
    chunk_size = len(data) // 6
    print("Chunk Analysis:")

    for i in range(0, len(data) - chunk_size + 1, chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        gradient = chunk['mean_value'].diff().abs().mean()
        middle_index = i + chunk_size // 2
        midpoint_cost = data.loc[middle_index, 'mean_value']

        print(
            f"Chunk starting at index {i}: Gradient = {gradient}, Midpoint index = {middle_index}, Cost at Midpoint = {midpoint_cost}"
        )
    # Calculate the average final loss at the last iteration
    avg_final_loss = data.loc[len(data) - 1, 'mean_value']
    print(f"Average Final Loss at Last Iteration: {avg_final_loss}")
    return avg_final_loss


# Update the function to also return the target cost
def find_iteration_below_cost(main_csv_path, reference_csv_path):
    main_data = pd.read_csv(main_csv_path)
    main_data['mean_value'] = main_data.mean(axis=1)
    reference_data = pd.read_csv(reference_csv_path)
    reference_data['mean_value'] = reference_data.mean(axis=1)
    target_cost = -reference_data.loc[len(reference_data) - 1, 'mean_value']
    below_target = main_data[main_data['mean_value'] < target_cost]

    if below_target.empty:
        return "No iteration found where the average cost is below the target cost.", target_cost
    else:
        return int(below_target.iloc[0].name), target_cost

def get_final_cost_stats(csv_path):
    data = pd.read_csv(csv_path)
    final_row = data.iloc[-1]
    final_avg_cost = final_row.mean()
    final_cost_std = final_row.std()

    return final_avg_cost, final_cost_std


