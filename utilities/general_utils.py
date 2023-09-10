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
