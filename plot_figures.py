from utilities.plotting import plot_trajectories_on_level_sets
import torch

if __name__ == "__main__":
    # Test the function with the newly uploaded CSV file
    # save_phase_plane_from_variable_data("../data/CP_balancing_LYAP.csv")
    from PSDNets import ICNN
    import torch.nn.functional as F
    nn_value_func = ICNN([3, 64, 64, 1], F.softplus).to('cpu')
    nn_value_func.load_state_dict(torch.load('DI_LYAP_m-fwd_d-1_s-0.005_seed4.pt'))
    plot_trajectories_on_level_sets(nn_value_func, './data/DI_balancing_traj4.csv')