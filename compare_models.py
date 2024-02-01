import torch
import numpy as np
from gym_baselines.gym_models import CustomCartpole, CustomCartpoleBalance, CustomReacher, CustomDoubleIntegrator
from models import Cartpole, ModelParams, TwoLink2, DoubleIntegrator
from gym_baselines.gym_configs import configurations as config
torch.manual_seed(0)


def __gym_func(u):
    res = gym_model.step(u)
    obs = res[-1]
    obs.flatten()


def __cst_func(x, u):
    xd_new = cst_model(x, u)
    x = x + xd_new * 0.01
    return x


def __simulate_custom(func, x, u):
    res_traj = []
    for t in range(u.shape[0]):
        xd_new = func(x, u[t].reshape(1, 1, u.shape[-1]))
        x = x + xd_new * 0.01
        res_traj.append(x.flatten().numpy())

    return res_traj


def __simulate_gym(model, x, u):
    res_traj = []
    model.state = x
    for t in range(u.shape[0]):
        action = u[t]
        res = model.step(action)
        obs = res[-1]['state']
        res_traj.append(obs.flatten())

    return res_traj


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cst_model = Cartpole(1, cp_params, 'cpu', mode='fwd')
    cst_model.GEAR = 1
    cst_model.LENGTH = 1
    cst_model.MASS_P = 1
    cst_model.FRICTION = torch.Tensor([0.0, 0.1])

    gym_model = CustomCartpole(**config['CP_BALANCE_SAC']['model_params'])

    # init ctrl
    u_cst = torch.rand(300, 1, 1)
    u_gym = u_cst.reshape(300, 1).numpy()

    # init x
    qc_init = torch.FloatTensor(1, 1, 1).uniform_(-2, 2) * 1
    qp_init = torch.FloatTensor(1, 1, 1).uniform_(-0.6, 0.6)
    qd_init = torch.FloatTensor(1, 1, 2).uniform_(0.01, 0.01)
    x_init_cst = torch.cat((qc_init, qp_init, qd_init), 2).to('cpu')
    x_init_gym = x_init_cst.squeeze().numpy()


    res_gym = __simulate_gym(gym_model, x_init_gym, u_gym)
    res_cst = __simulate_custom(cst_model, x_init_cst, u_cst)

    res_gym = np.array(res_gym)
    res_cst = np.array(res_cst)
    diff = res_gym - res_cst
    norm_diff = np.linalg.norm(diff, axis=1)
    print(f"Swing up model comparison max trajectory_difference {max(norm_diff)}")
    # plt.plot(diff)
    # plt.show()

    #### Balancing comparison ######
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cst_model = Cartpole(1, cp_params, 'cpu', mode='fwd')
    cst_model.GEAR = 1
    cst_model.LENGTH = .3
    cst_model.MASS_P = .1
    cst_model.FRICTION = torch.Tensor([0.1, 0.1])

    gym_model = CustomCartpoleBalance(**config['CP_BALANCE_SAC']['model_params'])

    u_cst = torch.rand(300, 1, 1)
    u_gym = u_cst.reshape(300, 1).numpy()

    # init x
    qc_init = torch.FloatTensor(1, 1, 1).uniform_(-2, 2) * 1
    qp_init = torch.FloatTensor(1, 1, 1).uniform_(-0.6, 0.6)
    qd_init = torch.FloatTensor(1, 1, 2).uniform_(0.01, 0.01)
    x_init_cst = torch.cat((qc_init, qp_init, qd_init), 2).to('cpu')
    x_init_gym = x_init_cst.squeeze().numpy()

    res_gym = __simulate_gym(gym_model, x_init_gym, u_gym)
    res_cst = __simulate_custom(cst_model, x_init_cst, u_cst)

    # import matplotlib.pyplot as plt
    res_gym = np.array(res_gym)
    res_cst = np.array(res_cst)
    diff = res_gym - res_cst
    norm_diff = np.linalg.norm(diff, axis=1)
    print(f"CP balancing model comparison max trajectory_difference {max(norm_diff)}")
    # plt.plot(diff)
    # plt.show()

    #### Two link comparison ######
    tl_params = ModelParams(2, 2, 1, 4, 4)
    cst_model = TwoLink2(1, tl_params, device='cpu', mode='fwd')

    gym_model = CustomReacher(**config['TL_SAC']['model_params'])

    u_cst = torch.rand(300, 1, 2)
    u_gym = u_cst.reshape(300, 2).numpy()

    # init x
    qc_init = torch.FloatTensor(1, 1, 1).uniform_(-2, 2) * 1
    qp_init = torch.FloatTensor(1, 1, 1).uniform_(-0.6, 0.6)
    qd_init = torch.FloatTensor(1, 1, 2).uniform_(0.01, 0.01)
    x_init_cst = torch.cat((qc_init, qp_init, qd_init), 2).to('cpu')
    x_init_gym = x_init_cst.squeeze().numpy()

    res_gym = __simulate_gym(gym_model, x_init_gym, u_gym)
    res_cst = __simulate_custom(cst_model, x_init_cst, u_cst)

    # import matplotlib.pyplot as plt
    res_gym = np.array(res_gym)
    res_cst = np.array(res_cst)
    diff = res_gym - res_cst
    norm_diff = np.linalg.norm(diff, axis=1)
    print(f"TL model comparison max trajectory_difference {max(norm_diff)}")

    #### double integrator comparison ######
    di_params = ModelParams(1, 1, 1, 2, 2)
    cst_model = DoubleIntegrator(1, di_params, device='cpu', mode='fwd')

    gym_model = CustomDoubleIntegrator(**config['DI_SAC']['model_params'])

    u_cst = torch.rand(300, 1, 1)
    u_gym = u_cst.reshape(300, 1).numpy()

    # init x
    qc_init = torch.FloatTensor(1, 1, 1).uniform_(-2, 2) * 1
    qd_init = torch.FloatTensor(1, 1, 1).uniform_(0.01, 0.01)
    x_init_cst = torch.cat((qc_init, qd_init), 2).to('cpu')
    x_init_gym = x_init_cst.squeeze().numpy()

    res_gym = __simulate_gym(gym_model, x_init_gym, u_gym)
    res_cst = __simulate_custom(cst_model, x_init_cst, u_cst)

    import matplotlib.pyplot as plt
    res_gym = np.array(res_gym)
    res_cst = np.array(res_cst)
    diff = res_gym - res_cst
    norm_diff = np.linalg.norm(diff, axis=1)
    print(f"DI model comparison max trajectory_difference {max(norm_diff)}")
