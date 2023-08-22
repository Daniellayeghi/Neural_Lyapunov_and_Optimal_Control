import torch
import numpy as np
from gym_baselines.gym_models import CustomCartpole
from models import Cartpole, ModelParams
from gym_baselines.gym_configs import configurations as config
torch.manual_seed(0)


def __gym_func(u):
    res = gym_model.step(u)
    obs = res[0]
    obs.flatten()


def __cst_func(x, u):
    xd_new = cst_model(x, u)
    x = x + xd_new * 0.01
    return x


def __simulate_custom(func, x, u):
    res_traj = []
    for t in range(u.shape[0]):
        xd_new = func(x, u[t])
        x = x + xd_new * 0.01
        res_traj.append(x.flatten().numpy())

    return res_traj


def __simulate_gym(model, x, u):
    res_traj = []
    model.state = x
    for t in range(u.shape[0]):
        action = u[t]
        res = model.step(action)
        obs = res[0]
        res_traj.append(obs.flatten())

    return res_traj


# def __simulate_models(x_inits, us):
#     x_init_cst, x_inits_gym = x_inits
#     u_cst, u_gym = us
#     time = u_cst.shape[0]
#     gym_model.state = x_inits_gym
#     res_gym, res_cst = [], []
#     for t in range(time):
#         res_gym.append(__gym_func(u_gym[t]))
#         x_init_cst = __cst_func(x_init_cst, u_cst[t])
#         res_cst.append(x_init_cst.flatten().numpy())
#
#     return res_gym, res_cst

if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cst_model = Cartpole(1, cp_params, 'cpu', mode='fwd')
    cst_model.GEAR = 1
    cst_model.LENGTH = 0.3
    cst_model.MASS_P = 0.1
    cst_model.FRICTION = torch.Tensor([0.0, 0.1])

    gym_model = CustomCartpole(**config['CP_BALANCE_SAC']['model_params'])

    # init ctrl
    u_cst = torch.rand(100, 1, 1)
    u_gym = u_cst.reshape(100, 1).numpy()

    # init x
    qc_init = torch.FloatTensor(1, 1, 1).uniform_(-2, 2) * 1
    qp_init = torch.FloatTensor(1, 1, 1).uniform_(-0.6, 0.6)
    qd_init = torch.FloatTensor(1, 1, 2).uniform_(0.01, 0.01)
    x_init_cst = torch.cat((qc_init, qp_init, qd_init), 2).to('cpu')
    x_init_gym = x_init_cst.squeeze().numpy()

    # res_gym1, res_cst1 = __simulate_models((x_init_cst, x_init_gym), (u_cst, u_gym))

    res_gym = __simulate_gym(gym_model, x_init_gym, u_gym)
    res_cst = __simulate_custom(cst_model, x_init_cst, u_cst)

    import matplotlib.pyplot as plt
    res_gym = np.array(res_gym)
    res_cst = np.array(res_cst)
    diff = res_gym - res_cst
    plt.plot(diff)
    plt.show()


