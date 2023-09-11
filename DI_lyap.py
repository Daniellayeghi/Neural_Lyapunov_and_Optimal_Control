
from models import DoubleIntegrator, ModelParams
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from PSDNets import ICNN
import wandb
import argparse
parser = argparse.ArgumentParser(description='Seed input')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
seed = args.seed

di_params = ModelParams(1, 1, 1, 2, 2)
sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 20, 400, 0.01)
di = DoubleIntegrator(sim_params.nsim, di_params, device)
max_iter, alpha, dt, discount, step, scale, mode = 100, .5, 0.01, 1, 0.005, 1, 'fwd'
Q = torch.diag(torch.Tensor([10, .1])).repeat(sim_params.nsim, 1, 1).to(device)*1
Qf = torch.diag(torch.Tensor([10, .1])).repeat(sim_params.nsim, 1, 1).to(device) * 10
R = torch.diag(torch.Tensor([.1])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))

torch.manual_seed(seed)

wandb.init(project='DI_lyap', anonymous='allow')

def plot_2d_funcition(xs: torch.Tensor, ys: torch.Tensor, xy_grid, f_mat, func, trace=None, contour=True):
    assert len(xs) == len(ys)
    trace = trace.detach().clone().cpu().squeeze()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            in_tensor = torch.tensor((x, y)).repeat(2, 1, 1).float().to(device)
            f_mat[i, j] = torch.mean(func(torch.tensor(0), in_tensor).detach().squeeze())

    [X, Y] = xy_grid
    f_mat = f_mat.cpu()
    plt.clf()
    ax = plt.axes()
    if contour:
        ax.contourf(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    plt.xlabel("q", fontsize=14)
    plt.ylabel("v", fontsize=14)
    plt.title("Lyapunov Function", fontsize=18)

    n_plots = trace.shape[1]
    for i in range(n_plots):
        ax.plot(trace[:, i, 0], trace[:, i, 1])
    plt.pause(0.001)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    return x


def batch_state_encoder(x: torch.Tensor):
    return x




def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT


import torch.nn.functional as F
nn_value_func = ICNN([sim_params.nqv+1, 64, 64, 1], F.softplus).to(device)


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-4].view(t-4, nsim, r, c).clone()
    x_final = x[-4:].view(4, nsim, r, c).clone()
    l_running = (loss_quadratic(x_run, Q)).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal), dim=0)


def inv_dynamics_reg(acc: torch.Tensor, alpha):
    u_batch = acc
    loss = u_batch @ R @ u_batch.mT
    return torch.sum(loss, 0)


def value_diff_loss(x: torch.Tensor, time):
    x_w = x.reshape(x.shape[0]*x.shape[1], x.shape[3])
    time = time.reshape(time.shape[0]*time.shape[1],  time.shape[3])
    values = nn_value_func(time, x_w).squeeze().reshape(x.shape[0], x.shape[1])
    value_differences = values[1:] - values[:-1]
    return value_differences.squeeze()


def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone().reshape(nsim, r, c)
    value_final = nn_value_func(torch.tensor(sim_params.ntime-1)/0.01, x_final).squeeze()
    return value_final


def batch_inv_dynamics_loss(x, acc, alpha):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = di._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = di._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    Tf = di._Tfric(v).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = ((M @ acc.mT).mT - C + Tf) * di._Bvec()
    return (u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale).squeeze()


def loss_function(x, xd, batch_time, alpha=1):
    x_running, acc_running = x[:-1].clone(), xd[:-1, ..., sim_params.nv:].clone()
    l_run_ctrl = batch_inv_dynamics_loss(x_running, acc_running, alpha) *0
    l_run_state = batch_state_loss(x_running)
    l_value_diff = value_diff_loss(x, batch_time)
    l_backup = torch.max((l_run_state + l_run_ctrl)*dt + l_value_diff, torch.zeros_like(l_value_diff))
    l_backup = torch.sum(l_backup, dim=0)
    # print(f"constaints: \n {l_backup.squeeze()} \n {x_init.squeeze()}")
    l_terminal = 0 * torch.square(value_terminal_loss(x))
    return torch.mean(l_backup + l_terminal), l_backup + l_terminal, torch.mean(torch.sum((l_run_state + l_run_ctrl), dim=0)).squeeze()


pos_arr = torch.linspace(-1, 1, 100).to(device)
vel_arr = torch.linspace(-1, 1, 100).to(device)
f_mat = torch.zeros((100, 100)).to(device)
[X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).requires_grad_(True).to(device)
time_input = time.clone().reshape(time.shape[0], 1, 1, 1).repeat(1, sim_params.nsim, 1, 1).requires_grad_(True)

dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=di, mode=mode, step=step, scale=scale, R=R
).to(device)

optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2)
lambdas = build_discounts(lambdas, discount).to(device)

log = f"DI_LYAP_m-{mode}_d-{discount}_s-{step}_seed{seed}"
wandb.watch(dyn_system, loss_function, log="all")
total_time_steps = 0


if __name__ == "__main__":

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 1
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * .7
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    trajectory = x_init.detach().clone().unsqueeze(0)
    iteration = 0
    traj = None
    while iteration < max_iter:
        optimizer.zero_grad()
        traj, dtraj_dt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        loss, losses, traj_loss = loss_function(traj, dtraj_dt, time_input, alpha)

        print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)}, T: {sim_params.ntime}, Total time steps: {total_time_steps}\n")
        wandb.log({'epoch': iteration+1, 'loss': loss.item(), 'traj_loss': traj_loss.item()})
        loss.backward()
        optimizer.step()

        # if iteration % 10 == 0:
        #     with torch.no_grad():
        #         plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)

        # plt.pause(0.01)

        iteration += 1
        total_time_steps += sim_params.ntime

    plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)
    plt.tick_params(labelsize=12)

    # Save plot as high definition PNG
    plt.savefig(f"{log}.png", dpi=300, bbox_inches='tight')
    plt.show()

    torch.save(dyn_system.value_func.to('cpu').state_dict(), f'{log}.pt')  # Export to TorchScript
