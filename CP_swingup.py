import random

import torch.optim.lr_scheduler as lr_scheduler
from models import Cartpole, ModelParams
from neural_value_synthesis_diffeq import *
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from time_search import optimal_time
import wandb
from mj_renderer import *
import argparse

parser = argparse.ArgumentParser(description='Seed input')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
seed = args.seed

wandb.init(project='CP_swingup', anonymous="allow")
torch.manual_seed(seed)
sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 200, 240, 0.01)
cp_params = ModelParams(2, 2, 1, 4, 4)
max_iter, max_time, alpha, dt, n_bins, discount, step, scale, mode = 150, 241, .5, 0.01, 3, 1, 15, 10, 'fwd'
Q = torch.diag(torch.Tensor([.05, 5, .1, .1])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([5, 300, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-2, sim_params.nsim, 1, 1))
cartpole = Cartpole(sim_params.nsim, cp_params, device)
renderer = MjRenderer("./xmls/cartpole.xml", 0.0001)

def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = torch.cos(qp) - 1
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = torch.cos(qp) - 1
    return torch.cat((qc, qp, v), 1).reshape((t, b, r, c))


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in+1, 128),
            nn.Softplus(beta=5),
            nn.Linear(128, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        nsim = x.shape[0]
        time = torch.ones((nsim, 1, 1)).to(x.device) * t
        aug_x = torch.cat((x, time), dim=2)
        return self.nn(aug_x)


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def loss_exp(x, gain):
    return 1 - torch.exp(-0.5 * loss_quadratic(x, gain))


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return loss_quadratic(x, Q)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    x_init = x[0, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func((sim_params.ntime - 1) * dt, x_init_w).squeeze()

    return -value_init + value_final


def batch_dynamics_loss(x, acc, alpha=1):
    t, b, r, c = x.shape
    x_reshape = x.reshape((t*b, 1, sim_params.nqv))
    a_reshape = acc.reshape((t*b, 1, sim_params.nv))
    acc_real = cartpole(x_reshape, a_reshape).reshape(x.shape)[:, :, :, sim_params.nv:]
    l_run = torch.sum((acc - acc_real)**2, 0).squeeze()
    return torch.mean(l_run) * alpha


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-1, :, :, :].view(t-1, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = torch.sum(loss_quadratic(x_run, Q), 0).squeeze()
    l_terminal = (loss_quadratic(x_final, Qf)).squeeze()

    return torch.mean(l_running + l_terminal)

def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def batch_inv_dynamics_loss(x, acc, alpha):
    x, acc = x[:-1].clone(), acc[:-1, :, :, sim_params.nv:].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT - C
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale


def loss_function(x, acc, alpha=1):
    l_ctrl, l_state, l_bellman = batch_inv_dynamics_loss(x, acc, alpha), batch_state_loss(x), backup_loss(x)
    return torch.mean(torch.square(l_ctrl + l_state + l_bellman))


dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params,
    encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale, R=R
).to(device)

time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=4e-3, amsgrad=True)
lr = lr_scheduler.MultiStepLR(optimizer, milestones=[120], gamma=0.1)
lambdas = build_discounts(lambdas, discount).to(device)


log = f"fwd_CP_TO_m-{mode}_d-{discount}_s-{step}_seed_{seed}"
wandb.watch(dyn_system, loss_function, log="all")


if __name__ == "__main__":
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0) * 2
    qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(torch.pi - 0.6, torch.pi + 0.6)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1)
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device).requires_grad_(True)
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj, dtraj_dt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        loss = loss_function(traj, dtraj_dt, alpha)
        loss.backward()
        sim_params.ntime, _ = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)
        optimizer.step()
        lr.step(iteration)
        wandb.log({'epoch': iteration+1, 'loss': loss.item()})

        print(f"Epochs: {iteration}, Loss: {loss.item()} \n")

        if iteration % 25 == 0:
            for i in range(0, sim_params.nsim, 30):
                selection = random.randint(0, sim_params.nsim - 1)
                renderer.render(traj[:, selection, 0, :sim_params.nq].cpu().detach().numpy())

        iteration += 1

    model_scripted = torch.jit.script(dyn_system.value_func.to('cpu'))  # Export to TorchScript
    model_scripted.save(f'{log}.pt')  # Save
