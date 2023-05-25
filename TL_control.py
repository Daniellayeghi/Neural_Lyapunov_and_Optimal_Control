import random
from models import TwoLink2, ModelParams
from neural_value_synthesis_diffeq import *
from utilities.mujoco_torch import SimulationParams
from time_search import optimal_time
from torchdiffeq_ctrl import odeint_adjoint as ctrl_odeint
from mj_renderer import *
import wandb
import argparse

parser = argparse.ArgumentParser(description='Seed input')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
seed = args.seed

torch.manual_seed(seed)
wandb.init(project='TL_control', anonymous='allow')

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 20, 290, 0.01)
tl_params = ModelParams(2, 2, 1, 4, 4)
max_iter, max_time, alpha, dt, discount, step, scale, mode = 51, 300, .5, 0.01, 1.0, 15, 1, 'fwd'
Q = torch.diag(torch.Tensor([1, 1, 0, 0])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([10000, 10000, 100, 100])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-0, sim_params.nsim, 1, 1))
tl = TwoLink2(sim_params.nsim, tl_params, device)
renderer = MjRenderer("./xmls/reacher.xml", dt=0.000001)

def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i] *= (discount)**i

    return lambdas.clone()


def atan2_encoder(x: torch.Tensor):
    return torch.atan2(torch.sin(x), torch.cos(x))

def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    q, v = x[:, :sim_params.nq].clone().unsqueeze(1), x[:, sim_params.nq:].clone().unsqueeze(1)
    q = atan2_encoder(q)
    return torch.cat((q, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    q, v = x[:, :sim_params.nq].clone().unsqueeze(1), x[:, sim_params.nq:].clone().unsqueeze(1)
    q = atan2_encoder(q)
    return torch.cat((q, v), 1).reshape((t, b, r, c))


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
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        time = torch.ones((sim_params.nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2)
        return self.nn(aug_x)


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return loss_quadratic(x, Q)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_init = x[0].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func((t - 1) * dt, x_init_w).squeeze()

    return -value_init + value_final


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-1].view(t-1, nsim, r, c).clone()
    x_final = x[-1].view(1, nsim, r, c).clone()
    l_running = loss_quadratic(x_run, Q).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal.unsqueeze(0)), dim=0)


def batch_inv_dynamics_loss(x, acc, alpha):
    x, acc = x[:-1].clone(), acc[:-1, :, :, sim_params.nv:].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = tl._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = tl._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT - C
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale


def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    return value_final


def loss_function(x, acc, alpha=1):
    l_run = torch.sum(batch_inv_dynamics_loss(x, acc, alpha) + batch_state_loss(x), dim=0)
    l_bellman = backup_loss(x)
    l_terminal = 1000 * value_terminal_loss(x)
    return torch.mean(torch.square(l_run + l_bellman + l_terminal))

init_lr = 8e-2
dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=tl, mode=mode, step=step, scale=scale
).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=init_lr, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)


log = f"TWOLINK_TO_m-{mode}_d-{discount}_s-{step}_seed{seed}"
wandb.watch(dyn_system, loss_function, log="all")


if __name__ == "__main__":

    def transform_coordinates_tl(traj: torch.Tensor):
       traj[:, :, :, 1] = torch.pi - (traj[:, :, :, 0] + (torch.pi - traj[:, :, :, 1]))
       return traj

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-torch.pi, torch.pi)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1) * 0
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device).requires_grad_(True)
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj, dtrj_dt = ctrl_odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        acc = dtrj_dt[:, :, :, sim_params.nv:]
        loss = loss_function(traj, dtrj_dt, alpha)
        loss.backward()
        sim_params.ntime, _ = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)
        optimizer.step()
        wandb.log({'epoch': iteration+1, 'loss': loss.item()})


        print(f"Epochs: {iteration}, Loss: {loss.item()}\n")

        if iteration % 25 == 0:
            for i in range(0, sim_params.nsim, 5):
                selection = random.randint(0, sim_params.nsim - 1)
                traj_tl_mj = transform_coordinates_tl(traj.clone())
                renderer.render(traj_tl_mj[:, selection, 0, :tl_params.nq].cpu().detach().numpy())

        iteration += 1

    torch.save(dyn_system.value_func.to('cpu').state_dict(), f'{log}.pt')