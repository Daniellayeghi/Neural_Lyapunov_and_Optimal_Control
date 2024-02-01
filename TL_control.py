import random
from models import TwoLink2, ModelParams
from neural_value_synthesis_diffeq import *
from utilities.mujoco_torch import SimulationParams
from time_search import optimal_time
from torchdiffeq_ctrl import odeint_adjoint as ctrl_odeint
from utilities.mj_renderer import *
import wandb
import argparse

parser = argparse.ArgumentParser(description='Seed input')
parser.add_argument('--seed', type=int, default=4, help='Random seed')
args = parser.parse_args()
seed = args.seed

torch.manual_seed(seed)
wandb.init(project='TL_control', anonymous='allow')

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 12, 160, 0.01)
tl_params = ModelParams(2, 2, 1, 4, 4)
max_iter, max_time, alpha, dt, discount, step, scale, mode = 61, 171, .5, 0.01, 1.0, 15, 1.5, 'fwd'
Q = torch.diag(torch.Tensor([1, 1, 0, 0])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([500, 500, 5, 5])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-0, sim_params.nsim, 1, 1))
tl = TwoLink2(sim_params.nsim, tl_params, device, mode='fwd')
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
        nsim = x.shape[0]
        if len(t.shape) == 0:
            t = torch.ones((nsim, 1, 1)).to(x.device) * t
        aug_x = torch.cat((x, t), dim=2)
        return self.nn(aug_x)


def loss_quadratic(x, gain):
    return x @ gain @ x.mT

def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return loss_quadratic(x, Q)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def value_diff_loss(x: torch.Tensor, time):
    x_w = batch_state_encoder(x)
    x_w = x_w.reshape(x_w.shape[0]*x_w.shape[1], x_w.shape[2],  x_w.shape[3])
    time = time.reshape(time.shape[0]*time.shape[1], time.shape[2],  time.shape[3])
    values = nn_value_func(time, x_w).squeeze().reshape(x.shape[0], x.shape[1])
    value_differences = values[1:] - values[:-1]
    return value_differences.squeeze()


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_init = x[0].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func((sim_params.ntime - 1) * dt, x_init_w).squeeze()

    return (-value_init + value_final).squeeze()


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-10].view(t-10, nsim, r, c).clone()
    x_final = x[-10:].view(10, nsim, r, c).clone()
    l_running = loss_quadratic(x_run, Q).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal), dim=0)


def batch_inv_dynamics_loss(x, acc, alpha):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = tl._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    Tf = tl._Tfric(v).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    C = tl._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = tl._Bvec() @ ((M @ acc.mT).mT - C + Tf).mT
    return (u_batch.mT @ torch.linalg.inv(M) @ u_batch / scale).squeeze()


def value_terminal_loss(x: torch.Tensor, time=0):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    value_final = nn_value_func(torch.tensor(time), x_final_w).squeeze()
    return value_final


def loss_function(x, xd, batch_time, alpha=1):
    x_running, acc_running = x[:-1].clone(), xd[:-1, ..., sim_params.nv:].clone()
    l_run_ctrl = batch_inv_dynamics_loss(x_running, acc_running, alpha) * 1
    l_run_state = batch_state_loss(x_running)
    l_value_diff = value_diff_loss(x, batch_time)
    l_backup = torch.square((l_run_state + l_run_ctrl)*dt + l_value_diff)
    l_backup = torch.sum(l_backup, dim=0)
    l_terminal = 0 * torch.square(value_terminal_loss(x))
    return torch.mean(l_backup + l_terminal), l_backup + l_terminal, torch.mean(torch.sum((l_run_state + l_run_ctrl), dim=0)).squeeze()

init_lr = 4e-3
dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=tl, mode=mode, step=step, scale=scale
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device).requires_grad_(True)
time_input = time.clone().reshape(time.shape[0], 1, 1, 1).repeat(1, sim_params.nsim, 1, 1).requires_grad_(True)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=init_lr, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

log = f"TWOLINK_TO_m-{mode}_d-{discount}_s-{step}_seed{seed}"
wandb.watch(dyn_system, loss_function, log="all")
total_time_steps = 0

if __name__ == "__main__":

    def transform_coordinates_tl(traj: torch.Tensor):
       traj[..., 1] = torch.pi - (traj[..., 0] + (torch.pi - traj[..., 1]))
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
        q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-torch.pi, torch.pi)
        qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1) * 0
        x_init = torch.cat((q_init, qd_init), 2).to(device)

        optimizer.zero_grad()
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj, dtrj_dt = ctrl_odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        loss, losses, traj_loss = loss_function(traj, dtrj_dt, time_input, alpha)
        loss.backward()
        sim_params.ntime, update = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)
        total_time_steps += sim_params.ntime

        if update:
            time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device).requires_grad_(True)
            time_input = time.clone().reshape(time.shape[0], 1, 1, 1).repeat(1, sim_params.nsim, 1, 1).requires_grad_(True)
        print(time[-1].item())
        print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)},"
              f" T: {sim_params.ntime}, Total time steps: {total_time_steps}, Update: {update}\n")
        wandb.log({'epoch': iteration+1, 'loss': loss.item(), 'traj_loss': traj_loss.item()})

        optimizer.step()
        if iteration == max_iter-1:
            _, indices = torch.topk(losses, 10, largest=False)
            for i in indices:
                traj_tl_mj = transform_coordinates_tl(traj.clone())
                renderer.render(traj_tl_mj[:, i, 0, :tl_params.nq].cpu().detach().numpy())

        iteration += 1
    torch.save(dyn_system.value_func.to('cpu').state_dict(), f'{log}.pt')